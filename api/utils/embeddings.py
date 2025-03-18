from sqlalchemy.orm import Session
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from api.v1.models.embeddings import PaperEmbedding
import os
import aiohttp
import httpx
import asyncio
from endpoint_tester import predict_custom_trained_model
from dotenv import load_dotenv
from google.auth import load_credentials_from_file, default
import io
import docx
import chardet
from api.db.database import get_db  # Ensure this function is correctly implemented
from playwright.async_api import async_playwright

db = next(get_db())  # Get a database session

load_dotenv(".env")

MAX_TOKENS = 255  # Vertex AI limit
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

try:
    if GOOGLE_APPLICATION_CREDENTIALS:
        credentials, project = load_credentials_from_file(
            GOOGLE_APPLICATION_CREDENTIALS
        )
    else:
        credentials, project = default()
        print("✅ Using default Google Cloud credentials")
except Exception as e:
    print(f"🚨 ERROR: Failed to load Google credentials → {e}")
    exit(1)

GCP_PROJECT = os.getenv("GCP_PROJECT")
VERTEX_ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION")
FAISS_INDEX_PATH = "faiss_index.bin"

def chunk_text(text, chunk_size=1000):
    """Split text into smaller chunks based on character length."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# Load embedding model
embedding_dim = 384  # Make sure this matches the model output


def rebuild_faiss_index(db: Session):
    """Rebuild FAISS index from database embeddings"""
    print("🔄 Rebuilding FAISS index...")

    # ✅ Step 1: Fetch all embeddings from the database
    papers = db.query(PaperEmbedding.id, PaperEmbedding.embedding).all()
    print(f"📌 DEBUG: Retrieved {len(papers)} embeddings from DB")

    if not papers:
        print("⚠️ No embeddings found in database. FAISS index will be empty.")
        return None

    # ✅ Step 2: Convert to NumPy arrays
    ids = np.array([paper.id for paper in papers], dtype=np.int64)  # Ensure int64
    embeddings = np.array([paper.embedding for paper in papers], dtype=np.float32)
    print(f"📌 DEBUG: NumPy array shape: {embeddings.shape}")

    # ✅ Step 3: Create an ID-mapped FAISS index
    dimension = embeddings.shape[1]
    base_index = faiss.IndexFlatL2(dimension)  # Base index
    index = faiss.IndexIDMap(base_index)  # ✅ Wrap with IndexIDMap

    # ✅ Step 4: Add embeddings with IDs
    index.add_with_ids(embeddings, ids)
    print(f"✅ FAISS index built with {index.ntotal} vectors")

    # ✅ Step 5: Save index to disk
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved with {index.ntotal} embeddings.")

    return index


# ✅ Load FAISS Index with Proper ID Handling
def load_faiss_index():
    """Load FAISS index from disk or rebuild it from the database if not found."""
    global index

    if os.path.exists(FAISS_INDEX_PATH):
        try:
            print("📂 Loading FAISS Index from disk...")
            index = faiss.read_index(FAISS_INDEX_PATH)

            # ✅ Ensure the index is properly wrapped
            if not isinstance(index, faiss.IndexIDMap):
                print("⚠️ Index is not ID-mapped. Wrapping it in IndexIDMap...")
                index = faiss.IndexIDMap(index)

            print(f"📌 DEBUG: FAISS index total vectors after loading: {index.ntotal}")

        except Exception as e:
            print(f"🚨 ERROR: Failed to load FAISS index, reinitializing → {e}")
            db = next(get_db())  # Get a database session
            index = rebuild_faiss_index(db) or faiss.IndexIDMap(faiss.IndexFlatL2(384))

    else:
        print("🚀 No saved FAISS index found. Rebuilding from database...")
        db = next(get_db())  # Get a database session
        index = rebuild_faiss_index(db) or faiss.IndexIDMap(faiss.IndexFlatL2(384))

    print(
        f"📌 DEBUG: FAISS index total vectors after loading/rebuilding: {index.ntotal}"
    )
    return index


# Save FAISS index
def save_faiss_index():
    """Save FAISS index to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)

# Initialize FAISS
index = None
load_faiss_index()

# Embeddings
async def fetch_embedding_from_vertex(text: str):
    """Fetch embeddings for text, splitting into batches if necessary."""

    text_chunks = chunk_text(text)  # Split into chunks

    embeddings = []
    for chunk in text_chunks:
        instances = {"text": chunk}  # Send each chunk separately

        try:
            response = predict_custom_trained_model(
                project=GCP_PROJECT,
                endpoint_id=VERTEX_ENDPOINT_ID,
                instances=instances,
                location=GCP_LOCATION,
            )

            if response:
                embedding = np.array(response[0], dtype=np.float32)

                # Ensure the embedding has the correct shape (384,) instead of (1, 384)
                if len(embedding.shape) == 2:
                    embedding = embedding.squeeze(0)  # Converts (1, 384) → (384,)

                embeddings.append(embedding)

        except Exception as e:
            print(f"🚨 ERROR: Vertex AI request failed → {e}")
            return None

    if embeddings:
        return np.mean(embeddings, axis=0)  # Take the mean to combine embeddings

    return None  # Return None if no embeddings were retrieved


async def process_and_store_paper_embeddings(papers, db: Session):
    """Generate embeddings via Vertex AI, store in DB, and update FAISS."""

    if not papers:
        print("⚠️ DEBUG: No papers provided for embedding.")
        return

    print(f"🔍 DEBUG: Processing {len(papers)} papers for embedding.")

    for paper in papers:
        title = paper.get("title", "Unknown Title")
        abstract = paper.get("abstract", "").strip()
        if not abstract:
            print(f"⚠️ DEBUG: Skipping {title} (no abstract available).")
            continue

        print(f"🛠️ DEBUG: Fetching embedding for {title} from Vertex AI...")

        # 1️⃣ Generate key findings
        key_findings = abstract.split(". ")[:2]

        # 2️⃣ Fetch embedding from Vertex AI
        embedding = await fetch_embedding_from_vertex(abstract)

        if embedding is None:
            print(f"❌ DEBUG: No embedding received for {title}")
        elif not hasattr(embedding, "shape"):
            print(f"❌ DEBUG: Embedding received but not a NumPy array: {type(embedding)}, value: {embedding}")
        elif embedding.shape[0] != embedding_dim:
            print(f"❌ DEBUG: Incorrect embedding shape {embedding.shape}, expected ({embedding_dim},)")
        else:
            print(f"✅ DEBUG: Successfully fetched embedding for {title}, shape: {embedding.shape}")
        if embedding is None or embedding.shape[0] != embedding_dim:
            print(f"❌ DEBUG: Embedding error for {title}")
            continue

        # 3️⃣ Store in DB
        try:
            new_paper = PaperEmbedding(
                title=title,
                abstract=abstract,
                key_findings=key_findings,
                link=paper.get("link", ""),
                embedding=embedding.tolist(),
            )
            db.add(new_paper)
            db.flush()  # Get ID before commit
            db.commit()

            # 4️⃣ Update FAISS index
            paper_id = new_paper.id
            index.add_with_ids(
                np.array([embedding], dtype=np.float32),
                np.array([paper_id], dtype=np.int64),
            )

            print(
                f"✅ DEBUG: Successfully stored embedding for {title} (ID: {paper_id})"
            )

        except Exception as e:
            print(f"🚨 ERROR: Failed to store embedding for {title} → {e}")
            continue

    print("💾 DEBUG: Database commit complete.")
    save_faiss_index()  # Save FAISS after update


# Text extraction

async def extract_text_from_webpage_async(url: str, max_retries=3) -> str:
    """Asynchronously extract text from a webpage, handling JS-heavy sites & captchas."""
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        print(
                            f"⚠️ HTTP {response.status} - Retrying... ({attempt+1}/{max_retries})"
                        )
                        await asyncio.sleep(2 * attempt)  # Exponential backoff
                        continue

                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, "html.parser")

                    # ✅ CAPTCHA DETECTION: If page says "Please enable JavaScript", it's likely a bot trap
                    if (
                        "Please enable JavaScript" in html_content
                        or "support ID" in html_content
                    ):
                        print(f"⚠️ CAPTCHA detected on {url}. Skipping...")
                        return ""

                    # ✅ REMOVE UNNECESSARY ELEMENTS (script, style, ads, navigation, etc.)
                    for tag in soup(
                        ["script", "style", "header", "footer", "nav", "aside", "form"]
                    ):
                        tag.extract()

                    # ✅ TARGET SPECIFIC ABSTRACT SECTIONS
                    abstract = None

                    # Check common places for abstracts
                    meta_description = soup.find("meta", attrs={"name": "description"})
                    if meta_description:
                        abstract = meta_description.get("content", "").strip()

                    elif soup.find("p"):  # Try first paragraph as a fallback
                        abstract = soup.find("p").get_text(strip=True)

                    if abstract:
                        return abstract

                    # ❗ If no useful text was found, consider using Playwright as a fallback
                    print(f"⚠️ No useful text extracted. Trying JavaScript rendering...")
                    return await extract_text_with_playwright(url)

        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionResetError) as e:
            print(f"⚠️ Error fetching {url}: {e} - Retrying ({attempt+1}/{max_retries})")
            await asyncio.sleep(2 * attempt)  # Exponential backoff

    print(f"❌ Error extracting text from {url} after {max_retries} attempts.")
    return ""


# ✅ PLAYWRIGHT FALLBACK FOR JAVASCRIPT-PROTECTED PAGES
async def extract_text_with_playwright(url: str) -> str:
    """Use Playwright to render JavaScript-heavy pages and extract text."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(url, timeout=30000)  # Wait for full page load
            html_content = await page.content()  # Get rendered HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove unnecessary elements
            for tag in soup(
                ["script", "style", "header", "footer", "nav", "aside", "form"]
            ):
                tag.extract()

            # Extract abstract
            abstract = None
            meta_description = soup.find("meta", attrs={"name": "description"})
            if meta_description:
                abstract = meta_description.get("content", "").strip()
            elif soup.find("p"):
                abstract = soup.find("p").get_text(strip=True)

            await browser.close()
            return abstract if abstract else ""

        except Exception as e:
            print(f"🚨 Playwright failed on {url}: {e}")
            await browser.close()
            return ""


def extract_text_from_pdf(pdf_url: str) -> str:
    """Extract text from a given PDF link without saving to disk, handling encoding issues."""
    try:
        response = requests.get(pdf_url, stream=True, timeout=10)
        response.raise_for_status()

        # Convert content to a file-like BytesIO object
        pdf_stream = io.BytesIO(response.content)

        text = ""
        with fitz.open(pdf_stream) as doc:  # Open PDF from BytesIO
            for page in doc:
                text += page.get_text("text") + "\n"

        # Detect encoding and decode properly
        detected = chardet.detect(text.encode())
        encoding = detected.get("encoding", "utf-8")

        return text.encode(encoding, errors="replace").decode("utf-8").strip()

    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""


async def extract_text_from_pdf_async(pdf_url: str) -> str:
    """Extracts text from a given PDF link asynchronously."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url, timeout=10)
            response.raise_for_status()

        return await asyncio.to_thread(extract_text_from_pdf, response.content)
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""


def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF or DOCX file."""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text("text") for page in doc])
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

