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

load_dotenv(".env")

GCP_PROJECT = os.getenv("GCP_PROJECT")
VERTEX_ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION")


FAISS_INDEX_PATH = "faiss_index.bin"

# Load embedding model
embedding_dim = 768  # Make sure this matches the model output


# Load FAISS index
def load_faiss_index():
    """Load FAISS index from disk or initialize a new one."""
    global index
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            print("Loading FAISS Index from the disk...")
            index = faiss.read_index(FAISS_INDEX_PATH)

            # Ensure the index is wrapped properly
            if not isinstance(index, faiss.IndexIDMap):
                print("⚠️ Index is not ID-mapped. Wrapping it in IndexIDMap...")
                index = faiss.IndexIDMap(index)

        except Exception as e:
            print(f"🚨 ERROR: Failed to load FAISS index, reinitializing → {e}")
            base_index = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIDMap(base_index)
            save_faiss_index()  # Ensure the new index is saved

    else:
        print("🚀 No saved FAISS index found. Creating a new one...")
        base_index = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIDMap(base_index)


# Save FAISS index
def save_faiss_index():
    """Save FAISS index to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)

# Initialize FAISS
index = None
load_faiss_index()


async def fetch_embedding_from_vertex(text: str):
    """Fetch embedding from Vertex AI using existing function."""
    instances = {"text": text}
    response = predict_custom_trained_model(
        project=GCP_PROJECT,
        endpoint_id=VERTEX_ENDPOINT_ID,
        instances=instances,
        location=GCP_LOCATION,
    )

    if response:
        return np.array(response[0], dtype=np.float32)  # Ensure correct format
    return None


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


async def extract_text_from_webpage_async(url: str) -> str:
    """Asynchronously extract text content from a webpage efficiently."""
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                html_content = await response.text()

                soup = BeautifulSoup(html_content, "html.parser")

                # Remove unnecessary elements
                for tag in soup(
                    ["script", "style", "header", "footer", "nav", "aside", "form"]
                ):
                    tag.extract()

                return soup.get_text(separator="\n").strip()
        except Exception as e:
            print(f"❌ Error extracting text from {url}: {e}")
            return ""


def extract_text_from_pdf(pdf_url: str) -> str:
    """Extract text from a given PDF link without saving to disk."""
    try:
        response = requests.get(pdf_url, stream=True, timeout=10)
        response.raise_for_status()

        text = ""
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"

        return text.strip()
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


def rebuild_faiss_index(db: Session):
    """Rebuild FAISS index from the database if necessary."""
    global index
    index = faiss.IndexFlatIP(embedding_dim)  # Reset index with correct similarity metric

    embeddings = db.query(PaperEmbedding).all()

    vectors = []
    ids = []
    for paper in embeddings:
        vectors.append(np.array(paper.embedding, dtype=np.float32))
        ids.append(paper.id)

    if vectors:
        index.add_with_ids(np.vstack(vectors), np.array(ids, dtype=np.int64))
        save_faiss_index()
