from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from api.v1.models.embeddings import PaperEmbedding
import os
from requests_html import AsyncHTMLSession
from typing import Dict, List, Union
from google.cloud import aiplatform
import asyncio
from dotenv import load_dotenv
from google.auth import load_credentials_from_file, default

load_dotenv(".env")
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

FAISS_INDEX_PATH = "faiss_index.bin"

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding_dim = 768  # Make sure this matches the model output


# Load FAISS index
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

# Google Cloud AI Model Prediction
async def predict_custom_trained_model(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """Sends a prediction request to a Vertex AI custom model."""
    
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    instances = instances if isinstance(instances, list) else [instances]
    formatted_instances = [{"inputs": instance["text"]} for instance in instances]

    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    try:
        response = client.predict(endpoint=endpoint, instances=formatted_instances, parameters={})
        predictions = response.predictions

        if not predictions:
            print("🚨 ERROR: No predictions received.")
            return None

        return predictions  # Return raw predictions

    except Exception as e:
        print(f"🚨 ERROR: Failed to fetch embeddings → {e}")
        return None

# Function to Process Papers and Store Embeddings
async def process_and_store_paper_embeddings(papers, db: Session):
    """Fetch embeddings from the hosted model, store in DB, and update FAISS."""

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

        print(f"🛠️ DEBUG: Generating embedding for {title} using the hosted model...")

        # 1️⃣ Generate key findings
        key_findings = abstract.split(". ")[:2]

        # 2️⃣ Fetch embedding from API
        try:
            embedding_response = await predict_custom_trained_model(
                project="178740183321",
                endpoint_id="3947058727227490304",
                location="us-central1",
                instances={"text": abstract}
            )

            if not embedding_response:
                print(f"❌ DEBUG: No embedding received for {title}")
                continue

            # Extract the embedding vector from response
            embedding = np.array(embedding_response[0], dtype=np.float32)  # Assuming first response is embedding

            if embedding.shape[0] != embedding_dim:
                print(f"❌ DEBUG: Embedding dimension mismatch for {title}")
                continue

        except Exception as e:
            print(f"🚨 ERROR: Embedding generation failed for {title} → {e}")
            continue

        # 3️⃣ Store in DB
        try:
            new_paper = PaperEmbedding(
                title=title,
                abstract=abstract,
                key_findings=key_findings,
                link=paper.get("link", ""),
                embedding=embedding.tolist()
            )
            db.add(new_paper)
            db.flush()  # Get ID before commit
            db.commit()

            # 4️⃣ Update FAISS index
            paper_id = new_paper.id  # Ensure ID is fetched
            index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([paper_id], dtype=np.int64))

            print(f"✅ DEBUG: Successfully stored embedding for {title} (ID: {paper_id})")

        except Exception as e:
            print(f"🚨 ERROR: Failed to store embedding for {title} → {e}")
            continue

    print("💾 DEBUG: Database commit complete.")
    save_faiss_index()  # Save after update

async def extract_text_from_webpage_async(url: str) -> str:
    """Asynchronously extract text content from a webpage, including JavaScript-rendered content."""
    session = AsyncHTMLSession()
    
    try:
        response = await session.get(url, timeout=15)  # Increased timeout
        await response.html.arender(timeout=20, sleep=2)  # Render JavaScript with delay

        soup = BeautifulSoup(response.html.html, "html.parser")

        # Remove unnecessary elements
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
            tag.extract()

        return soup.get_text(separator="\n").strip()

    except Exception as e:
        print(f"❌ Error extracting text from {url}: {e}")
        return ""

    finally:
        await session.close()  # Ensures session is closed

def extract_text_from_pdf(pdf_url: str) -> str:
    """Extract text from a given PDF link."""
    response = requests.get(pdf_url, stream=True, timeout=10)
    if response.status_code == 200:
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        
        text = ""
        with fitz.open("temp.pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text.strip()
    
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
