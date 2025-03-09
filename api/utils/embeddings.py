from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from api.v1.models.embeddings import PaperEmbedding
import os
from transformers import pipeline

FAISS_INDEX_PATH = "faiss_index.bin"

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding_dim = 768  # Make sure this matches the model output

# Load FAISS index
def load_faiss_index():
    """Load FAISS index from disk or initialize a new one."""
    global index
    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading FAISS index from disk...")
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        print("🚀 No saved FAISS index found. Creating a new one...")
        index = faiss.IndexHNSWFlat(embedding_dim, 32)

# Save FAISS index
def save_faiss_index():
    """Save FAISS index to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)

# Initialize FAISS
index = None
load_faiss_index()

async def process_and_store_paper_embeddings(papers, db: Session):
    """Generate embeddings, store in DB, and update FAISS."""
    
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

        print(f"🛠️ DEBUG: Generating embedding for {title}...")

        # 1️⃣ Generate key findings
        key_findings = abstract.split(". ")[:2]

        # 2️⃣ Generate embedding
        try:
            embedding = embed_model.encode(abstract).tolist()
            if not embedding:
                print(f"❌ DEBUG: Failed to generate embedding for {title}")
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
                embedding=embedding
            )
            db.add(new_paper)
            db.flush()  # Get ID before commit

            # 4️⃣ Update FAISS index
            paper_id = new_paper.id  # Ensure ID is fetched
            index.add(np.array([embedding], dtype=np.float32), np.array([paper_id], dtype=np.int64))

            print(f"✅ DEBUG: Successfully stored embedding for {title} (ID: {paper_id})")

        except Exception as e:
            print(f"🚨 ERROR: Failed to store embedding for {title} → {e}")
            continue

    db.commit()
    print("💾 DEBUG: Database commit complete.")
    save_faiss_index()  # Save after update

def extract_text_from_webpage(url: str) -> str:
    """Extract text content from a webpage."""
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.extract()
        return soup.get_text(separator="\n").strip()
    
    return ""

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
    index = faiss.IndexHNSWFlat(embedding_dim, 32)  # Reset index

    embeddings = db.query(PaperEmbedding).all()

    vectors = []
    ids = []
    for paper in embeddings:
        vectors.append(np.array(paper.embedding, dtype=np.float32))
        ids.append(paper.id)

    if vectors:
        index.add(np.vstack(vectors), np.array(ids, dtype=np.int64))
        save_faiss_index()

def add_embedding_to_faiss(embedding: list, paper_id: int):
    """Add a new embedding to the FAISS index."""
    global index
    vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
    index.add(vector, np.array([paper_id], dtype=np.int64))
    save_faiss_index()
