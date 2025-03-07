from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from api.v1.models.embeddings import PaperEmbedding
import os

FAISS_INDEX_PATH = "faiss_index.bin"

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding_dim = 768  # Keep it consistent

# Global FAISS index
index = faiss.IndexHNSWFlat(embedding_dim, 32)
index.hnsw.efConstruction = 200  # Controls indexing speed/accuracy
index.hnsw.efSearch = 64  # Controls search speed/accuracy


def preload_external_papers(db: Session, external_papers: list):
    """Preload FAISS index with external research papers."""
    entries = []
    for paper in external_papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        key_findings = paper.get("key_findings", [])
        link = paper.get("link", "")

        full_text = f"{title}. {abstract} {' '.join(key_findings)}"
        embedding = embed_model.encode(full_text, normalize_embeddings=True).tolist()

        entries.append(PaperEmbedding(
            title=title, abstract=abstract, key_findings=key_findings, link=link, embedding=embedding
        ))

    if entries:
        db.add_all(entries)
        db.commit()
        np_embeddings = np.array([e.embedding for e in entries], dtype=np.float32)
        index.add(np_embeddings)

    print(f"✅ Preloaded {len(entries)} papers into FAISS.")


def fetch_external_papers():
    """Fetch research papers from ArXiv, Google, and SERP API."""
    query = "artificial intelligence in education"
    url = f"http://127.0.0.1:8000/api/v1/research/search/?query={query}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        papers = []
        
        for source in ["arxiv", "serpapi"]:
            for paper in data["results"].get(source, []):
                papers.append({
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),  # Ensure this is captured
                    "key_findings": paper.get("key_findings", []),
                    "link": paper.get("link", "")
                })
        return papers
    else:
        print("⚠️ Failed to fetch external papers.")
        return []

def save_faiss_index():
    """Save FAISS index to disk."""
    if index and index.ntotal > 0:
        faiss.write_index(index, FAISS_INDEX_PATH)
        print("💾 FAISS index saved to disk.")
        
def load_faiss_index(db: Session):
    """Load FAISS index from disk or populate it from the database."""
    global index

    # Try loading from disk
    if os.path.exists(FAISS_INDEX_PATH):
        print("📂 Loading FAISS index from disk...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        return

    # Otherwise, rebuild from DB
    print("⚠️ No saved FAISS index found. Rebuilding from DB...")
    papers = db.query(PaperEmbedding).all()
    if not papers:
        print("⚠️ No embeddings found in DB.")
        return  

    embeddings = np.array([p.embedding for p in papers], dtype=np.float32)

    if embeddings.shape[0] > 0:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        save_faiss_index()  # Save index after loading



def extract_text_from_webpage(url: str) -> str:
    """Extracts text content from a webpage."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.extract()
        return soup.get_text(separator="\n").strip()
    
    return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a given PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()


def search_similar_papers(query: str, db: Session, top_k: int = 3):
    """Find similar research papers using FAISS."""
    query_embedding = np.array(embed_model.encode(query)).reshape(1, -1)

    # Ensure FAISS is loaded
    if index.ntotal == 0:
        print("⚠️ FAISS index is empty. Loading from DB...")
        load_faiss_index(db)

    if index.ntotal > 0:
        distances, indices = index.search(query_embedding, top_k)
        papers = db.query(PaperEmbedding).all()
        results = [{"title": papers[i].title, "abstract": papers[i].abstract} for i in indices[0]]
        return results
    else:
        return []
