import httpx
import os
import asyncio
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
import shutil
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from api.db.database import get_db
from api.v1.models.cache import CachedResult
from api.utils.embeddings import (
    process_and_store_paper_embeddings,
    extract_text_from_pdf_async,
    extract_text_from_webpage_async,
    extract_text_from_file,
    rebuild_faiss_index,
    chunk_text
)
from api.utils.embeddings import index, fetch_embedding_from_vertex
from api.v1.models.embeddings import PaperEmbedding
import numpy as np
import re
from xml.etree import ElementTree as ET
import faiss
load_dotenv()


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Validate API keys
if not GOOGLE_API_KEY:
    print("⚠️ Missing GOOGLE_API_KEY.")
if not GOOGLE_CX:
    print("⚠️ Missing GOOGLE_CX (Custom Search Engine ID).")
if not SERP_API_KEY:
    print("⚠️ Missing SERP_API_KEY.")

research = APIRouter(prefix="/research", tags=["Research"])


# Paper Search Functions
async def fetch_arxiv_results(query: str, max_results=5):
    """Fetches research papers from ArXiv, including abstracts and metadata."""
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            return [
                {
                    "title": entry.find(
                        "{http://www.w3.org/2005/Atom}title"
                    ).text.strip(),
                    "abstract": entry.find(
                        "{http://www.w3.org/2005/Atom}summary"
                    ).text.strip(),
                    "link": entry.find("{http://www.w3.org/2005/Atom}id").text.strip(),
                }
                for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry")
            ]
        except Exception as e:
            print(f"❌ Error fetching ArXiv results: {e}")
            return []


async def fetch_google_results(query: str, max_results=5):
    """Fetches results from Google Custom Search API and extracts abstracts."""
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []

    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ Error fetching Google results: {e}")
            return []

    results = []

    async def process_result(item):
        """Processes each search result asynchronously."""
        title = item.get("title", "Untitled")
        link = item.get("link", "")

        abstract = await (
            extract_text_from_pdf_async(link)
            if link.endswith(".pdf")
            else extract_text_from_webpage_async(link)
        )
        return {"title": title, "link": link, "abstract": abstract[:500]}

    tasks = [process_result(item) for item in data.get("items", [])[:max_results]]
    return await asyncio.gather(*tasks)


async def fetch_serp_results(query: str, max_results=5):
    """Fetches results from SERP API and extracts abstracts asynchronously."""
    if not SERP_API_KEY:
        return []

    url = f"https://serpapi.com/search.json?q={query}&api_key={SERP_API_KEY}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ SERP API request failed: {e}")
            return []

    async def process_result(item):
        """Processes each search result asynchronously."""
        title = item.get("title", "Untitled")
        link = item.get("link", "")

        abstract = await (
            extract_text_from_pdf_async(link)
            if link.endswith(".pdf")
            else extract_text_from_webpage_async(link)
        )
        return {"title": title, "link": link, "abstract": abstract[:500]}

    tasks = [
        process_result(item) for item in data.get("organic_results", [])[:max_results]
    ]
    return await asyncio.gather(*tasks)


def clean_text(text):
    """Cleans up text by removing unwanted phrases and excess whitespace."""
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = re.sub(
        r"(Skip to content|Twitter|Facebook|LinkedIn|Email|Print)", "", text
    )  # Remove boilerplate
    return text.strip()


# Endpoints

@research.get("/search/")
async def search_papers(query: str, db: Session = Depends(get_db)):
    """Search for research papers, cache results, and store embeddings."""

    print(f"🔎 DEBUG: Searching for '{query}'...")

    # 🔍 Step 1: Check cache first
    cached_result = db.query(CachedResult).filter(CachedResult.query == query).first()
    if cached_result and not cached_result.is_expired():
        print("⚡ DEBUG: Returning cached results.")
        return {"query": query, "results": cached_result.results, "cached": True}

    # 🌍 Step 2: Fetch results from external sources
    arxiv_results, google_results, serp_results = await asyncio.gather(
        fetch_arxiv_results(query),
        fetch_google_results(query),
        fetch_serp_results(query),
    )

    # 🧹 Step 3: Clean abstracts
    for result_set in [google_results, serp_results]:
        for paper in result_set:
            if "abstract" in paper:
                paper["abstract"] = clean_text(paper["abstract"])

    # 📌 Step 4: Combine results
    combined_results = {
        "arxiv": arxiv_results,
        "google": google_results,
        "serpapi": serp_results,
    }

    # 🚨 Step 5: Ensure papers exist before processing
    all_papers = arxiv_results + google_results + serp_results
    print(f"🔍 DEBUG: Found {len(all_papers)} papers.")

    if all_papers:
        # 🚀 Step 6: Store new results in cache (if not already stored)
        if not cached_result:
            new_cache_entry = CachedResult(
                query=query,
                results=combined_results,
                expires_at=datetime.utcnow() + timedelta(days=7),
            )
            db.add(new_cache_entry)
            db.commit()

        # 🛠️ Step 7: Generate embeddings & store in FAISS
        print("📌 DEBUG: Processing papers for embeddings...")
        await process_and_store_paper_embeddings(all_papers, db)
    else:
        print("⚠️ DEBUG: No papers found, skipping embedding storage.")

    return {"query": query, "results": combined_results, "cached": False}


@research.get("/faiss-search/")
async def faiss_search(query: str, top_k: int = 5, db: Session = Depends(get_db)):
    """Fast FAISS search for research papers"""
    try:
        print(f"🔎 DEBUG: Searching FAISS for query: '{query}'")

        # ✅ Step 1: Get embedding from Vertex AI
        query_embedding = await fetch_embedding_from_vertex(query)

        if query_embedding is None or len(query_embedding) == 0:
            print("⚠️ DEBUG: No embedding generated for query.")
            return {"query": query, "results": []}

        # Ensure embedding is a NumPy array with shape (1, embedding_dim)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # ✅ Step 2: Search FAISS index
        print(
            f"📌 DEBUG: FAISS index total vectors: {index.ntotal}"
        )  # Check index size

        D, I = index.search(query_embedding, top_k)

        # Debugging output: Check returned indices and distances
        print(f"📌 DEBUG: FAISS search result indices: {I}")
        print(f"📌 DEBUG: FAISS search result distances: {D}")

        # ✅ Step 3: Convert indices & filter out invalid ones
        valid_indices = [int(idx) for idx in I[0] if idx >= 0]
        if not valid_indices:
            print("⚠️ DEBUG: No matching papers found in FAISS.")
            return {"query": query, "results": []}

        # ✅ Step 4: Fetch matching papers in **one batch query**
        papers = (
            db.query(PaperEmbedding).filter(PaperEmbedding.id.in_(valid_indices)).all()
        )

        # ✅ Step 5: Create a dictionary for fast lookup
        paper_dict = {paper.id: paper for paper in papers}

        # ✅ Step 6: Map FAISS results to papers & sort by similarity
        results = [
            {
                "title": paper_dict[idx].title,
                "abstract": paper_dict[idx].abstract,
                "key_findings": paper_dict[idx].key_findings,
                "link": paper_dict[idx].link,
                "similarity_score": float(score),
            }
            for idx, score in zip(I[0], D[0])
            if idx in paper_dict
        ]

        # ✅ Step 7: Sort results by similarity (highest first)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        print(f"✅ DEBUG: Returning {len(results)} results.")

        return {"query": query, "results": results}

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"query": query, "results": [], "error": str(e)}


@research.post("/upload/")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a PDF or DOCX file, extract text, generate embeddings, and store them."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    text = extract_text_from_file(file_path)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from file.")

    # Chunk text
    text_chunks = chunk_text(text)

    # Generate embeddings for each chunk
    embeddings = []
    for chunk in text_chunks:
        embedding = await fetch_embedding_from_vertex(chunk)  # Await the coroutine
        if embedding is not None:
            embeddings.append(embedding.tolist())  # Convert to list

    if not embeddings:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.")

    # Store in database
    for emb in embeddings:
        new_entry = PaperEmbedding(
            title=file.filename,
            abstract="",
            key_findings=[],
            link=file_path,
            embedding=emb,
        )
        db.add(new_entry)
    db.commit()

    # Update FAISS index
    rebuild_faiss_index(db)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Failed to delete file {file_path}: {e}")

    return {"message": "File processed and stored successfully", "file": file.filename}


def format_context_for_model(query: str, retrieved_docs):
    """Format retrieved research papers into a structured prompt."""

    context = ""
    for doc in retrieved_docs:
        context += f"Title: {doc['title']}\n"
        context += f"Key Findings: {', '.join(doc['key_findings'])}\n"
        context += f"Abstract: {doc['text']}\n\n"

    
    return format


def search_relevant_docs(query: str, db: Session, top_k=5):
    """Retrieve top_k most relevant documents using FAISS."""

    # Convert query to embedding
    query_embedding = fetch_embedding_from_vertex(query)
    if query_embedding is None:
        raise ValueError("Failed to generate query embedding")

    # Convert to numpy array
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

    # Load FAISS index
    index = faiss.read_index("faiss_index.bin")

    # Search FAISS for similar embeddings
    distances, indices = index.search(query_vector, top_k)

    # Fetch the corresponding documents from the database
    results = []
    for idx in indices[0]:
        if idx == -1:  # FAISS returns -1 if no match is found
            continue
        doc = db.query(PaperEmbedding).filter(PaperEmbedding.id == idx).first()
        if doc:
            results.append(
                {
                    "title": doc.title,
                    "text": doc.abstract,
                    "key_findings": doc.key_findings,
                }
            )

    return results

@research.get("/generate_report/")
def generate_research_report(query: str, db: Session = Depends(get_db)):
    """Retrieve relevant documents and generate structured text for research reports."""
    
    # Retrieve research papers using FAISS
    retrieved_docs = search_relevant_docs(query, db)

    if not retrieved_docs:
        return {"report": "No relevant documents found"}

    # Format retrieved documents into structured context
    formatted_context = format_context_for_model(query, retrieved_docs)
    
    return {"query": query, "report": formatted_context}
