import httpx
import os
import asyncio
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from api.db.database import get_db, SessionLocal
from api.v1.models.cache import CachedResult
from api.utils.embeddings import (
    process_and_store_paper_embeddings,
    extract_text_from_pdf_async,
    extract_text_from_webpage_async,
    predict_custom_trained_model,
)
from api.utils.embeddings import index
from api.v1.models.embeddings import PaperEmbedding
import numpy as np
import re
import psutil
from api.utils.embeddings import embedding_dim
import xml.etree.ElementTree as ET

# Load environment variables
load_dotenv()

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

import asyncio
import httpx
import xml.etree.ElementTree as ET


# Paper Search Functions
async def fetch_arxiv_results(query: str, max_results=5):
    """Fetch research papers from ArXiv, including abstract and metadata."""
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"❌ ArXiv API request failed: {e}")
            return []

    root = ET.fromstring(response.content)
    results = []
    for entry in root.iterfind(".//{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        link = entry.find("{http://www.w3.org/2005/Atom}id").text
        results.append({"title": title, "abstract": abstract, "link": link})

    return results


async def fetch_google_results(query: str, max_results=5):
    """Fetch results from Google Custom Search API and extract abstracts."""
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        print("❌ Google API key or CX is missing.")
        return []

    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"❌ Google API request failed: {e}")
            return []

    data = response.json()
    results = []
    tasks = []

    for item in data.get("items", [])[:max_results]:
        title = item["title"]
        link = item["link"]

        if link.endswith(".pdf"):
            tasks.append(extract_text_from_pdf_async(link))
        else:
            tasks.append(extract_text_from_webpage_async(link))

    abstracts = await asyncio.gather(*tasks, return_exceptions=True)

    for i, item in enumerate(data.get("items", [])[:max_results]):
        results.append(
            {
                "title": item["title"],
                "link": item["link"],
                "abstract": (
                    abstracts[i][:500]
                    if isinstance(abstracts[i], str)
                    else "Failed to retrieve abstract"
                ),
            }
        )

    return results


async def fetch_serp_results(query: str, max_results=5):
    """Fetch results from SERP API and extract abstracts asynchronously."""
    if not SERP_API_KEY:
        print("❌ SERP API key is missing.")
        return []

    url = f"https://serpapi.com/search.json?q={query}&api_key={SERP_API_KEY}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"❌ SERP API request failed: {e}")
            return []

    data = response.json()
    results = []
    tasks = []

    for item in data.get("organic_results", [])[:max_results]:
        title = item.get("title", "Untitled")
        link = item.get("link", "")

        if link.endswith(".pdf"):
            tasks.append(extract_text_from_pdf_async(link))
        else:
            tasks.append(extract_text_from_webpage_async(link))

    abstracts = await asyncio.gather(*tasks, return_exceptions=True)

    for i, item in enumerate(data.get("organic_results", [])[:max_results]):
        results.append(
            {
                "title": item.get("title", "Untitled"),
                "link": item.get("link", ""),
                "abstract": (
                    abstracts[i][:500]
                    if isinstance(abstracts[i], str)
                    else "Failed to retrieve abstract"
                ),
            }
        )

    return results


def clean_text(text):
    """Cleans up text by removing unwanted phrases and excess whitespace."""
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = re.sub(r"(Skip to content|Twitter|Facebook|LinkedIn|Email|Print)", "", text)  # Remove boilerplate
    return text.strip()

@research.get("/search/")
async def search_papers(query: str, db: Session = Depends(get_db)):
    """Search for research papers, cache results, and store embeddings."""

    # 🔍 Check cache first
    cached_result = db.query(CachedResult).filter(CachedResult.query == query).first()
    if cached_result and not cached_result.is_expired():
        return {"query": query, "results": cached_result.results, "cached": True}

    # 🌍 Fetch results
    arxiv_results, google_results, serp_results = await asyncio.gather(
        fetch_arxiv_results(query),
        fetch_google_results(query),
        fetch_serp_results(query)
    )

    # 🧹 Clean abstracts
    for result_set in [google_results, serp_results]:
        for paper in result_set:
            if "abstract" in paper:
                paper["abstract"] = clean_text(paper["abstract"])

    # 📌 Combine results
    combined_results = {
        "arxiv": arxiv_results,
        "google": google_results,
        "serpapi": serp_results
    }

    # 🚨 DEBUG: Check if any papers exist
    all_papers = arxiv_results + google_results + serp_results
    print(f"🔍 DEBUG: Found {len(all_papers)} papers to process.")  

    # 🛑 Cache results if not already stored
    if not db.query(CachedResult).filter(CachedResult.query == query).first():
        new_cache_entry = CachedResult(
            query=query,
            results=combined_results,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
        db.add(new_cache_entry)
        db.commit()

    # 🚀 Process embeddings
    if all_papers:
        print("📌 DEBUG: Sending papers for embedding storage...")  
        await process_and_store_paper_embeddings(all_papers, db)  
    else:
        print("⚠️ DEBUG: No papers found, skipping embedding storage.")

    return {"query": query, "results": combined_results, "cached": False}


@research.get("/faiss-search/")
async def faiss_search(query: str, top_k: int = 5, db: Session = Depends(get_db)):
    """Fast FAISS search for research papers"""
    try:
        # 🛠️ Fetch query embedding from hosted model API
        embedding_response = await predict_custom_trained_model(
            project="178740183321",
            endpoint_id="3947058727227490304",
            location="us-central1",
            instances={"text": query},
        )

        if not embedding_response:
            print("❌ DEBUG: No embedding received for query")
            return {"error": "Failed to generate embedding for query"}

        # Extract the embedding vector
        query_embedding = np.array(
            embedding_response[0], dtype=np.float32
        )  # First response is embedding

        if query_embedding.shape[0] != embedding_dim:
            print(f"❌ DEBUG: Query embedding dimension mismatch")
            return {"error": "Embedding dimension mismatch"}

        # 🔍 Search FAISS index with retrieved embedding
        D, I = index.search(np.array([query_embedding]), top_k)

        # Convert numpy.int64 indices to Python int and remove invalid indices (-1)
        valid_indices = [int(idx) for idx in I[0] if idx >= 0]

        if not valid_indices:
            return {"query": query, "results": []}

        # Fetch all matching papers in one batch query
        papers = (
            db.query(PaperEmbedding).filter(PaperEmbedding.id.in_(valid_indices)).all()
        )

        # Create a dictionary of papers for fast lookup
        paper_dict = {paper.id: paper for paper in papers}

        # Map FAISS results to database papers
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

        # Sort results by similarity score (descending)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        return {"query": query, "results": results}

    except Exception as e:
        print(f"🚨 ERROR: Query embedding retrieval failed → {e}")
        return {"error": "Failed to process query embedding"}

    finally:
        # Cleanup multiprocessing processes
        for proc in psutil.process_iter():
            try:
                if "faiss" in proc.name().lower():
                    proc.terminate()
            except psutil.NoSuchProcess:
                pass
