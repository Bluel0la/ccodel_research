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
    save_faiss_index,
    process_and_store_paper_embeddings,
    extract_text_from_pdf,
    extract_text_from_webpage_async
)
from api.utils.embeddings import embed_model, index
from api.v1.models.embeddings import PaperEmbedding
import numpy as np
import re
import psutil

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

# Paper Search Functions
async def fetch_arxiv_results(query: str, max_results=5):
    """Fetch research papers from ArXiv, including abstract and metadata."""
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    if response.status_code == 200:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        
        return [
            {
                "title": entry.find("{http://www.w3.org/2005/Atom}title").text,
                "abstract": entry.find("{http://www.w3.org/2005/Atom}summary").text.strip(),  # Extract abstract
                "link": entry.find("{http://www.w3.org/2005/Atom}id").text,
            }
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry")
        ]
    return []

async def fetch_google_results(query: str, max_results=5):
    """Fetch results from Google Custom Search API and extract abstracts."""
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []
    
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    if response.status_code == 200:
        data = response.json()
        results = []
        for item in data.get("items", [])[:max_results]:
            title = item["title"]
            link = item["link"]

            # Extract text from link
            if link.endswith(".pdf"):
                abstract = extract_text_from_pdf(link)
            else:
                abstract = extract_text_from_webpage_async(link)

            results.append({"title": title, "link": link, "abstract": abstract[:500]})  # Trim abstract
        return results
    
    return []

async def fetch_serp_results(query: str, max_results=5):
    """Fetch results from SERP API and extract abstracts asynchronously."""
    if not SERP_API_KEY:
        return []

    url = f"https://serpapi.com/search.json?q={query}&api_key={SERP_API_KEY}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
        except httpx.HTTPError as e:
            print(f"❌ SERP API request failed: {e}")
            return []

    data = response.json()
    results = []

    async def process_result(item):
        """Processes each search result asynchronously."""
        title = item.get("title", "Untitled")
        link = item.get("link", "")

        if link.endswith(".pdf"):
            abstract = extract_text_from_pdf(link)  # Use existing function
        else:
            abstract = await extract_text_from_webpage_async(link)

        return {"title": title, "link": link, "abstract": abstract[:500]}  # Trim abstract

    tasks = [process_result(item) for item in data.get("organic_results", [])[:max_results]]
    results = await asyncio.gather(*tasks)  # Run all tasks concurrently

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
        query_embedding = embed_model.encode(query).tolist()
        D, I = index.search(np.array([query_embedding]), top_k)

        # Convert numpy.int64 indices to Python int and remove invalid indices (-1)
        valid_indices = [int(idx) for idx in I[0] if idx >= 0]

        if not valid_indices:
            return {"query": query, "results": []}

        # Fetch all matching papers in one batch query
        papers = db.query(PaperEmbedding).filter(PaperEmbedding.id.in_(valid_indices)).all()

        # Create a dictionary of papers for fast lookup
        paper_dict = {paper.id: paper for paper in papers}

        # Map FAISS results to database papers
        results = [
            {
                "title": paper_dict[idx].title,
                "abstract": paper_dict[idx].abstract,
                "key_findings": paper_dict[idx].key_findings,
                "link": paper_dict[idx].link,
                "similarity_score": float(score)
            }
            for idx, score in zip(I[0], D[0]) if idx in paper_dict
        ]

        # Sort results by similarity score (descending)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        return {"query": query, "results": results}

    finally:
        # Cleanup multiprocessing processes
        for proc in psutil.process_iter():
            try:
                if proc.name() == "python":
                    proc.terminate()  # Terminate any lingering Python processes
            except psutil.NoSuchProcess:
                pass
