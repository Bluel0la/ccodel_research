import requests
from fastapi import FastAPI, Depends, APIRouter
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from api.db.database import get_db
from api.v1.models.cache import CachedResult
import os
from dotenv import load_dotenv
from api.utils.embeddings import search_similar_papers
import httpx
import asyncio
from api.utils.embeddings import fetch_external_papers, preload_external_papers, save_faiss_index
from fastapi_utils.tasks import repeat_every  # Auto-scheduler

load_dotenv(".env")

research = APIRouter(prefix="/research", tags=["Research"])

# Load API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")  # Custom Search Engine ID
SERP_API_KEY = os.getenv("SERP_API_KEY")

async def fetch_arxiv_results(query: str, max_results=5):
    """Fetch relevant papers from ArXiv asynchronously."""
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    
    if response.status_code == 200:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        results = [
            {"title": entry.find("{http://www.w3.org/2005/Atom}title").text, "link": entry.find("{http://www.w3.org/2005/Atom}id").text}
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry")
        ]
        return results
    return []

async def fetch_google_results(query: str, max_results=5):
    """Fetch results from Google Custom Search API asynchronously."""
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return [{"title": item["title"], "link": item["link"]} for item in data.get("items", [])[:max_results]]
    return []

async def fetch_serp_results(query: str, max_results=5):
    """Fetch results from SERP API asynchronously."""
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERP_API_KEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return [{"title": item["title"], "link": item["link"]} for item in data.get("organic_results", [])[:max_results]]
    return []

@research.get("/search/")
async def search(query: str, db: Session = Depends(get_db)):
    """Main search function with caching, FAISS, and async requests."""
    cached_entry = db.query(CachedResult).filter(CachedResult.query == query).first()
    
    if cached_entry and not cached_entry.is_expired():
        return {"cached": True, "query": query, "results": cached_entry.results}

    # Run all fetch calls asynchronously
    arxiv_results, google_results, serp_results = await asyncio.gather(
        fetch_arxiv_results(query),
        fetch_google_results(query),
        fetch_serp_results(query)
    )

    # Get FAISS results
    faiss_results = search_similar_papers(query, db)

    combined_results = {
        "arxiv": arxiv_results,
        "google": google_results,
        "serpapi": serp_results,
        "faiss": faiss_results
    }

    # Store in cache
    new_cache = CachedResult(
        query=query,
        results=combined_results,
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(new_cache)
    db.commit()

    return {"cached": False, "query": query, "results": combined_results}


@research.on_event("startup")
@repeat_every(seconds=86400)  # Every 24 hours
def update_papers_periodically():
    """Fetch new research papers daily and update FAISS."""
    db = next(get_db())  # Get database session
    external_papers = fetch_external_papers()
    
    if external_papers:
        preload_external_papers(db, external_papers)
        save_faiss_index()  # Save updated FAISS index
        print("🔄 FAISS index updated with new papers.")
    else:
        print("⚠️ No new external papers found.")

@research.get("/faiss-search/")
async def faiss_search(query: str, db: Session = Depends(get_db)):
    """Search only using FAISS (no external sources)."""
    results = search_similar_papers(query, db)
    return {"query": query, "faiss_results": results}
