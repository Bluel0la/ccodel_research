import httpx
import os
import asyncio
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from api.db.database import get_db
from api.v1.models.cache import CachedResult
from api.utils.embeddings import (
    process_and_store_paper_embeddings,
    extract_text_from_pdf_async,
    extract_text_from_webpage_async
)
from api.utils.embeddings import index, fetch_embedding_from_vertex
from api.v1.models.embeddings import PaperEmbedding
import numpy as np
import re
from xml.etree import ElementTree as ET
from endpoint_tester import predict_custom_trained_model
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
        query_embedding = await fetch_embedding_from_vertex(
            query
        )  # Ensure this returns a list
        if not query_embedding:
            print("⚠️ DEBUG: No embedding generated for query.")
            return {"query": query, "results": []}

        # ✅ Step 2: Search FAISS index
        D, I = index.search(np.array([query_embedding]), top_k)

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
