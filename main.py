import uvicorn
from contextlib import asynccontextmanager
#from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from api.db.database import create_database
from api.v1.routes import api_version_one
#from api.utils.embeddings import load_faiss_index
#from sqlalchemy.orm import Session
from api.db.database import SessionLocal

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_database()
    yield
    ## write shutdown logic below yield


app = FastAPI(lifespan=lifespan)

#@app.on_event("startup")
#def startup_event():
#    db: Session = SessionLocal()
#    load_faiss_index(db)
#    db.close()



origins = [
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_version_one)


@app.get("/", tags=["Home"])
async def get_root(request: Request) -> dict:
    return {
        "message": "Welcome to API",
        "URL": "",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", port=7001, reload=True)
