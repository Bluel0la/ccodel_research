from sqlalchemy.orm import Session
from api.db.database import SessionLocal
from api.v1.models.embeddings import PaperEmbedding
from sentence_transformers import SentenceTransformer

# Initialize DB session
db: Session = SessionLocal()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample data
sample_papers = [
    {"title": "AI Research", "abstract": "AI is evolving rapidly."},
    {"title": "Machine Learning", "abstract": "ML models are improving every day."},
]

for paper in sample_papers:
    embedding = embed_model.encode(paper["abstract"]).tolist()
    new_entry = PaperEmbedding(title=paper["title"], abstract=paper["abstract"], embedding=embedding)
    db.add(new_entry)

db.commit()
db.close()

print("✅ Sample embeddings added to database!")
