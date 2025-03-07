from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.dialects.postgresql import ARRAY
from api.db.database import Base

class PaperEmbedding(Base):
    __tablename__ = "paper_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    abstract = Column(String, nullable=False)
    key_findings = Column(ARRAY(String), nullable=False)
    link = Column(String, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)  # pgvector stores vectors
