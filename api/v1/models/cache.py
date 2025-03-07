from sqlalchemy import Column, Integer, Text, JSON, TIMESTAMP, func, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
from api.db.database import Base

class CachedResult(Base):
    __tablename__ = "cached_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False, index=True)  # Index for faster lookups
    results = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    expires_at = Column(TIMESTAMP, nullable=False, default=lambda: datetime.utcnow() + timedelta(days=7))

    def is_expired(self):
        """Check if the cache entry has expired."""
        return datetime.utcnow() > self.expires_at
