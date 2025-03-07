from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, UUID
from sqlalchemy.sql import func
from api.db.database import Base


class Citation(Base):
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(UUID(as_uuid=True), ForeignKey("reports.id"), nullable=False)
    citation_text = Column(Text, nullable=False)  # Formatted citation
    format = Column(String, nullable=False)  # APA, MLA, IEEE, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
