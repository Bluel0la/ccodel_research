from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, UUID
from sqlalchemy.sql import func
from api.db.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(UUID(as_uuid=True), ForeignKey("reports.id"), nullable=False)
    file_type = Column(String, nullable=False)  # "latex" or "pdf"
    file_path = Column(String, nullable=False)  # Stored file path
    created_at = Column(DateTime(timezone=True), server_default=func.now())
