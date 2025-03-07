from sqlalchemy import Column, String, ForeignKey, Text, DateTime, UUID
from sqlalchemy.sql import func
import uuid
from api.db.database import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)  # AI-generated report content
    format = Column(String, default="LaTeX")  # LaTeX format
    created_at = Column(DateTime(timezone=True), server_default=func.now())
