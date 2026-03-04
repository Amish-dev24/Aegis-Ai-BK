"""
Evidence model for storing detection snapshots and metadata.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Evidence(Base):
    """Evidence model for storing detection snapshots."""
    __tablename__ = "evidence"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False)
    image_path = Column(String(500), nullable=False)  # Path to snapshot image
    video_path = Column(String(500), nullable=True)  # Path to video clip if available
    frame_number = Column(Integer, nullable=True)
    metadata_json = Column(Text)  # Additional metadata as JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    detection = relationship("Detection", back_populates="evidence")

