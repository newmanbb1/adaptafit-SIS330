# backend/models.py
from sqlalchemy import Column, String, DateTime, Text,ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    # Define la relación: Un Usuario tiene muchos Jobs
    jobs = relationship("AnalysisJob", back_populates="owner")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    job_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    cuestionario_json = Column(Text, nullable=False)
    result_model_url = Column(String, nullable=True)
    result_pdf_url = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    results_json = Column(Text, nullable=True)
    
    owner_id = Column(String, ForeignKey("users.id")) # La Clave Foránea
    owner = relationship("User", back_populates="jobs") # La Relación