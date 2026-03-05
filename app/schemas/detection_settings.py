"""
Pydantic schemas for detection module settings.
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ── Global Module Settings (Aegis Admin) ──

class GlobalModuleSettingsResponse(BaseModel):
    id: int
    module_name: str
    is_enabled: bool
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class GlobalModuleSettingsUpdate(BaseModel):
    is_enabled: bool


# ── Company Detection Settings ──

class CompanyDetectionSettingsCreate(BaseModel):
    module_name: str
    is_enabled: bool = True
    critical_threshold: Optional[float] = None
    high_threshold: Optional[float] = None
    medium_threshold: Optional[float] = None
    min_confidence: Optional[float] = None


class CompanyDetectionSettingsUpdate(BaseModel):
    is_enabled: Optional[bool] = None
    critical_threshold: Optional[float] = None
    high_threshold: Optional[float] = None
    medium_threshold: Optional[float] = None
    min_confidence: Optional[float] = None


class CompanyDetectionSettingsResponse(BaseModel):
    id: int
    company_id: int
    module_name: str
    is_enabled: bool
    critical_threshold: Optional[float] = None
    high_threshold: Optional[float] = None
    medium_threshold: Optional[float] = None
    min_confidence: Optional[float] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
