"""
API v1 routes aggregation.
"""

from fastapi import APIRouter

from app.api.v1 import (
    alerts,
    analytics,
    audit_logs,
    auth,
    cameras,
    companies,
    detection_settings,
    detections,
    emergency_contacts,
    evidence,
    retention,
    users,
    video_processing,
)

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(companies.router)
api_router.include_router(cameras.router)
api_router.include_router(detections.router)
api_router.include_router(alerts.router)
api_router.include_router(evidence.router)
api_router.include_router(analytics.router)
api_router.include_router(audit_logs.router)
api_router.include_router(detection_settings.router)
api_router.include_router(video_processing.router)
api_router.include_router(retention.router)
api_router.include_router(emergency_contacts.router)
