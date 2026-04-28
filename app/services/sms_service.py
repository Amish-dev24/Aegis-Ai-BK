"""
SMS service using Twilio for sending alert notifications.
"""

import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Try to import Twilio
try:
    from twilio.rest import Client as TwilioClient

    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("twilio not installed — SMS disabled. Install with: pip install twilio")


def is_sms_configured() -> bool:
    """Check if Twilio SMS is properly configured."""
    return bool(
        TWILIO_AVAILABLE
        and settings.TWILIO_ACCOUNT_SID
        and settings.TWILIO_AUTH_TOKEN
        and settings.TWILIO_PHONE_NUMBER
    )


def send_sms(to_number: str, message: str) -> bool:
    """
    Send an SMS via Twilio.

    Args:
        to_number: Recipient phone number (e.g., "+923001234567")
        message: SMS text (max 1600 chars, will be truncated)

    Returns:
        True if sent successfully, False otherwise.
    """
    if not is_sms_configured():
        logger.warning("SMS not configured — skipping SMS to %s", to_number)
        return False

    # Truncate message to SMS limit
    if len(message) > 1600:
        message = message[:1597] + "..."

    try:
        client = TwilioClient(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=to_number,
        )
        logger.info("SMS sent to %s (SID: %s)", to_number, msg.sid)
        return True
    except Exception as e:
        logger.error("Failed to send SMS to %s: %s", to_number, e)
        return False


def send_alert_sms(
    to_number: str,
    detection_type: str,
    threat_level: str,
    confidence: float,
    camera_name: str,
    timestamp: str,
) -> bool:
    """
    Send a formatted alert SMS.
    """
    message = (
        f"AEGIS AI ALERT\n"
        f"Threat: {detection_type.replace('_', ' ').upper()}\n"
        f"Level: {threat_level.upper()}\n"
        f"Confidence: {confidence:.0%}\n"
        f"Camera: {camera_name}\n"
        f"Time: {timestamp}\n"
        f"Check dashboard for details."
    )
    return send_sms(to_number, message)
