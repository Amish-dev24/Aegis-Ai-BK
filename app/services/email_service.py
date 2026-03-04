"""
Email service for sending security alerts.
"""
try:
    import aiosmtplib
except ImportError:
    aiosmtplib = None

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Optional
from pathlib import Path
from jinja2 import Template
from app.config import settings


class EmailService:
    """Service for sending email notifications."""
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.SMTP_FROM_EMAIL
        self.from_name = settings.SMTP_FROM_NAME
    
    async def send_alert_email(
        self,
        to_emails: List[str],
        subject: str,
        message: str,
        snapshot_path: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Send an alert email with optional snapshot attachment.
        
        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            message: Email body text
            snapshot_path: Path to snapshot image to attach
            metadata: Additional metadata to include in email
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            msg = MIMEMultipart("related")
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = subject
            
            # Create HTML body
            html_body = self._create_alert_html(message, metadata)
            msg.attach(MIMEText(html_body, "html"))
            
            # Attach snapshot if provided
            if snapshot_path and Path(snapshot_path).exists():
                with open(snapshot_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header("Content-Disposition", "attachment", filename="alert_snapshot.jpg")
                    msg.attach(img)
            
            # Send email
            if aiosmtplib is None:
                print("Warning: aiosmtplib not installed. Email sending skipped.")
                return False

            # Port 587 uses STARTTLS (start_tls), port 465 uses implicit TLS (use_tls)
            use_implicit_tls = self.smtp_port == 465
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                use_tls=use_implicit_tls,
                start_tls=not use_implicit_tls,
            )
            return True
        
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def _create_alert_html(self, message: str, metadata: Optional[dict]) -> str:
        """Create HTML email template."""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .alert-box { border: 2px solid #dc3545; padding: 20px; margin: 20px 0; }
                .metadata { background-color: #f8f9fa; padding: 15px; margin: 10px 0; }
                .footer { color: #6c757d; font-size: 12px; margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="alert-box">
                <h2>🚨 Security Alert</h2>
                <p>{{ message }}</p>
            </div>
            {% if metadata %}
            <div class="metadata">
                <h3>Event Details:</h3>
                <ul>
                    {% for key, value in metadata.items() %}
                    <li><strong>{{ key }}:</strong> {{ value }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
            <div class="footer">
                <p>This is an automated alert from Aegis AI Surveillance System.</p>
            </div>
        </body>
        </html>
        """
        template = Template(template_str)
        return template.render(message=message, metadata=metadata)


email_service = EmailService()

