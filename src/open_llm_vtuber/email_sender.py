import smtplib
from email.mime.text import MIMEText
from typing import Optional

class EmailSender:
    def __init__(self, host: str, port: int, use_ssl: bool, username: Optional[str], password: Optional[str]):
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.username = username
        self.password = password

    def send_email(self, recipient_email: str, subject: str, body: str) -> bool:
        if not self.host or not self.port:
            print("SMTP host or port not configured. Cannot send email.")
            return False

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = recipient_email

        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.host, self.port)
            else:
                server = smtplib.SMTP(self.host, self.port)
            
            # If using TLS (typically port 587), it's not SMTP_SSL from the start
            # It's an SMTP connection that's upgraded to TLS
            if not self.use_ssl and self.port == 587: # Common port for STARTTLS
                server.starttls()

            if self.username and self.password:
                server.login(self.username, self.password)
            
            server.sendmail(self.username or "noreply@example.com", recipient_email, msg.as_string())
            server.quit()
            print(f"Email sent successfully to {recipient_email}")
            return True
        except smtplib.SMTPAuthenticationError:
            print(f"SMTP Authentication Error. Please check username/password.")
            return False
        except smtplib.SMTPConnectError:
            print(f"SMTP Connection Error. Please check host/port or network.")
            return False
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False
