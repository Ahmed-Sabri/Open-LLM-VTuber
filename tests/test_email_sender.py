import pytest
from unittest.mock import patch, MagicMock
import smtplib

from open_llm_vtuber.email_sender import EmailSender

@pytest.fixture
def email_sender_config():
    return {
        "host": "smtp.example.com",
        "port": 587,
        "use_ssl": False,
        "username": "user@example.com",
        "password": "password123"
    }

@pytest.fixture
def email_sender_ssl_config():
    return {
        "host": "smtp.example.com",
        "port": 465,
        "use_ssl": True,
        "username": "user@example.com",
        "password": "password123"
    }

@patch('smtplib.SMTP')
def test_send_email_success_no_ssl(mock_smtp_constructor, email_sender_config):
    mock_smtp_instance = MagicMock()
    mock_smtp_constructor.return_value = mock_smtp_instance

    sender = EmailSender(**email_sender_config)
    success = sender.send_email("recipient@example.com", "Test Subject", "Test Body")

    assert success is True
    mock_smtp_constructor.assert_called_once_with(email_sender_config["host"], email_sender_config["port"])
    mock_smtp_instance.starttls.assert_called_once() # Port 587 implies STARTTLS
    mock_smtp_instance.login.assert_called_once_with(email_sender_config["username"], email_sender_config["password"])
    mock_smtp_instance.sendmail.assert_called_once()
    mock_smtp_instance.quit.assert_called_once()

@patch('smtplib.SMTP_SSL')
def test_send_email_success_ssl(mock_smtp_ssl_constructor, email_sender_ssl_config):
    mock_smtp_instance = MagicMock()
    mock_smtp_ssl_constructor.return_value = mock_smtp_instance

    sender = EmailSender(**email_sender_ssl_config)
    success = sender.send_email("recipient@example.com", "Test Subject SSL", "Test Body SSL")

    assert success is True
    mock_smtp_ssl_constructor.assert_called_once_with(email_sender_ssl_config["host"], email_sender_ssl_config["port"])
    assert not mock_smtp_instance.starttls.called # STARTTLS should not be called for SMTP_SSL
    mock_smtp_instance.login.assert_called_once_with(email_sender_ssl_config["username"], email_sender_ssl_config["password"])
    mock_smtp_instance.sendmail.assert_called_once()
    mock_smtp_instance.quit.assert_called_once()

@patch('smtplib.SMTP')
def test_send_email_authentication_error(mock_smtp_constructor, email_sender_config):
    mock_smtp_instance = MagicMock()
    mock_smtp_constructor.return_value = mock_smtp_instance
    mock_smtp_instance.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Authentication credentials invalid")

    sender = EmailSender(**email_sender_config)
    success = sender.send_email("recipient@example.com", "Auth Error", "Body")

    assert success is False
    mock_smtp_instance.login.assert_called_once()
    assert not mock_smtp_instance.sendmail.called # sendmail should not be called
    mock_smtp_instance.quit.assert_called_once() # quit should still be called

@patch('smtplib.SMTP')
def test_send_email_connection_error(mock_smtp_constructor, email_sender_config):
    mock_smtp_constructor.side_effect = smtplib.SMTPConnectError(550, "Connection refused")

    sender = EmailSender(**email_sender_config)
    success = sender.send_email("recipient@example.com", "Connect Error", "Body")
    
    assert success is False

@patch('smtplib.SMTP')
def test_send_email_generic_exception(mock_smtp_constructor, email_sender_config):
    mock_smtp_instance = MagicMock()
    mock_smtp_constructor.return_value = mock_smtp_instance
    mock_smtp_instance.sendmail.side_effect = Exception("Something went wrong")

    sender = EmailSender(**email_sender_config)
    success = sender.send_email("recipient@example.com", "Generic Error", "Body")

    assert success is False
    mock_smtp_instance.sendmail.assert_called_once()
    mock_smtp_instance.quit.assert_called_once()

def test_send_email_no_host_or_port():
    sender_no_host = EmailSender(host="", port=587, use_ssl=False, username="u", password="p")
    assert sender_no_host.send_email("r@e.com", "S", "B") is False

    sender_no_port = EmailSender(host="h", port=0, use_ssl=False, username="u", password="p")
    assert sender_no_port.send_email("r@e.com", "S", "B") is False

@patch('smtplib.SMTP_SSL')
def test_send_email_no_username_password_ssl(mock_smtp_ssl_constructor, email_sender_ssl_config):
    """Test sending email via SSL without username/password (e.g. open relay)."""
    mock_smtp_instance = MagicMock()
    mock_smtp_ssl_constructor.return_value = mock_smtp_instance

    config_no_auth = email_sender_ssl_config.copy()
    config_no_auth["username"] = None
    config_no_auth["password"] = None
    
    sender = EmailSender(**config_no_auth)
    success = sender.send_email("recipient@example.com", "Test Subject No Auth SSL", "Test Body No Auth SSL")

    assert success is True
    mock_smtp_ssl_constructor.assert_called_once_with(config_no_auth["host"], config_no_auth["port"])
    assert not mock_smtp_instance.login.called # Login should not be called
    mock_smtp_instance.sendmail.assert_called_once()
    # Check that "From" is a default when username is None
    args, _ = mock_smtp_instance.sendmail.call_args
    assert args[0] == "noreply@example.com" 
    mock_smtp_instance.quit.assert_called_once()

@patch('smtplib.SMTP')
def test_send_email_no_username_password_no_ssl_starttls(mock_smtp_constructor, email_sender_config):
    """Test sending email via STARTTLS without username/password."""
    mock_smtp_instance = MagicMock()
    mock_smtp_constructor.return_value = mock_smtp_instance

    config_no_auth = email_sender_config.copy()
    config_no_auth["username"] = None
    config_no_auth["password"] = None

    sender = EmailSender(**config_no_auth)
    success = sender.send_email("recipient@example.com", "Test Subject No Auth STARTTLS", "Test Body No Auth STARTTLS")

    assert success is True
    mock_smtp_constructor.assert_called_once_with(config_no_auth["host"], config_no_auth["port"])
    mock_smtp_instance.starttls.assert_called_once()
    assert not mock_smtp_instance.login.called # Login should not be called
    mock_smtp_instance.sendmail.assert_called_once()
    args, _ = mock_smtp_instance.sendmail.call_args
    assert args[0] == "noreply@example.com"
    mock_smtp_instance.quit.assert_called_once()

@patch('smtplib.SMTP') # Test with a non-standard port that wouldn't typically use STARTTLS by default
def test_send_email_no_ssl_non_starttls_port(mock_smtp_constructor, email_sender_config):
    mock_smtp_instance = MagicMock()
    mock_smtp_constructor.return_value = mock_smtp_instance

    config_alt_port = email_sender_config.copy()
    config_alt_port["port"] = 25 # Standard non-encrypted SMTP port

    sender = EmailSender(**config_alt_port)
    success = sender.send_email("recipient@example.com", "Test Subject Alt Port", "Test Body Alt Port")

    assert success is True
    mock_smtp_constructor.assert_called_once_with(config_alt_port["host"], config_alt_port["port"])
    assert not mock_smtp_instance.starttls.called # STARTTLS should not be called for port 25 unless explicitly configured
    mock_smtp_instance.login.assert_called_once_with(config_alt_port["username"], config_alt_port["password"])
    mock_smtp_instance.sendmail.assert_called_once()
    mock_smtp_instance.quit.assert_called_once()
