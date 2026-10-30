"""Sign-up and password-reset email over SMTP. Password sign-up stays off until every setting is given."""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional

from fastapi.concurrency import run_in_threadpool

SMTP_SETTINGS = tuple(f"STOCKINTEL_SMTP_{name}" for name in ("HOST", "PORT", "USER", "PASSWORD", "FROM"))
SMTP_TIMEOUT_S = 10  # PROVISIONAL: so a stuck mail server fails the request; time a real send to confirm


class SmtpMailer:
    def __init__(self, host: str, port: int, user: str, password: str, sender: str):
        self.host, self.port, self.user, self.password, self.sender = host, port, user, password, sender

    async def send(self, to: str, subject: str, text: str) -> None:
        message = EmailMessage()
        message["From"], message["To"], message["Subject"] = self.sender, to, subject
        message.set_content(text)
        await run_in_threadpool(self._deliver, message)

    def _deliver(self, message: EmailMessage) -> None:
        # Without a context smtplib skips the certificate check and would hand the password to anyone in between
        with smtplib.SMTP_SSL(self.host, self.port, timeout=SMTP_TIMEOUT_S,
                              context=ssl.create_default_context()) as smtp:
            smtp.login(self.user, self.password)
            smtp.send_message(message)


def mailer_from_env() -> Optional[SmtpMailer]:
    values = {name: os.environ.get(name, "") for name in SMTP_SETTINGS}
    missing = [name for name, value in values.items() if not value]
    if len(missing) == len(SMTP_SETTINGS):
        return None
    if missing:
        raise RuntimeError(f"{', '.join(missing)}: not set, but other STOCKINTEL_SMTP_ settings are. Set all five "
                           "from your email provider's SMTP page, or none to keep password sign-up off.")
    host, port, user, password, sender = values.values()
    if not port.isdigit():
        raise RuntimeError(f"STOCKINTEL_SMTP_PORT is {port!r}; set it to the provider's SSL port, usually 465.")
    return SmtpMailer(host, int(port), user, password, sender)
