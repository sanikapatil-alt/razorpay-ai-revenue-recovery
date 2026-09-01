"""
Minimal Razorpay webhook receiver for Test Mode recovery outcomes.

Run:
    python testmode_webhook.py

Environment:
    RAZORPAY_WEBHOOK_SECRET=your_dashboard_webhook_secret
    WEBHOOK_PORT=8000

This receiver listens for:
    payment_link.paid
    payment_link.partially_paid
    payment_link.cancelled
    payment_link.expired

It validates X-Razorpay-Signature against the raw request body.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

AUDIT_FILE = Path(
    os.getenv("RAZORPAY_AUDIT_FILE", "testmode_audit.jsonl")
)


def _write_audit(event: str, payload: dict) -> None:
    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "event": event,
        "payload": payload,
    }

    with AUDIT_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def _valid_signature(body: bytes, received: str) -> bool:
    secret = os.getenv(
        "RAZORPAY_WEBHOOK_SECRET",
        "",
    ).encode("utf-8")

    if not secret or not received:
        return False

    expected = hmac.new(
        secret,
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(
        expected,
        received,
    )


class WebhookHandler(BaseHTTPRequestHandler):

    def do_POST(self) -> None:
        length = int(
            self.headers.get("Content-Length", "0")
        )
        body = self.rfile.read(length)

        signature = self.headers.get(
            "X-Razorpay-Signature",
            "",
        )

        if not _valid_signature(
            body,
            signature,
        ):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(
                b"Invalid webhook signature"
            )
            return

        try:
            payload = json.loads(
                body.decode("utf-8")
            )
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(
                b"Invalid JSON"
            )
            return

        event = payload.get(
            "event",
            "unknown",
        )

        tracked_events = {
            "payment_link.paid",
            "payment_link.partially_paid",
            "payment_link.cancelled",
            "payment_link.expired",
        }

        if event in tracked_events:
            _write_audit(
                f"webhook:{event}",
                payload,
            )

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(
        self,
        format: str,
        *args,
    ) -> None:
        print(format % args)


if __name__ == "__main__":
    port = int(
        os.getenv("WEBHOOK_PORT", "8000")
    )

    print(
        f"Webhook receiver listening on http://127.0.0.1:{port}/"
    )

    HTTPServer(
        ("127.0.0.1", port),
        WebhookHandler,
    ).serve_forever()
