"""
Safe Razorpay Test Mode recovery executor.

Purpose:
- Execute ONLY the selected `payment_link` recovery action.
- Use Razorpay TEST keys only.
- Default to dry-run unless RAZORPAY_EXECUTION_ENABLED=true.
- Enforce a per-run maximum number of live test links.
- Store the Razorpay payment_link id/url and status in a local JSONL audit file.

Required environment variables:
    RAZORPAY_KEY_ID=rzp_test_...
    RAZORPAY_KEY_SECRET=...
    RAZORPAY_EXECUTION_ENABLED=false|true

Optional:
    RAZORPAY_MAX_TEST_LINKS=5
    RAZORPAY_WEBHOOK_SECRET=...
    RAZORPAY_AUDIT_FILE=testmode_audit.jsonl
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


API_BASE = "https://api.razorpay.com"
AUDIT_FILE = Path(
    os.getenv("RAZORPAY_AUDIT_FILE", "testmode_audit.jsonl")
)

ALLOWED_LIVE_ACTION = "payment_link"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _config() -> dict[str, Any]:
    execution_enabled = _env_bool(
        "RAZORPAY_EXECUTION_ENABLED",
        False
    )

    # Dry-run mode must work without credentials.
    if not execution_enabled:
        return {
            "key_id": "",
            "key_secret": "",
            "execution_enabled": False,
            "max_links": 0,
        }

    key_id = os.getenv(
        "RAZORPAY_KEY_ID",
        ""
    ).strip()

    key_secret = os.getenv(
        "RAZORPAY_KEY_SECRET",
        ""
    ).strip()

    if not key_id or not key_secret:
        raise RuntimeError(
            "Missing RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET."
        )

    # Safety: TEST keys only.
    if not key_id.startswith("rzp_test_"):
        raise RuntimeError(
            "Safety stop: this executor accepts TEST keys only "
            "(RAZORPAY_KEY_ID must start with 'rzp_test_')."
        )

    max_links = int(
        os.getenv(
            "RAZORPAY_MAX_TEST_LINKS",
            "3"
        )
    )

    if max_links < 1 or max_links > 10:
        raise ValueError(
            "RAZORPAY_MAX_TEST_LINKS must be between 1 and 10."
        )

    return {
        "key_id": key_id,
        "key_secret": key_secret,
        "execution_enabled": True,
        "max_links": max_links,
    }


def _count_created_test_links() -> int:
    if not AUDIT_FILE.exists():
        return 0

    count = 0
    with AUDIT_FILE.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if (
                row.get("event") == "payment_link_created"
                and row.get("mode") == "test"
            ):
                count += 1

    return count


def _write_audit(event: str, **payload: Any) -> None:
    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "timestamp": _utc_timestamp(),
        "event": event,
        "mode": "test",
        **payload,
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


def create_test_payment_link(
    *,
    payment_id: str,
    amount_inr: float,
    description: str | None = None,
    expire_by: int | None = None,
) -> dict[str, Any]:
    """
    Create ONE Razorpay Test Mode payment link for a payment selected
    by the agent.

    This function does not create links for blocked/no_action/escalate.
    It enforces a small per-run limit and rejects live keys.
    """

    if amount_inr <= 0:
        raise ValueError(
            "amount_inr must be greater than zero."
        )

    config = _config()

    if not config["execution_enabled"]:
        result = {
            "mode": "test",
            "executed": False,
            "status": "dry_run",
            "payment_id": payment_id,
            "amount_inr": float(amount_inr),
            "message": (
                "Dry-run only. Set RAZORPAY_EXECUTION_ENABLED=true "
                "to create a Test Mode Payment Link."
            ),
        }

        _write_audit(
            "payment_link_dry_run",
            **result,
        )
        return result

    created = _count_created_test_links()
    if created >= config["max_links"]:
        raise RuntimeError(
            "Bounded execution limit reached: "
            f"{created}/{config['max_links']} Test Mode links created."
        )

    amount_paise = int(round(float(amount_inr) * 100))

    payload: dict[str, Any] = {
        "amount": amount_paise,
        "currency": "INR",
        "accept_partial": False,
        "description": (
            description
            or f"Recovery for failed payment {payment_id}"
        ),
        "reference_id": str(payment_id),
        "notes": {
            "source": "AI Revenue Recovery Agent",
            "mode": "test",
            "original_payment_id": str(payment_id),
        },
    }

    if expire_by is not None:
        payload["expire_by"] = int(expire_by)

    response = requests.post(
        f"{API_BASE}/v1/payment_links",
        auth=(
            config["key_id"],
            config["key_secret"],
        ),
        json=payload,
        timeout=20,
    )

    response.raise_for_status()
    data = response.json()

    result = {
        "mode": "test",
        "executed": True,
        "status": "created",
        "payment_id": str(payment_id),
        "amount_inr": float(amount_inr),
        "payment_link_id": data.get("id"),
        "payment_link_url": data.get("short_url"),
        "razorpay_status": data.get("status"),
    }

    _write_audit(
        "payment_link_created",
        **result,
    )

    return result


if __name__ == "__main__":
    print(
        "Safe Razorpay Test Mode executor loaded."
    )
    print(
        "Live financial execution is explicitly blocked."
    )
    print(
        "Set RAZORPAY_EXECUTION_ENABLED=true only when "
        "you are ready to create bounded TEST links."
    )
