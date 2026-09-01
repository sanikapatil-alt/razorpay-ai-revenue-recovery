import joblib
import pandas as pd

from business_config import (
    ACTIONS,
    MAX_RETRY_ATTEMPTS,
    RECOVERY_WINDOW_HOURS,
    HIGH_VALUE_RETRY_LIMIT_INR,
    LOW_PROBABILITY_THRESHOLD,
    action_cost,
    guardrail,
    expected_net_value,
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(payment):
    payment = payment.copy()

    total_attempts = (
        payment["successful_payments"]
        + payment["failed_payments"]
    )

    payment["total_payment_attempts"] = total_attempts

    payment["payment_success_rate"] = (
        payment["successful_payments"]
        / max(total_attempts, 1)
    )

    payment["failure_rate"] = (
        payment["failed_payments"]
        / max(total_attempts, 1)
    )

    payment["retry_exhaustion"] = min(
        payment["previous_attempts"] / MAX_RETRY_ATTEMPTS,
        1,
    )

    payment["transaction_to_customer_value"] = (
        payment["amount_inr"]
        / max(payment["customer_value_inr"], 1)
    )

    payment["time_decay"] = (
        1 / (1 + payment["hours_since_failure"])
    )

    payment["customer_activity_score"] = (
        payment["successful_payments"]
        / (payment["customer_tenure_days"] + 1)
    )

    return payment


# ============================================================
# RECOVERY RECOMMENDATION ENGINE
# ============================================================

def recommend(
    payment,
    model_path="models/recovery_model.joblib",
):

    model = joblib.load(model_path)

    payment = add_features(payment)

    candidates = []

    # --------------------------------------------------------
    # Evaluate every possible action
    # --------------------------------------------------------

    for action in ACTIONS:

        row = payment.copy()
        row["action"] = action

        x = pd.DataFrame([row])

        probability = float(
            model.predict_proba(x)[0, 1]
        )

        cost = action_cost(
            payment["amount_inr"],
            action,
        )

        expected_value = expected_net_value(
            probability,
            payment["amount_inr"],
            action,
        )

        allowed, reason, guardrail_code = guardrail(
            payment,
            action,
        )

        candidates.append({
            "action": action,
            "probability": probability,
            "action_cost_inr": cost,
            "expected_value_inr": expected_value,
            "allowed": allowed,
            "guardrail_reason": reason,
            "guardrail_code": guardrail_code
        })

    # --------------------------------------------------------
    # HARD BUSINESS RULE
    # High-value + exhausted retries = manual review
    # --------------------------------------------------------

    if (
        payment["amount_inr"]
        >= HIGH_VALUE_RETRY_LIMIT_INR
        and payment["previous_attempts"]
        >= MAX_RETRY_ATTEMPTS
    ):

        return {
            "decision": "escalate",
            "decision_type": "manual_review_override",
            "confidence": 1.0,
            "expected_net_recovery_inr": None,
            "decision_reason": (
                "High-value payment has reached the "
                "maximum retry threshold; manual review required."
            ),
            "candidates": candidates,
        }

    # --------------------------------------------------------
    # Keep only guardrail-approved actions
    # --------------------------------------------------------

    allowed = [
        item
        for item in candidates
        if item["allowed"]
    ]

    # --------------------------------------------------------
    # Nothing allowed -> manual review
    # --------------------------------------------------------

    if not allowed:

        return {
            "decision": "escalate",
            "decision_type": "fallback_manual_review",
            "confidence": 0.0,
            "expected_net_recovery_inr": None,
            "decision_reason": (
                "No recovery action passed the configured guardrails."
            ),
            "candidates": candidates,
        }

    # --------------------------------------------------------
    # Escalation has no measured financial estimate.
    # Rank automated actions only.
    # --------------------------------------------------------

    automated = [
        item
        for item in allowed
        if item["action"] != "escalate"
    ]

    if automated:

        best = max(
            automated,
            key=lambda x: x["expected_value_inr"],
        )

    else:
        best = None

    # --------------------------------------------------------
    # Low-confidence decisions -> manual review
    # --------------------------------------------------------

    if (
        best is None
        or best["probability"]
        < LOW_PROBABILITY_THRESHOLD
    ):

        escalation = next(
            (
                item
                for item in allowed
                if item["action"] == "escalate"
            ),
            None,
        )

        if escalation is not None:

            return {
                "decision": "escalate",
                "decision_type": "low_confidence_manual_review",
                "confidence": (
                    best["probability"]
                    if best is not None
                    else 0.0
                ),
                "expected_net_recovery_inr": None,
                "decision_reason": (
                    "No automated action reached the "
                    "minimum recovery-probability threshold."
                ),
                "candidates": candidates,
            }

        return {
            "decision": "no_action",
            "decision_type": "low_confidence",
            "confidence": (
                best["probability"]
                if best is not None
                else 0.0
            ),
            "expected_net_recovery_inr": 0.0,
            "decision_reason": (
                "No automated action reached the "
                "minimum recovery-probability threshold."
            ),
            "candidates": candidates,
        }

    # --------------------------------------------------------
    # Final automated decision
    # --------------------------------------------------------

    return {
        "decision": best["action"],
        "decision_type": "automated_recovery",
        "confidence": best["probability"],
        "expected_value_inr": max(
            best["expected_value_inr"],
            0.0,
        ),
        "decision_reason": (
            "Selected the allowed automated action "
            "with the highest expected net recovery."
        ),
        "candidates": candidates,
    }


if __name__ == "__main__":
    print("agent.py loaded successfully.")
    print("Business configuration loaded successfully.")
    print("Production model was NOT modified.")
    print("Use recommend(payment) to generate a decision.")
