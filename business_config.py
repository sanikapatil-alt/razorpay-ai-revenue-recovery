"""
Central business configuration for the Razorpay Revenue Recovery Agent.

IMPORTANT:
- Action costs are PROTOTYPE ASSUMPTIONS.
- They are NOT verified Razorpay charges.
- Replace them with real business economics when available.
- Model predictions must never be presented as actual recovered revenue.
"""

ACTIONS = [
    "payment_link",
    "reminder",
    "retry",
    "escalate",
]

ACTION_COST_RATE = {
    "payment_link": 0.0025,
    "reminder": 0.0010,
    "retry": 0.0050,
    "escalate": 0.0100,
}

MAX_RETRY_ATTEMPTS = 3
RECOVERY_WINDOW_HOURS = 48
HIGH_VALUE_RETRY_LIMIT_INR = 50_000

LOW_PROBABILITY_THRESHOLD = 0.25


def action_cost(amount_inr, action):
    """Return estimated prototype action cost."""

    return (
        float(amount_inr)
        * ACTION_COST_RATE.get(action, 0.0)
    )


def guardrail(payment, action):
    """
    Apply deterministic business rules.

    Returns:
        allowed, reason, guardrail_code
    """

    previous_attempts = float(
        payment["previous_attempts"]
    )

    hours_since_failure = float(
        payment["hours_since_failure"]
    )

    amount_inr = float(
        payment["amount_inr"]
    )

    # Retry limit
    if (
        action == "retry"
        and previous_attempts >= MAX_RETRY_ATTEMPTS
    ):
        return (
            False,
            "Retry blocked: maximum retry attempts reached.",
            "MAX_RETRY_ATTEMPTS",
        )

    # Recovery window
    if (
        action in {"retry", "reminder"}
        and hours_since_failure > RECOVERY_WINDOW_HOURS
    ):
        return (
            False,
            "Action blocked: recovery window is too old.",
            "RECOVERY_WINDOW",
        )

    # High-value retry
    if (
        action == "retry"
        and amount_inr >= HIGH_VALUE_RETRY_LIMIT_INR
    ):
        return (
            False,
            "High-value retry requires manual review.",
            "HIGH_VALUE_RETRY",
        )

    # Escalation is always allowed
    if action == "escalate":
        return (
            True,
            "Manual review allowed.",
            "MANUAL_REVIEW",
        )

    return (
        True,
        "Allowed.",
        "NONE",
    )


def expected_net_value(
    probability,
    amount_inr,
    action,
):
    """
    Calculate model-side expected net recovery.

    This is used ONLY for choosing an action.
    It is NOT measured recovered revenue.
    """

    if action == "escalate":
        return None

    expected_gross_recovery = (
        float(probability)
        * float(amount_inr)
    )

    cost = action_cost(
        amount_inr,
        action,
    )

    return (
        expected_gross_recovery
        - cost
    )