import joblib
import pandas as pd
import numpy as np

from datetime import datetime

from agent import ACTIONS, guardrail
from business_config import (
    ACTION_COST_RATE,
    LOW_PROBABILITY_THRESHOLD,
)
# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/recovery_model.joblib"


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_features_to_batch(df):

    df = df.copy()

    total_attempts = (
        df["successful_payments"]
        + df["failed_payments"]
    )

    df["total_payment_attempts"] = total_attempts

    df["payment_success_rate"] = (
        df["successful_payments"]
        / total_attempts.clip(lower=1)
    )

    df["failure_rate"] = (
        df["failed_payments"]
        / total_attempts.clip(lower=1)
    )

    df["retry_exhaustion"] = (
        df["previous_attempts"] / 3
    ).clip(upper=1)

    df["transaction_to_customer_value"] = (
        df["amount_inr"]
        / df["customer_value_inr"].clip(lower=1)
    )

    df["time_decay"] = (
        1 / (1 + df["hours_since_failure"])
    )

    df["customer_activity_score"] = (
        df["successful_payments"]
        / (df["customer_tenure_days"] + 1)
    )

    return df


# =========================================================
# BATCH ANALYSIS
# =========================================================

def analyze_batch(df):

    df = df.copy()

    # -----------------------------------------------------
    # ONE RECORD PER PAYMENT
    # -----------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset="payment_id"
        )
        .reset_index(drop=True)
    )

    if df.empty:
        return pd.DataFrame()

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    df = add_features_to_batch(df)

    # -----------------------------------------------------
    # CREATE ACTION CANDIDATES
    # -----------------------------------------------------

    candidates = []

    for action in ACTIONS:

        temp = df.copy()

        temp["action"] = action

        candidates.append(temp)

    candidates = pd.concat(
        candidates,
        ignore_index=True
    )

    # -----------------------------------------------------
    # LOAD MODEL ONCE
    # -----------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # -----------------------------------------------------
    # PREDICT RECOVERY PROBABILITY
    # -----------------------------------------------------

    candidates["probability"] = (
        model
        .predict_proba(candidates)[:, 1]
        .clip(0, 1)
    )

    # -----------------------------------------------------
    # APPLY GUARDRAILS
    # -----------------------------------------------------

    allowed = []
    reasons = []

    for _, row in candidates.iterrows():

        ok, reason, code = guardrail(
            row,
            row["action"]
        )

        allowed.append(ok)
        reasons.append(reason)

    candidates["allowed"] = allowed
    candidates["guardrail_reason"] = reasons

    # -----------------------------------------------------
    # EXPECTED NET RECOVERY OPPORTUNITY
    #
    # Probability × payment amount
    # minus an estimated action cost.
    #
    # Action costs are modeled as percentages of the payment
    # amount instead of tiny fixed rupee values. These are
    # configurable prototype assumptions, not actual Razorpay
    # operating costs.
    # -----------------------------------------------------


    candidates["estimated_action_cost_inr"] = (
        candidates["amount_inr"]
        * candidates["action"].map(ACTION_COST_RATE).fillna(0)
    )

    candidates["expected_net_value_inr"] = (
        candidates["probability"]
        * candidates["amount_inr"]
        - candidates["estimated_action_cost_inr"]
    )

    # Escalation is manual review.
    # It is not treated as automated recovery.

    candidates.loc[
        candidates["action"] == "escalate",
        "expected_net_value_inr"
    ] = 0

    # -----------------------------------------------------
    # BLOCK DISALLOWED ACTIONS
    # -----------------------------------------------------

    candidates["selection_value"] = np.where(
        candidates["allowed"],
        candidates["expected_net_value_inr"],
        -np.inf
    )

    # -----------------------------------------------------
    # SELECT BEST ACTION
    # -----------------------------------------------------

    best_index = (
        candidates
        .groupby("payment_id")[
            "selection_value"
        ]
        .idxmax()
    )

    results = (
        candidates
        .loc[best_index]
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # LOW-PROBABILITY STOPPING RULE
    # -----------------------------------------------------

    results["recommended_action"] = (
        results["action"]
    )

    low_probability = (
        results["probability"] < LOW_PROBABILITY_THRESHOLD
    )

    results.loc[
        low_probability,
        "recommended_action"
    ] = "no_action"

    # -----------------------------------------------------
    # RECOVERY PROBABILITY
    # -----------------------------------------------------

    results["recovery_probability"] = (
        results["probability"]
        .clip(0, 1)
    )

    # -----------------------------------------------------
    # EXPECTED NET RECOVERY VALUE
    # -----------------------------------------------------

    results["expected_net_recovery_inr"] = (
        results["expected_net_value_inr"]
        .clip(lower=0)
    )

    if "estimated_action_cost_inr" not in results.columns:
        results["estimated_action_cost_inr"] = 0.0

    # No action = no expected actionable recovery.

    results.loc[
        results["recommended_action"]
        == "no_action",
        "expected_net_recovery_inr"
    ] = 0

    # Manual review is not automated recovery.

    results.loc[
        results["recommended_action"]
        == "escalate",
        "expected_net_recovery_inr"
    ] = 0

    # -----------------------------------------------------
    # RETURN CLEAN RESULTS
    # -----------------------------------------------------

    return results[
        [
            "payment_id",
            "amount_inr",
            "failure_reason",
            "payment_method",
            "recommended_action",
            "recovery_probability",
            "expected_net_recovery_inr",
            "estimated_action_cost_inr"
        ]
    ].copy()


# =========================================================
# SIMULATED RECOVERY WORKFLOW
# =========================================================

def execute_recovery_workflow(results):
    """
    Execute a SAFE OFFLINE simulation of the recommended recovery actions.

    IMPORTANT:
    No real payment, customer, or financial transaction is executed.
    Recovery outcomes are simulated using the model probability.
    """

    results = results.copy()

    audit = []

    # -----------------------------------------------------
    # DEFAULT EXECUTION FIELDS
    # -----------------------------------------------------

    results["execution_status"] = "skipped"

    results["execution_result"] = "no_action_required"

    results["simulated_recovered"] = False

    results["simulated_recovery_amount_inr"] = 0.0

    results["simulated_net_recovery_inr"] = 0.0

    # -----------------------------------------------------
    # PROCESS EACH PAYMENT
    # -----------------------------------------------------

    for index, row in results.iterrows():

        payment_id = str(
            row["payment_id"]
        )

        action = str(
            row["recommended_action"]
        ).lower()

        amount = float(
            row["amount_inr"]
        )

        probability = float(
            row["recovery_probability"]
        )

        expected_value = float(
            row.get(
                "expected_net_recovery_inr",
                row.get(
                    "expected_value_inr",
                    0.0
                )
            )
        )

        # -------------------------------------------------
        # NO ACTION
        # -------------------------------------------------

        if action == "no_action":

            status = "skipped"

            execution_result = (
                "no_action_required"
            )

            recovered = False

            recovery_amount = 0.0

            net_recovery = 0.0

        # -------------------------------------------------
        # MANUAL REVIEW
        # -------------------------------------------------

        elif action == "escalate":

            status = "manual_review"

            execution_result = (
                "routed_to_manual_review"
            )

            recovered = False

            recovery_amount = 0.0

            net_recovery = 0.0

        # -------------------------------------------------
        # AUTOMATED ACTION
        # -------------------------------------------------

        else:

            status = "simulated"

            execution_result = (
                f"{action}_simulated"
            )

            # -------------------------------------------------
            # DETERMINISTIC PROBABILITY-BASED SIMULATION
            # -------------------------------------------------

            import hashlib

            seed = int(
                hashlib.sha256(
                    payment_id.encode("utf-8")
                ).hexdigest()[:8],
                16
            )

            simulated_draw = (
                seed / 0xFFFFFFFF
            )

            recovered = (
                simulated_draw < probability
            )

            # -------------------------------------------------
            # ACTION COST
            # -------------------------------------------------

            action_cost_rate = {
                "payment_link": 0.010,
                "reminder": 0.005,
                "retry": 0.015,
            }.get(action, 0.0)

            action_cost = (
                amount * action_cost_rate
            )

            # -------------------------------------------------
            # SIMULATED RECOVERY RESULT
            # -------------------------------------------------

            if recovered:

                recovery_amount = amount

                net_recovery = (
                    recovery_amount
                    - action_cost
                )

            else:

                recovery_amount = 0.0

                net_recovery = (
                    -action_cost
                )

        # -------------------------------------------------
        # STORE SIMULATION RESULT
        # -------------------------------------------------

        results.at[
            index,
            "execution_status"
        ] = status

        results.at[
            index,
            "execution_result"
        ] = execution_result

        results.at[
            index,
            "simulated_recovered"
        ] = recovered

        results.at[
            index,
            "simulated_recovery_amount_inr"
        ] = recovery_amount

        results.at[
            index,
            "simulated_net_recovery_inr"
        ] = net_recovery

        # -------------------------------------------------
        # AUDIT RECORD
        # -------------------------------------------------

        audit.append({

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "payment_id":
                payment_id,

            "amount_inr":
                amount,

            "recommended_action":
                action,

            "recovery_probability":
                probability,

            "expected_net_recovery_inr":
                expected_value,

            "execution_status":
                status,

            "execution_result":
                execution_result,

            "simulated_recovered":
                recovered,

            "simulated_recovery_amount_inr":
                recovery_amount,

            "simulated_net_recovery_inr":
                net_recovery
        })

    return (
        results,
        pd.DataFrame(audit)
    )