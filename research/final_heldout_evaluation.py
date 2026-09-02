"""
Final held-out policy evaluation for the Razorpay Revenue Recovery Agent.

Purpose
-------
Evaluate the current production decision logic on completely unseen payment IDs.

Design
------
1. Split by payment_id (80/20) so no payment appears in both train and test.
2. Train the same Extra Trees pipeline used by train.py using ONLY training payments.
3. Predict every candidate action for ONLY the unseen test payments.
4. Apply the canonical business_config.guardrail() to candidate actions.
5. Select the AI action using expected_net_value() and the configured probability threshold.
6. Attach the OBSERVED recovered outcome for the selected payment/action pair.
7. Compare against:
   - highest predicted probability
   - historical best action (training data only)
   - random allowed action
   - oracle upper bound (reference only)
8. Bootstrap the AI-vs-baseline net recovery difference across test payments.

Important
---------
- This evaluates observed synthetic outcomes in a held-out test set.
- It does NOT execute real Razorpay transactions.
- The oracle is not deployable.
- Action costs come from business_config.py.
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "recovery_training_data.csv"
RESULTS_DIR = ROOT / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
BOOTSTRAP_ITERATIONS = 5000


# ---------------------------------------------------------------------
# CANONICAL BUSINESS POLICY
# ---------------------------------------------------------------------

from business_config import (  # noqa: E402
    ACTIONS,
    LOW_PROBABILITY_THRESHOLD,
    action_cost,
    expected_net_value,
    guardrail,
)


# ---------------------------------------------------------------------
# MODEL FEATURES — SAME AS train.py
# ---------------------------------------------------------------------

FEATURES = [
    "amount_inr",
    "failure_reason",
    "payment_method",
    "successful_payments",
    "failed_payments",
    "previous_attempts",
    "hours_since_failure",
    "customer_tenure_days",
    "prior_recovery_rate",
    "customer_value_inr",
    "failure_hour",
    "total_payment_attempts",
    "payment_success_rate",
    "failure_rate",
    "retry_exhaustion",
    "transaction_to_customer_value",
    "time_decay",
    "customer_activity_score",
    "action",
]

CATEGORICAL = [
    "failure_reason",
    "payment_method",
    "action",
]

NUMERIC = [
    col for col in FEATURES
    if col not in CATEGORICAL
]


# ---------------------------------------------------------------------
# FEATURE ENGINEERING — SAME AS train.py
# ---------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    total = (
        df["successful_payments"]
        + df["failed_payments"]
    )

    df["total_payment_attempts"] = total

    df["payment_success_rate"] = (
        df["successful_payments"]
        / total.clip(lower=1)
    )

    df["failure_rate"] = (
        df["failed_payments"]
        / total.clip(lower=1)
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


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

def build_model() -> Pipeline:
    preprocess = ColumnTransformer([
        (
            "numeric",
            "passthrough",
            NUMERIC,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
    ])

    return Pipeline([
        (
            "preprocess",
            preprocess,
        ),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


# ---------------------------------------------------------------------
# POLICY HELPERS
# ---------------------------------------------------------------------

def choose_ai_action(candidates: pd.DataFrame) -> pd.Series:
    allowed = candidates[candidates["allowed"]].copy()

    automated = allowed[
        allowed["action"] != "escalate"
    ].copy()

    if automated.empty:
        escalation = allowed[
            allowed["action"] == "escalate"
        ]
        return escalation.iloc[0]

    best = (
        automated
        .sort_values(
            ["expected_net_recovery", "predicted_probability"],
            ascending=[False, False],
        )
        .iloc[0]
    )

    if (
        float(best["predicted_probability"])
        < LOW_PROBABILITY_THRESHOLD
    ):
        escalation = allowed[
            allowed["action"] == "escalate"
        ]
        if not escalation.empty:
            return escalation.iloc[0]

    return best


def choose_probability_action(candidates: pd.DataFrame) -> pd.Series:
    allowed = candidates[candidates["allowed"]].copy()

    return (
        allowed
        .sort_values(
            ["predicted_probability", "expected_net_recovery"],
            ascending=[False, False],
        )
        .iloc[0]
    )


def choose_historical_action(
    candidates: pd.DataFrame,
    historical_rates: pd.Series,
) -> pd.Series:
    allowed = candidates[candidates["allowed"]].copy()
    allowed["historical_rate"] = (
        allowed["action"].map(historical_rates).fillna(0.0)
    )

    return (
        allowed
        .sort_values(
            ["historical_rate", "predicted_probability"],
            ascending=[False, False],
        )
        .iloc[0]
    )


def apply_policy(
    candidates: pd.DataFrame,
    chooser,
    **kwargs,
) -> pd.Series:
    rows = []

    for payment_id, group in candidates.groupby(
        "payment_id",
        sort=False,
    ):
        chosen = chooser(group, **kwargs)
        record = chosen.to_dict()
        record["payment_id"] = payment_id
        rows.append(record)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# OUTCOME CALCULATION
# ---------------------------------------------------------------------

def attach_observed_outcomes(
    selected: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    actual = (
        test[
            [
                "payment_id",
                "action",
                "recovered",
            ]
        ]
        .drop_duplicates(["payment_id", "action"])
        .rename(
            columns={
                "recovered": "actual_recovered",
            }
        )
    )

    result = selected.merge(
        actual,
        on=["payment_id", "action"],
        how="left",
        validate="one_to_one",
    )

    if result["actual_recovered"].isna().any():
        missing = int(result["actual_recovered"].isna().sum())
        raise RuntimeError(
            f"Could not attach observed outcomes for {missing} "
            "selected payment/action pairs."
        )

    result["actual_recovered"] = (
        result["actual_recovered"]
        .astype(int)
    )

    result["actual_recovered_value"] = (
        result["amount_inr"]
        * result["actual_recovered"]
    )

    result["actual_action_cost"] = (
        result.apply(
            lambda row: action_cost(
                row["amount_inr"],
                row["action"],
            ),
            axis=1,
        )
    )

    result["actual_net_recovery"] = (
        result["actual_recovered_value"]
        - result["actual_action_cost"]
    )

    return result


def summarize_policy(
    policy_name: str,
    result: pd.DataFrame,
) -> dict:
    total_failed_value = float(
        result["amount_inr"].sum()
    )
    gross_recovered = float(
        result["actual_recovered_value"].sum()
    )
    action_cost_total = float(
        result["actual_action_cost"].sum()
    )
    net_recovery = float(
        result["actual_net_recovery"].sum()
    )
    recovery_rate = float(
        result["actual_recovered"].mean()
    )

    return {
        "Policy": policy_name,
        "Test Payments": int(len(result)),
        "Total Failed Value": total_failed_value,
        "Recovery Rate": recovery_rate,
        "Gross Recovered Value": gross_recovered,
        "Action Cost": action_cost_total,
        "Net Recovery": net_recovery,
    }


def bootstrap_ci(
    differences: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 12345,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(differences)

    means = np.empty(iterations)

    for i in range(iterations):
        sample = rng.choice(
            differences,
            size=n,
            replace=True,
        )
        means[i] = sample.mean()

    lower, upper = np.percentile(
        means,
        [2.5, 97.5],
    )

    return (
        float(differences.mean()),
        float(lower),
        float(upper),
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("FINAL HELD-OUT POLICY EVALUATION")
    print("=" * 72)

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    df = pd.read_csv(DATA)

    # Only raw columns must exist in the CSV.
    # The engineered features below are created by add_features().
    RAW_REQUIRED_COLUMNS = [
        "payment_id",
        "amount_inr",
        "failure_reason",
        "payment_method",
        "successful_payments",
        "failed_payments",
        "previous_attempts",
        "hours_since_failure",
        "customer_tenure_days",
        "prior_recovery_rate",
        "customer_value_inr",
        "failure_hour",
        "action",
        "recovered",
    ]

    missing_columns = sorted(
        set(RAW_REQUIRED_COLUMNS) - set(df.columns)
    )
    if missing_columns:
        raise ValueError(
            "Dataset is missing required raw columns: "
            + ", ".join(missing_columns)
        )

    # Create engineered model features after validating the raw dataset.
    df = add_features(df)

    # Verify that feature engineering produced every model feature.
    missing_features = sorted(
        set(FEATURES) - set(df.columns)
    )
    if missing_features:
        raise ValueError(
            "Feature engineering did not create required model features: "
            + ", ".join(missing_features)
        )

    print(f"Dataset rows: {len(df):,}")
    print(
        f"Unique payments: "
        f"{df['payment_id'].nunique():,}"
    )

    # ---------------------------------------------------------------
    # GROUPED SPLIT
    # ---------------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df[FEATURES],
            df["recovered"],
            groups=df["payment_id"],
        )
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    train_payment_ids = set(
        train["payment_id"]
    )
    test_payment_ids = set(
        test["payment_id"]
    )

    overlap = train_payment_ids & test_payment_ids
    if overlap:
        raise RuntimeError(
            f"Payment leakage detected: {len(overlap)} IDs overlap."
        )

    print()
    print("-" * 72)
    print("HELD-OUT SPLIT")
    print("-" * 72)
    print(
        f"Training payments: {len(train_payment_ids):,}"
    )
    print(
        f"Test payments:     {len(test_payment_ids):,}"
    )
    print(
        f"Payment overlap:   {len(overlap)}"
    )

    # ---------------------------------------------------------------
    # TRAIN ONLY ON TRAINING PAYMENTS
    # ---------------------------------------------------------------

    model = build_model()

    print()
    print("Training model on training payments only...")
    model.fit(
        train[FEATURES],
        train["recovered"],
    )
    print("Training complete.")

    # ---------------------------------------------------------------
    # ONE BASE ROW PER TEST PAYMENT
    # ---------------------------------------------------------------

    base_test = (
        test
        .sort_values(["payment_id", "action"])
        .groupby("payment_id")
        .first()
        .reset_index()
    )

    print(
        f"Candidate test payments: "
        f"{len(base_test):,}"
    )

    # ---------------------------------------------------------------
    # SCORE ALL ACTIONS
    # ---------------------------------------------------------------

    candidate_rows = []

    for _, payment in base_test.iterrows():
        for action in ACTIONS:
            row = payment.copy()
            row["action"] = action
            candidate_rows.append(row)

    candidates = pd.DataFrame(candidate_rows)

    candidates["predicted_probability"] = (
        model.predict_proba(
            candidates[FEATURES]
        )[:, 1]
    )

    candidates["expected_net_recovery"] = (
        candidates.apply(
            lambda row: expected_net_value(
                row["predicted_probability"],
                row["amount_inr"],
                row["action"],
            ),
            axis=1,
        )
    )

    guardrail_results = [
        guardrail(
            row,
            row["action"],
        )
        for _, row in candidates.iterrows()
    ]

    candidates["allowed"] = [
        result[0]
        for result in guardrail_results
    ]
    candidates["guardrail_reason"] = [
        result[1]
        for result in guardrail_results
    ]
    candidates["guardrail_code"] = [
        result[2]
        for result in guardrail_results
    ]

    # ---------------------------------------------------------------
    # POLICIES
    # ---------------------------------------------------------------

    ai_selected = apply_policy(
        candidates,
        choose_ai_action,
    )

    probability_selected = apply_policy(
        candidates,
        choose_probability_action,
    )

    historical_rates = (
        train
        .groupby("action")["recovered"]
        .mean()
    )

    historical_selected = apply_policy(
        candidates,
        choose_historical_action,
        historical_rates=historical_rates,
    )

    allowed_candidates = candidates[
        candidates["allowed"]
    ].copy()

    rng = np.random.default_rng(RANDOM_STATE)
    random_rows = []

    for payment_id, group in allowed_candidates.groupby(
        "payment_id",
        sort=False,
    ):
        idx = int(
            rng.integers(
                low=0,
                high=len(group),
            )
        )
        chosen = group.iloc[idx]
        random_rows.append(
            chosen.to_dict()
        )

    random_selected = pd.DataFrame(random_rows)

    # ---------------------------------------------------------------
    # ORACLE — REFERENCE ONLY
    # ---------------------------------------------------------------

    oracle_source = (
        test[
            [
                "payment_id",
                "action",
                "amount_inr",
                "recovered",
            ]
        ]
        .drop_duplicates(
            ["payment_id", "action"]
        )
        .copy()
    )

    oracle_source["actual_action_cost"] = (
        oracle_source.apply(
            lambda row: action_cost(
                row["amount_inr"],
                row["action"],
            ),
            axis=1,
        )
    )

    oracle_source["actual_net_recovery"] = (
        oracle_source["amount_inr"]
        * oracle_source["recovered"]
        - oracle_source["actual_action_cost"]
    )

    oracle_selected = (
        oracle_source
        .sort_values(
            [
                "payment_id",
                "actual_net_recovery",
            ],
            ascending=[True, False],
        )
        .groupby("payment_id")
        .first()
        .reset_index()
    )

    oracle_selected["actual_recovered"] = (
        oracle_selected["recovered"]
        .astype(int)
    )
    oracle_selected["actual_recovered_value"] = (
        oracle_selected["amount_inr"]
        * oracle_selected["actual_recovered"]
    )

    oracle_selected["predicted_probability"] = np.nan
    oracle_selected["expected_net_recovery"] = np.nan
    oracle_selected["allowed"] = True
    oracle_selected["guardrail_reason"] = "Oracle reference only."
    oracle_selected["guardrail_code"] = "ORACLE"
    oracle_selected["action_cost_inr"] = (
        oracle_selected["actual_action_cost"]
    )
    oracle_selected["actual_action_cost"] = (
        oracle_selected["actual_action_cost"]
    )

    # Attach observed outcomes to deployable policies.
    ai_result = attach_observed_outcomes(
        ai_selected,
        test,
    )
    probability_result = attach_observed_outcomes(
        probability_selected,
        test,
    )
    historical_result = attach_observed_outcomes(
        historical_selected,
        test,
    )
    random_result = attach_observed_outcomes(
        random_selected,
        test,
    )

    # ---------------------------------------------------------------
    # SUMMARIES
    # ---------------------------------------------------------------

    summaries = [
        summarize_policy(
            "AI expected-net-recovery",
            ai_result,
        ),
        summarize_policy(
            "Highest probability",
            probability_result,
        ),
        summarize_policy(
            "Historical best action",
            historical_result,
        ),
        summarize_policy(
            "Random allowed action",
            random_result,
        ),
        summarize_policy(
            "Oracle upper bound",
            oracle_selected,
        ),
    ]

    summary = pd.DataFrame(summaries)

    random_net = float(
        summary.loc[
            summary["Policy"]
            == "Random allowed action",
            "Net Recovery",
        ].iloc[0]
    )

    summary["vs Random Net Lift"] = (
        summary["Net Recovery"]
        - random_net
    )

    summary["vs Random Improvement %"] = (
        summary["vs Random Net Lift"]
        / abs(random_net)
        * 100
    )

    # ---------------------------------------------------------------
    # PAIRED BOOTSTRAP
    # ---------------------------------------------------------------

    paired = ai_result[
        [
            "payment_id",
            "actual_net_recovery",
        ]
    ].merge(
        probability_result[
            [
                "payment_id",
                "actual_net_recovery",
            ]
        ],
        on="payment_id",
        suffixes=("_ai", "_probability"),
        validate="one_to_one",
    )

    paired["ai_minus_probability"] = (
        paired["actual_net_recovery_ai"]
        - paired["actual_net_recovery_probability"]
    )

    mean_diff, ci_low, ci_high = bootstrap_ci(
        paired["ai_minus_probability"].to_numpy(
            dtype=float
        )
    )

    # Also compute AI vs random bootstrap interval.
    ai_random_paired = ai_result[
        [
            "payment_id",
            "actual_net_recovery",
        ]
    ].merge(
        random_result[
            [
                "payment_id",
                "actual_net_recovery",
            ]
        ],
        on="payment_id",
        suffixes=("_ai", "_random"),
        validate="one_to_one",
    )

    ai_random_paired["ai_minus_random"] = (
        ai_random_paired["actual_net_recovery_ai"]
        - ai_random_paired["actual_net_recovery_random"]
    )

    random_mean_diff, random_ci_low, random_ci_high = (
        bootstrap_ci(
            ai_random_paired["ai_minus_random"].to_numpy(
                dtype=float
            ),
            seed=54321,
        )
    )

    # ---------------------------------------------------------------
    # GUARDRAIL STATS
    # ---------------------------------------------------------------

    blocked_retry = int(
        (
            (candidates["action"] == "retry")
            & (~candidates["allowed"])
        )
        .groupby(candidates["payment_id"])
        .any()
        .sum()
    )

    ai_manual_review = int(
        (
            ai_result["action"]
            .astype(str)
            .str.lower()
            .eq("escalate")
        ).sum()
    )

    ai_action_distribution = (
        ai_result["action"]
        .value_counts()
        .rename_axis("action")
        .reset_index(name="payments")
    )

    # ---------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------

    display_summary = summary.copy()

    money_cols = [
        "Total Failed Value",
        "Gross Recovered Value",
        "Action Cost",
        "Net Recovery",
        "vs Random Net Lift",
    ]

    for col in money_cols:
        display_summary[col] = (
            display_summary[col]
            .round(2)
        )

    display_summary["Recovery Rate"] = (
        display_summary["Recovery Rate"]
        .mul(100)
        .round(2)
    )

    display_summary["vs Random Improvement %"] = (
        display_summary["vs Random Improvement %"]
        .round(2)
    )

    print()
    print("=" * 72)
    print("OBSERVED OUTCOME — HELD-OUT TEST SET")
    print("=" * 72)
    print(display_summary.to_string(index=False))

    print()
    print("-" * 72)
    print("AI ACTION DISTRIBUTION")
    print("-" * 72)
    print(ai_action_distribution.to_string(index=False))

    print()
    print("-" * 72)
    print("GUARDRAILS")
    print("-" * 72)
    print(
        f"Payments with at least one blocked retry candidate: "
        f"{blocked_retry}"
    )
    print(
        f"AI-selected escalations on held-out payments: "
        f"{ai_manual_review}"
    )

    print()
    print("-" * 72)
    print("PAIRED BOOTSTRAP: AI vs HIGHEST PROBABILITY")
    print("-" * 72)
    print(
        f"Mean difference (AI - probability): ₹{mean_diff:,.2f}"
    )
    print(
        f"95% CI: ₹{ci_low:,.2f} to ₹{ci_high:,.2f}"
    )

    print()
    print("-" * 72)
    print("PAIRED BOOTSTRAP: AI vs RANDOM ALLOWED")
    print("-" * 72)
    print(
        f"Mean difference (AI - random allowed): "
        f"₹{random_mean_diff:,.2f}"
    )
    print(
        f"95% CI: ₹{random_ci_low:,.2f} to ₹{random_ci_high:,.2f}"
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    summary_path = (
        RESULTS_DIR / "final_heldout_policy_summary.csv"
    )
    paired_path = (
        RESULTS_DIR / "final_heldout_paired_results.csv"
    )
    candidate_path = (
        RESULTS_DIR / "final_heldout_action_candidates.csv"
    )
    ai_path = (
        RESULTS_DIR / "final_heldout_ai_decisions.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    candidates.to_csv(
        candidate_path,
        index=False,
    )

    ai_result.to_csv(
        ai_path,
        index=False,
    )

    metadata = {
        "dataset": str(DATA),
        "dataset_rows": int(len(df)),
        "unique_payments": int(df["payment_id"].nunique()),
        "training_payments": int(len(train_payment_ids)),
        "test_payments": int(len(test_payment_ids)),
        "payment_overlap": int(len(overlap)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "low_probability_threshold": float(
            LOW_PROBABILITY_THRESHOLD
        ),
        "actions": list(ACTIONS),
        "bootstrap_ai_minus_probability": {
            "mean": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "bootstrap_ai_minus_random_allowed": {
            "mean": random_mean_diff,
            "ci_low": random_ci_low,
            "ci_high": random_ci_high,
        },
        "note": (
            "Held-out evaluation uses observed synthetic outcomes from "
            "payment/action pairs that were not used for training. "
            "No real Razorpay transactions are executed."
        ),
    }

    metadata_path = (
        RESULTS_DIR / "final_heldout_evaluation_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("SAVED RESULTS")
    print("=" * 72)
    print(summary_path)
    print(paired_path)
    print(candidate_path)
    print(ai_path)
    print(metadata_path)
    print()
    print("Production model and agent were NOT changed.")


if __name__ == "__main__":
    main()
