import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# ACTION INTERACTION EXPERIMENT
#
# Goal:
# Test whether explicit payment/customer × action features
# improve recovery prediction.
#
# IMPORTANT:
# - Production model is NOT changed.
# - Same grouped test split is used.
# - Same test payments are used for both models.
# ============================================================

DATA = "data/recovery_training_data.csv"
RANDOM_STATE = 42


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_base_features(df):

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


def add_action_interactions(df):

    df = df.copy()

    # --------------------------------------------------------
    # Numeric × action interactions
    #
    # These allow the model to learn whether an action behaves
    # differently for different payment/customer situations.
    # --------------------------------------------------------

    df["amount_action"] = (
        df["amount_inr"].astype(str)
        + "_"
        + df["action"].astype(str)
    )

    df["failure_rate_action"] = (
        df["failure_rate"].round(2).astype(str)
        + "_"
        + df["action"].astype(str)
    )

    df["success_rate_action"] = (
        df["payment_success_rate"].round(2).astype(str)
        + "_"
        + df["action"].astype(str)
    )

    df["prior_recovery_action"] = (
        df["prior_recovery_rate"].round(2).astype(str)
        + "_"
        + df["action"].astype(str)
    )

    df["attempts_action"] = (
        df["previous_attempts"].astype(str)
        + "_"
        + df["action"].astype(str)
    )

    # --------------------------------------------------------
    # Categorical × action interactions
    # --------------------------------------------------------

    df["failure_reason_action"] = (
        df["failure_reason"].astype(str)
        + "_"
        + df["action"].astype(str)
    )

    df["payment_method_action"] = (
        df["payment_method"].astype(str)
        + "_"
        + df["action"].astype(str)
    )

    return df


# ============================================================
# LOAD + FEATURES
# ============================================================

df = pd.read_csv(DATA)

df = add_base_features(df)

print("Dataset:", df.shape)
print(
    "Unique payments:",
    df["payment_id"].nunique()
)


# ============================================================
# BASE FEATURES
# ============================================================

base_features = [
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
    "action"
]


# ============================================================
# ADD INTERACTIONS
# ============================================================

df = add_action_interactions(df)

interaction_features = [
    "amount_action",
    "failure_rate_action",
    "success_rate_action",
    "prior_recovery_action",
    "attempts_action",
    "failure_reason_action",
    "payment_method_action"
]


interaction_feature_set = (
    base_features
    + interaction_features
)


target = "recovered"

groups = df["payment_id"]


# ============================================================
# SAME GROUPED TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        df[base_features],
        df[target],
        groups=groups
    )
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    features,
    categorical
):

    numeric = [
        col
        for col in features
        if col not in categorical
    ]

    X_train = df.iloc[train_idx][features]
    X_test = df.iloc[test_idx][features]

    y_train = df.iloc[train_idx][target]
    y_test = df.iloc[test_idx][target]

    preprocess = ColumnTransformer([
        (
            "numeric",
            "passthrough",
            numeric
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical
        )
    ])

    model = Pipeline([
        (
            "preprocess",
            preprocess
        ),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    prediction = (
        probability >= 0.50
    ).astype(int)

    return {
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            prediction
        ),
        "Precision": precision_score(
            y_test,
            prediction,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            prediction,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            prediction,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probability
        )
    }


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

base_categorical = [
    "failure_reason",
    "payment_method",
    "action"
]

interaction_categorical = [
    "failure_reason_action",
    "payment_method_action",
    "amount_action",
    "failure_rate_action",
    "success_rate_action",
    "prior_recovery_action",
    "attempts_action"
]


# ============================================================
# RUN BOTH MODELS
# ============================================================

print()
print("==============================")
print("MODEL 1: CURRENT FEATURES")
print("==============================")


baseline_result = evaluate_model(
    "Current Extra Trees",
    base_features,
    base_categorical
)


print()
print("==============================")
print("MODEL 2: ACTION INTERACTIONS")
print("==============================")


interaction_result = evaluate_model(
    "Extra Trees + Action Interactions",
    interaction_feature_set,
    (
        base_categorical
        + interaction_categorical
    )
)


# ============================================================
# COMPARISON
# ============================================================

results = pd.DataFrame([
    baseline_result,
    interaction_result
])


print()
print("==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    results
    .round(4)
    .to_string(index=False)
)


# ============================================================
# IMPROVEMENT
# ============================================================

baseline_auc = baseline_result["ROC_AUC"]
interaction_auc = interaction_result["ROC_AUC"]

baseline_f1 = baseline_result["F1"]
interaction_f1 = interaction_result["F1"]


print()
print("==============================")
print("IMPROVEMENT")
print("==============================")


print(
    "ROC-AUC change:",
    round(
        interaction_auc - baseline_auc,
        4
    )
)

print(
    "F1 change:",
    round(
        interaction_f1 - baseline_f1,
        4
    )
)


# ============================================================
# ACTION-SPECIFIC COMPARISON
# ============================================================

print()
print("==============================")
print("ACTION-SPECIFIC ROC-AUC")
print("==============================")


def action_auc(
    features,
    categorical
):

    numeric = [
        col
        for col in features
        if col not in categorical
    ]

    X_train = df.iloc[train_idx][features]
    X_test = df.iloc[test_idx][features]

    y_train = df.iloc[train_idx][target]
    y_test = df.iloc[test_idx][target]

    preprocess = ColumnTransformer([
        (
            "numeric",
            "passthrough",
            numeric
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical
        )
    ])

    model = Pipeline([
        (
            "preprocess",
            preprocess
        ),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    output = {}

    for action in sorted(
        X_test["action"].unique()
    ):

        mask = (
            X_test["action"]
            == action
        )

        if y_test[mask].nunique() < 2:
            output[action] = None
        else:
            output[action] = roc_auc_score(
                y_test[mask],
                probability[mask]
            )

    return output


baseline_action_auc = action_auc(
    base_features,
    base_categorical
)

interaction_action_auc = action_auc(
    interaction_feature_set,
    (
        base_categorical
        + interaction_categorical
    )
)


for action in sorted(
    baseline_action_auc.keys()
):

    before = baseline_action_auc[action]
    after = interaction_action_auc[action]

    print(
        f"{action}: "
        f"{before:.4f} -> {after:.4f}"
    )


print()
print("==============================")
print("EXPERIMENT COMPLETED")
print("==============================")

print(
    "Production model was NOT changed."
)

print(
    "No model file was overwritten."
)
