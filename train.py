import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.ensemble import ExtraTreesClassifier


DATA = "data/recovery_training_data.csv"
MODEL = "models/recovery_model.joblib"


def add_features(df):

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


df = pd.read_csv(DATA)

print("Dataset:", df.shape)
print(
    "Unique payments:",
    df["payment_id"].nunique()
)


df = add_features(df)


features = [
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


X = df[features]
y = df["recovered"]

categorical = [
    "failure_reason",
    "payment_method",
    "action"
]

numeric = [
    col
    for col in features
    if col not in categorical
]


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
            random_state=42,
            n_jobs=-1
        )
    )
])


# Group split prevents the same payment
# from appearing in both train and test.
groups = df["payment_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print(
    "Training payments:",
    groups.iloc[train_idx].nunique()
)

print(
    "Test payments:",
    groups.iloc[test_idx].nunique()
)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    prob
)


print("\nMODEL PERFORMANCE")
print("-----------------")

print(
    "Accuracy :",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1 Score :",
    round(f1, 4)
)

print(
    "ROC-AUC  :",
    round(auc, 4)
)


os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL
)

print("\nModel saved to:")
print(MODEL)

print("\nTraining completed successfully.")