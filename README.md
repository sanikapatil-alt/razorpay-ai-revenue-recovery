# Razorpay Revenue Recovery Agent

An AI-assisted revenue recovery prototype for failed payments.

## 1. Problem

Failed payments create a direct revenue recovery opportunity, but repeatedly applying the same recovery action is inefficient.

For every failed payment, a recovery system needs to answer:

- How likely is this payment to be recovered?
- Which recovery action should be attempted?
- Is the action worth its cost?
- Should the payment be left alone?
- When should the case be escalated for manual review?

The goal of this project is to build a decision-making agent that answers these questions while applying explicit business guardrails.

---

## 2. Solution

The system combines:

1. Machine-learning recovery probability prediction
2. Expected-value based action selection
3. Deterministic business guardrails
4. Bounded recovery execution
5. Offline recovery simulation
6. Audit logging
7. A Streamlit dashboard for analysis

The decision flow is:

Failed Payment
       ↓
Customer / Payment Features
       ↓
Recovery Probability Model
       ↓
Evaluate Recovery Actions
       ↓
Apply Business Guardrails
       ↓
Expected Net Recovery
       ↓
Best Allowed Action
       ↓
Bounded Execution / Offline Simulation
       ↓
Audit Trail
       ↓
Dashboard

---

## 3. Recovery Actions

The agent evaluates four recovery actions:

- `retry`
- `payment_link`
- `reminder`
- `escalate`

The action is not selected from probability alone.

For automated actions, the system considers:

    Expected Net Recovery
    = Recovery Probability × Payment Amount
      − Estimated Action Cost

The action with the highest allowed expected net recovery is selected.

---

## 4. Machine Learning Model

The current model is a regularized Extra Trees classifier.

Configuration:

- 300 trees
- Maximum depth: 12
- Minimum samples per leaf: 5
- Balanced class weights
- Fixed random seed for reproducibility

The model predicts:

    P(recovery | payment context, action)

The model evaluates the same payment under each candidate action.

### Features

The model uses payment and customer context including:

- Payment amount
- Failure reason
- Payment method
- Successful payment history
- Failed payment history
- Previous recovery attempts
- Hours since failure
- Customer tenure
- Prior recovery rate
- Customer value
- Failure hour

It also derives additional features such as:

- Payment success rate
- Failure rate
- Retry exhaustion
- Transaction-to-customer-value ratio
- Time decay
- Customer activity score
- Total payment attempts

The training dataset is synthetic and is intended for prototyping the decision loop.

---

## 5. Business Guardrails

Machine learning does not have unrestricted control over recovery actions.

The policy layer applies deterministic guardrails.

### Retry limit

Retry is blocked when the maximum configured retry threshold has been reached.

### Recovery window

Retry and reminder are blocked when the payment is outside the configured recovery window.

### High-value payments

High-value payments cannot automatically be retried and can be routed for manual review.

### Low-confidence decisions

If no automated action reaches the minimum recovery probability threshold, the system can escalate instead of forcing an automated action.

These controls are intentionally separated from the ML model.

---

## 6. Expected Recovery vs Actual Recovery

The project deliberately distinguishes between:

### Expected Net Recovery

A model-side estimate used to rank recovery actions.

It is calculated from predicted probability, payment value and prototype action costs.

It is **not measured revenue**.

### Simulated Recovery

The project includes an offline simulation layer for evaluating the decision workflow.

The simulation uses the model's predicted recovery probability to generate reproducible hypothetical outcomes.

Simulated recovery is **not real customer recovery** and should not be interpreted as actual Razorpay revenue.

---

## 7. Evaluation

The project uses payment-grouped evaluation.

The same payment is never placed in both training and test sets.

### Model validation

The regularized Extra Trees model was selected on a validation set and then evaluated on a final held-out test set.

Final held-out results:

| Metric | Result |
|---|---:|
| Accuracy | 63.64% |
| Precision | 55.06% |
| Recall | 56.99% |
| F1 | 56.01% |
| ROC-AUC | 67.56% |
| PR-AUC | 57.28% |
| Brier Score | 0.2262 |

The current production model was not automatically replaced.

### Business policy test

On a held-out set of 700 payments:

| Policy | Recovery Rate | Net Recovery |
|---|---:|---:|
| AI expected-net-recovery | 56.86% | ₹2,060,322 |
| Highest probability | 56.71% | ₹2,062,294 |
| Historical best action | 45.00% | ₹1,643,393 |
| Random allowed action | 41.43% | ₹1,523,300 |
| Random action | 41.86% | ₹1,447,746 |

The AI policy produced a ₹612,575 higher net recovery than the random-action baseline in this test.

That corresponds to a 42.31% improvement over the random baseline.

### Important interpretation

The direct comparison between expected-net optimization and highest-probability selection was not statistically decisive.

A paired bootstrap test produced a 95% confidence interval of approximately:

    -₹28.85 to ₹16.42

Therefore, the project does not claim that expected-net optimization is statistically superior to probability ranking.

This is intentional: the evaluation is reported without overstating the result.

---

## 8. Auditability

Every recovery decision can be recorded with information including:

- Payment ID
- Payment amount
- Recommended action
- Recovery probability
- Expected net recovery
- Execution status
- Execution result
- Simulated recovery outcome
- Simulated recovery amount
- Simulated net recovery

This creates a decision trail that can be inspected after batch processing.

---

## 9. Dashboard

The Streamlit application provides:

- Individual payment analysis
- Batch payment analysis
- Recovery probability
- Recommended recovery action
- Expected recovery value
- Action-level comparison
- Recovery opportunity summaries
- Simulation results
- Recovery priority views
- Audit trail
- CSV export

The dashboard is designed to make the agent's reasoning visible rather than hiding the decision behind a single prediction.

---

## 10. Data Leakage Controls

The project includes a dedicated feature/data leakage audit.

The audit checks:

- Duplicate payment/action rows
- Target-like features
- Feature availability
- Within-payment consistency
- Train/test payment overlap
- Null and infinite values
- Feature-target associations
- Action-conditioned outcomes

The training and evaluation process uses payment-grouped splitting so the same payment cannot appear in both training and testing.

The dataset is synthetic. Historical customer/payment features are treated as information assumed to be available at the time of the recovery decision.

---

## 11. Project Structure

```text
razorpay_revenue_recovery/
│
├── app.py
├── agent.py
├── batch_processor.py
├── business_config.py
├── train.py
├── test_guardrails.py
├── test_action_interactions.py
├── README.md
├── requirements.txt
│
├── data/
│   └── recovery_training_data.csv
│
├── models/
│   └── recovery_model.joblib
│
└── research/
    └── results/