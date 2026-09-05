# Razorpay Revenue Recovery Agent

An AI-assisted revenue recovery prototype for failed payments, built for the **Razorpay AI Buildathon – Track 3: AI Revenue Recovery**.

The system analyzes failed payments, estimates recovery probability for candidate recovery actions, selects the best allowed action using expected net recovery, applies deterministic business guardrails, prioritizes recovery opportunities, simulates recovery outcomes, supports bounded Razorpay Test Mode execution, and records an audit trail.

> **Prototype / Synthetic Data**
>
> This is a buildathon prototype using synthetic data and configurable business assumptions. Evaluation results shown below are offline results on synthetic data and do not represent real Razorpay customer revenue.

---

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

### Decision Flow

```text
Failed Payment
      ↓
Customer / Payment Features
      ↓
Recovery Probability Model
      ↓
Evaluate Candidate Actions
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
Dashboard / Outcome Tracking
```

The same payment is evaluated under multiple candidate actions so the agent can choose the most economically useful allowed intervention.

---

## 3. Recovery Actions

The agent evaluates four recovery actions:

| Action | Purpose |
|---|---|
| `retry` | Attempt the payment again |
| `payment_link` | Generate a payment link for the customer |
| `reminder` | Send a payment reminder |
| `escalate` | Route the case for manual intervention |

The action is **not selected from probability alone**.

For automated actions:

```text
Expected Net Recovery
= Recovery Probability × Payment Amount
  − Estimated Action Cost
```

The action with the highest allowed expected net recovery is selected.

Escalation is handled as a policy outcome rather than being treated as a normal automated revenue action.

---

## 4. Machine Learning Model

The current model is an **Extra Trees classifier**.

### Configuration

- 300 trees
- Maximum depth: 12
- Minimum samples per leaf: 5
- Balanced class weights
- Fixed random seed for reproducibility

The model predicts:

```text
P(recovery | payment context, action)
```

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

The trained model is stored at:

```text
models/recovery_model.joblib
```

---

## 5. Business Guardrails

Machine learning does not have unrestricted control over recovery actions.

The policy layer applies deterministic guardrails.

### Retry Limit

Retry is blocked when the configured maximum retry threshold has been reached.

### Recovery Window

Retry and reminder are blocked when the payment is outside the configured recovery window.

### High-Value Payments

High-value payments cannot automatically be retried and can be routed for manual review.

### Low-Confidence Decisions

If no automated action reaches the minimum recovery probability threshold, the system can escalate instead of forcing an automated action.

### Current Prototype Policy

| Rule | Configuration |
|---|---:|
| Maximum retry attempts | 3 |
| Recovery window | 48 hours |
| High-value retry threshold | ₹50,000 |
| Low-probability threshold | 0.25 |

### Prototype Action Costs

| Action | Cost Rate |
|---|---:|
| `payment_link` | 0.25% |
| `reminder` | 0.10% |
| `retry` | 0.50% |
| `escalate` | 1.00% |

These business assumptions are centralized in:

```text
business_config.py
```

This keeps policy separate from the ML model.

---

## 6. Expected Recovery vs Actual Recovery

The project deliberately distinguishes between **expected net recovery**, **observed held-out outcomes**, and **simulated recovery**.

### Expected Net Recovery

A model-side estimate used to rank recovery actions.

It is calculated from:

- predicted recovery probability
- payment value
- prototype action costs

It is **not measured revenue**.

### Held-Out Observed Outcome

The final offline policy evaluation uses a completely held-out synthetic test set where the observed `recovered` outcome is available for the tested payment/action.

This makes it possible to compare the AI policy against baseline policies.

### Simulated Recovery

The Streamlit application includes an offline simulation layer for demonstrating the workflow on unseen batches.

Simulation uses the model's predicted recovery probability to generate reproducible hypothetical outcomes.

Simulated recovery is **not real customer recovery** and should not be interpreted as actual Razorpay revenue.

---

## 7. Evaluation

The project uses payment-grouped evaluation so the same payment is never placed in both training and test sets.

### Dataset Split

```text
Total rows:            14,000
Unique payments:        3,500
Training payments:      2,800
Held-out payments:        700
Payment overlap:            0
```

### Model Validation

The regularized Extra Trees model was selected during model-development validation and evaluated as part of the final evaluation workflow.

| Metric | Result |
|---|---:|
| Accuracy | 63.64% |
| Precision | 55.06% |
| Recall | 56.99% |
| F1 | 56.01% |
| ROC-AUC | 67.56% |
| PR-AUC | 57.28% |
| Brier Score | 0.2262 |

These metrics describe model performance on the synthetic evaluation setup; they are not production performance guarantees.

### Business Policy Test

On a completely held-out set of 700 payments, the policy was evaluated after training on 2,800 other payments with zero payment-ID overlap.

| Policy | Recovery Rate | Net Recovery |
|---|---:|---:|
| AI expected-net-recovery | 57.14% | ₹2,099,003 |
| Highest probability | 56.71% | ₹2,091,482 |
| Historical best action | 45.00% | ₹1,670,031 |
| Random allowed action | 43.14% | ₹1,580,286 |
| Oracle upper bound | 86.29% | ₹3,124,572 |

Against the random-allowed baseline, the AI policy produced approximately:

```text
Net recovery lift:    ₹518,716
Relative improvement: 32.82%
```

A paired bootstrap comparison of AI expected-net-recovery versus the random-allowed policy produced:

```text
95% CI: ₹418.25 to ₹1,062.26 per payment
```

### Important Interpretation

The direct comparison between expected-net optimization and highest-probability selection was **not statistically decisive**.

The paired bootstrap interval was approximately:

```text
95% CI: -₹29.22 to ₹49.02 per payment
```

Therefore, the project does **not** claim that expected-net optimization is statistically superior to probability ranking on this synthetic test set.

This is intentional: the evaluation is reported without overstating the result.

> **Important:** The dataset is synthetic and the recovery outcomes are used for offline evaluation. These numbers do not represent confirmed Razorpay customer payments or real recovered revenue.

---

## 8. Held-Out Action Distribution

On the held-out synthetic test set, the AI policy selected:

| Action | Selections |
|---|---:|
| `payment_link` | 293 |
| `retry` | 238 |
| `reminder` | 165 |
| `escalate` | 4 |

The policy identified:

```text
Payments with at least one blocked retry candidate: 31
AI-selected escalations: 4
```

This demonstrates that the policy considers business constraints instead of blindly choosing the model's highest-probability action.

---

## 9. Independent Batch Test

The Streamlit application was also tested on a separate synthetic batch containing:

```text
500 unique failed payments
```

This batch was independent of the training dataset and was used to verify that the full application pipeline works on unseen input.

The batch workflow successfully produced:

- recovery recommendations
- recovery priority ranking
- expected recovery values
- action distribution
- simulated outcomes
- audit records
- failure-reason analysis
- payment-method analysis
- high-value payment analysis
- downloadable results

Example independent batch run:

```text
Failed payments:          500
Failed value:             ~₹18.66 lakh
Expected net recovery:    ~₹72.87 lakh
Simulated net recovery:   ~₹72.05 lakh
```

These are **application demonstration outputs**, not observed real-world revenue.

---

## 10. Auditability

Every recovery decision can be recorded with information including:

- Payment ID
- Payment amount
- Recommended action
- Recovery probability
- Expected net recovery
- Guardrail decision
- Execution status
- Execution result
- Simulated recovery outcome
- Simulated recovery amount
- Simulated net recovery
- Timestamp

This creates a decision trail that can be inspected after batch processing and used for future outcome analysis.

---

## 11. Dashboard

The Streamlit application provides:

- Individual payment analysis
- Batch payment analysis
- Recovery probability
- Recommended recovery action
- Expected recovery value
- Action-level comparison
- Recovery opportunity summaries
- Recovery priority views
- Simulation results
- Audit trail
- CSV export

Run the dashboard with:

```bash
python -m streamlit run app.py
```

The dashboard is designed to make the agent's reasoning visible rather than hiding the decision behind a single prediction.

---

## 12. Data Leakage Controls

The project includes a dedicated feature and data leakage audit.

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

The final split verifies:

```text
Training payments:    2,800
Test payments:          700
Payment overlap:          0
```

The dataset is synthetic. Historical customer/payment features are treated as information assumed to be available at the time of the recovery decision.

For a production deployment, feature availability and timing would need to be validated against real payment and customer systems.

---

## 13. Razorpay Test Mode Integration

The repository includes a bounded Razorpay Test Mode integration to demonstrate how a selected recovery action can move toward workflow execution.

Files:

```text
testmode_recovery.py
testmode_webhook.py
RAZORPAY_TESTMODE_SETUP.md
```

The integration is intentionally restricted to Test Mode.

### Safety Controls

- Test keys only
- Live-key rejection
- Dry-run support
- Execution disabled by default
- Maximum test execution limit
- Bounded recovery actions
- Audit logging

Example environment configuration:

```powershell
$env:RAZORPAY_KEY_ID="rzp_test_..."
$env:RAZORPAY_KEY_SECRET="..."
$env:RAZORPAY_EXECUTION_ENABLED="true"
$env:RAZORPAY_MAX_TEST_LINKS="1"
```

Never commit API credentials to GitHub.

See `RAZORPAY_TESTMODE_SETUP.md` for the detailed Test Mode setup.

---

## 14. Webhook Handling

The project includes a local webhook receiver for Razorpay Test Mode payment-link events.

Supported events include:

```text
payment_link.paid
payment_link.partially_paid
payment_link.cancelled
payment_link.expired
```

Webhook signatures are verified using the raw request body and HMAC SHA-256 before the event is accepted.

The intended workflow is:

```text
AI Recovery Decision
        ↓
Bounded Test Mode Action
        ↓
Payment Link
        ↓
Customer Payment
        ↓
Razorpay Webhook
        ↓
Outcome Tracking
        ↓
Audit / Analytics
        ↓
Future Model Improvement
```

This provides the foundation for measuring whether a recovery intervention actually succeeded.

---

## 15. Project Structure

```text
razorpay_revenue_recovery/
│
├── app.py
├── agent.py
├── batch_processor.py
├── business_config.py
├── train.py
│
├── test_guardrails.py
├── test_action_interactions.py
│
├── testmode_recovery.py
├── testmode_webhook.py
├── RAZORPAY_TESTMODE_SETUP.md
│
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── data/
│   └── recovery_training_data.csv
│
├── models/
│   └── recovery_model.joblib
│
└── research/
    ├── EVALUATION_NOTES.md
    ├── final_heldout_evaluation.py
    └── results/
        ├── final_policy_results.csv
        ├── final_policy_summary.json
        ├── final_policy_summary.txt
        └── final_policy_bootstrap.json
```

---

## 16. Running the Project

### Clone the repository

```bash
git clone https://github.com/sanikapatil-alt/razorpay-ai-revenue-recovery.git
cd razorpay-ai-revenue-recovery
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

This generates:

```text
models/recovery_model.joblib
```

### Run the dashboard

```bash
python -m streamlit run app.py
```

### Run tests

```bash
python test_guardrails.py
python test_action_interactions.py
```

---

## 17. Prototype Business Assumptions

This buildathon prototype uses synthetic data and configurable business parameters.

Examples include:

- Recovery action costs
- Maximum retry attempts
- Recovery window
- High-value threshold
- Customer history
- Payment failure patterns
- Simulated recovery outcomes

These assumptions are isolated in the project so they can be replaced with production business logic and validated telemetry later.

---

## 18. Production Roadmap

A production system would replace prototype assumptions with validated operational data and infrastructure.

### Real Payment Data

Connect the agent to real payment and transaction telemetry.

### Real Customer History

Use verified customer payment behavior and historical recovery performance.

### Live Outcome Tracking

Use payment webhooks and transaction status to determine whether each intervention succeeded.

### Model Calibration and Learning

Use confirmed recovery outcomes to:

- recalibrate recovery probabilities
- evaluate action-level performance
- retrain models
- detect model drift

### Business Optimization

Tune:

- action costs
- retry limits
- recovery windows
- customer-value thresholds
- escalation thresholds

using real business economics.

### Monitoring

Track:

- recovery rate
- recovered revenue
- net recovered revenue
- intervention cost
- escalation rate
- guardrail blocks
- action-level recovery
- model calibration

### Controlled Production Execution

Before live execution, introduce:

- approval policies
- rate limiting
- idempotency
- strong authentication
- access controls
- execution monitoring
- failure handling
- rollback/recovery procedures

---

## 19. Limitations

This is a buildathon prototype, not a production revenue recovery system.

Important limitations:

1. Training and evaluation data are synthetic.
2. Historical customer features are assumed to be available before the recovery decision.
3. Offline policy performance does not guarantee real-world business uplift.
4. The oracle result is an upper-bound benchmark and is not a deployable policy.
5. Simulated recovery outcomes are not real payment outcomes.
6. Action costs are prototype assumptions.
7. Test Mode execution demonstrates bounded integration rather than live production recovery.
8. More real-world data would be required for robust model calibration.
9. Production deployment would require stronger execution, monitoring, security, and reliability controls.

---

## 20. Why This Meets the Track Objective

The project is designed around the core revenue recovery workflow:

```text
Detect Revenue at Risk
        ↓
Estimate Recovery Probability
        ↓
Choose the Right Intervention
        ↓
Optimize Expected Economic Value
        ↓
Apply Business Guardrails
        ↓
Execute a Bounded Workflow
        ↓
Track Outcomes
        ↓
Maintain an Audit Trail
```

The system therefore goes beyond simply predicting whether a payment will recover.

It demonstrates an AI decision agent combining:

- machine learning
- action selection
- economic optimization
- business policy
- guardrails
- bounded workflow execution
- outcome tracking
- auditability

The central objective is to make a reasoned recovery decision for each failed payment rather than applying the same recovery strategy to every payment.

---

## 21. Project Objective

The goal of the AI Revenue Recovery Agent is to maximize recovered revenue from failed payments while minimizing unnecessary interventions and respecting business constraints.

The core decision is:

> **For every failed payment, what should the system do next — and why?**

This project provides a complete prototype architecture for answering that question.

---

## Safety

This repository is designed for demonstration and Test Mode experimentation.

**Do not use production Razorpay credentials with the prototype execution scripts.**

Never commit API secrets, `.env` files, or other sensitive credentials to GitHub.
