# Clean Business Assumptions & Evaluation

## Why this layer exists

The project previously mixed several different definitions of action cost and,
in some execution paths, generated recovery outcomes from the model's own
probability. That makes those outputs unsuitable as measured business impact.

This layer separates:

1. **Prediction** — the ML model estimates recovery probability.
2. **Decision** — the policy selects an action using expected net value and guardrails.
3. **Measurement** — evaluation reads the observed historical outcome for the selected action.

## Canonical policy

- Actions: `payment_link`, `reminder`, `retry`, `escalate`, `no_action`
- Maximum retries during execution: 2
- Low-probability stopping threshold: 0.25
- Retry blocked after 3 previous attempts
- Retry/reminder blocked after 48 hours
- Retry blocked for payments >= ₹50,000
- Failed retry escalates
- Failed payment link moves to reminder
- Failed reminder escalates
- Escalation means manual review, not automated recovery

## Canonical prototype economics

These are **assumptions**, not verified Razorpay operating costs:

| Action | Estimated cost |
|---|---:|
| Payment Link | 1.0% of amount |
| Reminder | 0.5% of amount |
| Retry | 1.5% of amount |
| Escalate | 5.0% of amount |
| No Action | 0% |

Replace these with real merchant-specific economics before claiming actual
financial impact.

## What counts as measured recovery

For a held-out payment:

- The model chooses an action.
- The evaluator finds that action's historical row for the same payment.
- `recovered == 1` means the historical outcome recovered the payment.
- Recovered amount is the payment amount when recovered, otherwise ₹0.
- Net recovery = observed recovered amount - estimated action cost.

The evaluator does **not** generate a recovery event from the predicted probability.

## No-action baseline

The dataset contains action rows for the intervention choices but does not
provide an observed no-action counterfactual. Therefore no-action is reported
only as a zero-recovery accounting baseline. It must not be described as the
actual revenue that would have occurred without intervention.

## Causal limitation

Because action assignment in historical data was not established as randomized,
offline policy evaluation can compare observed outcomes, but it should not be
called a causal uplift estimate.

A future experiment with randomized action assignment would be required for a
strong causal claim.

## Files

- `business_config.py` — single source of truth for policy/economics.
- `evaluate_policy_clean.py` — clean held-out policy evaluation.
- `clean_policy_results.csv` — generated when evaluation runs.
- `clean_fixed_action_baselines.csv` — generated when evaluation runs.
