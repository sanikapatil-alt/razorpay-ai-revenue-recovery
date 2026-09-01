# Razorpay Test Mode Execution

This is an optional final integration for the Revenue Recovery prototype.

## What it adds

The agent can remain the decision-maker while a separate bounded executor performs ONE safe real-world-style action in Razorpay Test Mode:

    AI decision
        ↓
    Guardrail
        ↓
    Payment Link action
        ↓
    Razorpay Test Mode
        ↓
    payment_link.paid webhook
        ↓
    Local audit trail

The executor refuses live keys.

## 1. Install dependency

From the project root:

```powershell
pip install requests
```

Also add this to `requirements.txt`:

```text
requests
```

## 2. Create Razorpay TEST credentials

Use Razorpay Dashboard Test Mode credentials only.

The executor requires:

```text
RAZORPAY_KEY_ID=rzp_test_...
RAZORPAY_KEY_SECRET=...
```

Do NOT put these values in source code or GitHub.

## 3. Start in dry-run mode

PowerShell:

```powershell
$env:RAZORPAY_KEY_ID="rzp_test_YOUR_KEY"
$env:RAZORPAY_KEY_SECRET="YOUR_SECRET"
$env:RAZORPAY_EXECUTION_ENABLED="false"
$env:RAZORPAY_MAX_TEST_LINKS="3"
```

Test:

```powershell
python -c "from testmode_recovery import create_test_payment_link; print(create_test_payment_link(payment_id='demo_001', amount_inr=100))"
```

Expected result:

```text
executed: False
status: dry_run
```

## 4. Enable bounded Test Mode execution

ONLY after the dry-run succeeds:

```powershell
$env:RAZORPAY_EXECUTION_ENABLED="true"
$env:RAZORPAY_MAX_TEST_LINKS="3"
```

Then:

```powershell
python -c "from testmode_recovery import create_test_payment_link; print(create_test_payment_link(payment_id='demo_001', amount_inr=100))"
```

The response should contain a Test Mode payment link URL.

Razorpay's documentation currently states that Test Mode Payment Links can be created and tested, with a test-mode limit of 30 Payment Links per business. We deliberately set a much smaller application-level limit for this prototype. See the official Payment Links API documentation.

## 5. Test the payment

Open the returned payment link in a browser.

Use Razorpay Test Mode to complete a success or failure flow.

## 6. Webhook receiver

Start:

```powershell
$env:RAZORPAY_WEBHOOK_SECRET="YOUR_WEBHOOK_SECRET"
python testmode_webhook.py
```

For a local demo, expose the local port using a trusted tunnelling solution and configure the resulting HTTPS URL in Razorpay Dashboard → Account & Settings → Webhooks.

Subscribe to:

```text
payment_link.paid
payment_link.partially_paid
payment_link.cancelled
payment_link.expired
```

The receiver validates `X-Razorpay-Signature` using the raw request body before recording the event.

## 7. What to show in the pitch

Show ONE bounded flow:

    Failed payment
       ↓
    AI recommends Payment Link
       ↓
    Guardrail permits it
       ↓
    Test Mode Payment Link created
       ↓
    Test payment succeeds
       ↓
    payment_link.paid webhook received
       ↓
    Audit record changes from "created" to "paid"

Do not claim this is production revenue recovery. It is a Test Mode demonstration.

## 8. Important safety controls

The executor:

- rejects non-`rzp_test_` keys
- defaults to dry-run
- supports only the Payment Link action
- enforces a small test-link limit
- writes an audit record for each action
- validates webhook signatures
- never uses live financial execution

Razorpay's Payment Link API uses `POST /v1/payment_links`, and Payment Link webhooks include `payment_link.paid` and other lifecycle events.
