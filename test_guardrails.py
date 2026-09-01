from agent import guardrail


tests = [
    {
        "name": "Retry limit",
        "payment": {
            "previous_attempts": 3,
            "hours_since_failure": 2,
            "amount_inr": 5000
        },
        "action": "retry"
    },
    {
        "name": "Old payment",
        "payment": {
            "previous_attempts": 0,
            "hours_since_failure": 50,
            "amount_inr": 5000
        },
        "action": "retry"
    },
    {
        "name": "High-value retry",
        "payment": {
            "previous_attempts": 0,
            "hours_since_failure": 2,
            "amount_inr": 75000
        },
        "action": "retry"
    },
    {
        "name": "Normal payment",
        "payment": {
            "previous_attempts": 0,
            "hours_since_failure": 2,
            "amount_inr": 5000
        },
        "action": "retry"
    }
]


print("\nGUARDRAIL TEST")
print("=" * 40)

for test in tests:

    allowed, reason, code = guardrail(
        test["payment"],
        test["action"],
    )

    print(f"\n{test['name']}")
    print("Allowed:", allowed)
    print("Reason:", reason)
