import streamlit as st
import pandas as pd

from agent import recommend
from batch_processor import (
    analyze_batch,
    execute_recovery_workflow
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Revenue Recovery Agent",
    page_icon="💰",
    layout="wide"
)


# =========================================================
# CONSTANTS
# =========================================================

REQUIRED_COLUMNS = [
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
    "failure_hour"
]

FAILURE_REASONS = [
    "bank_timeout",
    "insufficient_funds",
    "card_declined",
    "checkout_abandoned",
    "technical_error"
]

PAYMENT_METHODS = [
    "upi",
    "card",
    "netbanking",
    "wallet"
]


# =========================================================
# PAGE STYLE
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 20px;
    }

    .section {
        font-size: 24px;
        font-weight: 700;
        margin-top: 25px;
    }

    .prototype-banner {
        padding: 10px 14px;
        border-radius: 8px;
        background: #fff4d6;
        border: 1px solid #f0d98a;
        color: #5f4b12;
        margin: 8px 0 18px 0;
        font-size: 14px;
    }

    .decision-note {
        padding: 10px 14px;
        border-radius: 8px;
        background: #f5f7fa;
        border: 1px solid #e1e5ea;
        margin: 8px 0 16px 0;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">💰 AI Revenue Recovery Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Decision intelligence for failed-payment recovery —
    predict, prioritize, recommend and monitor recovery actions.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="prototype-banner">
        <b>🧪 PROTOTYPE / SYNTHETIC DATA</b>
        &nbsp; • &nbsp; Model estimates and simulated outcomes are for offline
        demonstration only. No real payment is executed.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HOW THE AGENT WORKS
# =========================================================

with st.expander("🤖 How the AI Recovery Agent Works"):

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("### 1️⃣")
        st.write("Analyze")
        st.caption(
            "Understand payment and customer context."
        )

    with col2:
        st.markdown("### 2️⃣")
        st.write("Predict")
        st.caption(
            "Estimate recovery probability."
        )

    with col3:
        st.markdown("### 3️⃣")
        st.write("Optimize")
        st.caption(
            "Choose the highest-value allowed action."
        )

    with col4:
        st.markdown("### 4️⃣")
        st.write("Prioritize")
        st.caption(
            "Rank the biggest recovery opportunities."
        )

    with col5:
        st.markdown("### 5️⃣")
        st.write("Monitor")
        st.caption(
            "Track simulated execution and outcomes."
        )


# =========================================================
# MODE SELECTION
# =========================================================

mode = st.radio(
    "Analysis Mode",
    [
        "🔎 Single Payment Analysis",
        "📊 Batch Recovery Analysis"
    ],
    horizontal=True
)


# =========================================================
# SINGLE PAYMENT ANALYSIS
# =========================================================

if mode == "🔎 Single Payment Analysis":

    st.sidebar.header(
        "💳 Failed Payment Details"
    )

    amount = st.sidebar.number_input(
        "Amount (₹)",
        min_value=100.0,
        max_value=250000.0,
        value=8500.0,
        step=100.0
    )

    reason = st.sidebar.selectbox(
        "Failure Reason",
        FAILURE_REASONS
    )

    method = st.sidebar.selectbox(
        "Payment Method",
        PAYMENT_METHODS
    )

    successful = st.sidebar.number_input(
        "Previous Successful Payments",
        min_value=0,
        max_value=100,
        value=12
    )

    failed = st.sidebar.number_input(
        "Previous Failed Payments",
        min_value=0,
        max_value=30,
        value=1
    )

    attempts = st.sidebar.number_input(
        "Previous Recovery Attempts",
        min_value=0,
        max_value=5,
        value=0
    )

    hours = st.sidebar.number_input(
        "Hours Since Failure",
        min_value=0.1,
        max_value=72.0,
        value=3.0
    )

    tenure = st.sidebar.number_input(
        "Customer Tenure (days)",
        min_value=1,
        max_value=3000,
        value=450
    )

    prior_recovery = st.sidebar.slider(
        "Prior Recovery Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.50
    )

    customer_value = st.sidebar.number_input(
        "Customer Value (₹)",
        min_value=1000.0,
        max_value=1000000.0,
        value=50000.0,
        step=1000.0
    )

    failure_hour = st.sidebar.slider(
        "Failure Hour",
        min_value=0,
        max_value=23,
        value=14
    )

    # -----------------------------------------------------
    # BUILD PAYMENT
    # -----------------------------------------------------

    payment = {
        "amount_inr": amount,
        "failure_reason": reason,
        "payment_method": method,
        "successful_payments": successful,
        "failed_payments": failed,
        "previous_attempts": attempts,
        "hours_since_failure": hours,
        "customer_tenure_days": tenure,
        "prior_recovery_rate": prior_recovery,
        "customer_value_inr": customer_value,
        "failure_hour": failure_hour
    }

    # -----------------------------------------------------
    # ANALYZE
    # -----------------------------------------------------

    if st.button(
        "🚀 Analyze Payment",
        use_container_width=True
    ):

        try:

            result = recommend(payment)

            st.divider()

            st.subheader(
                "🤖 AI Decision"
            )

            decision = result["decision"]

            if decision == "no_action":

                st.warning(
                    "⏸️ No Action Recommended"
                )

            elif decision == "escalate":

                st.error(
                    "🚨 Manual Review Required"
                )

            else:

                st.success(
                    "Recommended Action: "
                    f"**{decision.replace('_', ' ').title()}**"
                )

            # -------------------------------------------------
            # KPI
            # -------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Recovery Probability",
                    f"{result.get('confidence', 0) * 100:.2f}%"
                )

            with col2:
                selected_value = result.get("expected_value_inr")

                if selected_value is None:
                    selected_action = result.get("decision")
                    for candidate in result.get("candidates", []):
                        if candidate.get("action") == selected_action:
                            selected_value = candidate.get(
                                "expected_value_inr",
                                candidate.get("expected_net_recovery_inr", 0)
                            )
                            break

                selected_value = float(selected_value or 0)

                st.metric(
                    "Expected Net Recovery Value",
                    f"₹{selected_value:,.2f}"
                )

            with col3:
                st.metric(
                    "Payment Value",
                    f"₹{amount:,.2f}"
                )

            st.caption(
                "Expected Net Recovery Value is a model-based "
                "estimate after the configured action cost."
            )

            # -------------------------------------------------
            # ACTION MATRIX
            # -------------------------------------------------

            st.subheader(
                "📊 Action Decision Matrix"
            )

            candidates = pd.DataFrame(
                result["candidates"]
            )

            candidates["action"] = (
                candidates["action"]
                .str.replace("_", " ")
                .str.title()
            )

            candidates["probability"] = (
                candidates["probability"] * 100
            ).round(2)

            candidates["expected_value_inr"] = (
                candidates["expected_value_inr"]
                .clip(lower=0)
                .round(2)
            )

            candidates["allowed"] = (
                candidates["allowed"]
                .map({
                    True: "✅ Allowed",
                    False: "❌ Blocked"
                })
            )

            candidates = candidates[
                [
                    "action",
                    "probability",
                    "expected_value_inr",
                    "allowed",
                    "guardrail_reason"
                ]
            ]

            candidates.columns = [
                "Action",
                "Recovery Probability (%)",
                "Expected Net Value (₹)",
                "Status",
                "Guardrail"
            ]

            st.dataframe(
                candidates,
                use_container_width=True,
                hide_index=True
            )

            # -------------------------------------------------
            # EXPLANATION
            # -------------------------------------------------

            st.subheader(
                "💡 Why This Decision?"
            )

            if decision == "no_action":

                st.write(
                    "The predicted recovery probability is below "
                    "the configured threshold. The agent therefore "
                    "avoids unnecessary recovery effort."
                )

            elif decision == "escalate":

                st.write(
                    "Automated recovery is restricted for this "
                    "payment, so the agent routes it to manual review."
                )

            else:

                st.write(
                    f"The agent selected **"
                    f"{decision.replace('_', ' ').title()}** "
                    "because it provides the highest expected "
                    "net recovery value among the allowed actions."
                )

            st.info(
                "This is a decision-support prototype. "
                "No real financial transaction is executed."
            )

        except Exception as e:

            st.error(
                f"Unable to analyze payment: {e}"
            )


# =========================================================
# BATCH RECOVERY ANALYSIS
# =========================================================

else:

    st.header(
        "📊 Batch Recovery Control Center"
    )

    st.write(
        "Upload failed-payment data and the AI agent will "
        "analyze every unique payment, recommend a recovery "
        "action and create a prioritized recovery queue."
    )

    # -----------------------------------------------------
    # FILE UPLOAD
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "📁 Upload Failed Payments CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"Loaded {len(df):,} payment records."
            )

            # -------------------------------------------------
            # VALIDATION
            # -------------------------------------------------

            missing = [
                col
                for col in REQUIRED_COLUMNS
                if col not in df.columns
            ]

            if missing:

                st.error(
                    "The uploaded file is missing required columns:"
                )

                st.write(missing)

                st.stop()

            # -------------------------------------------------
            # DUPLICATES
            # -------------------------------------------------

            duplicates = (
                df["payment_id"]
                .duplicated()
                .sum()
            )

            if duplicates > 0:

                st.warning(
                    f"{duplicates:,} duplicate records found. "
                    "Only one record per payment ID will be analyzed."
                )

                df = (
                    df
                    .drop_duplicates(
                        subset="payment_id"
                    )
                    .reset_index(drop=True)
                )

            # -------------------------------------------------
            # PREVIEW
            # -------------------------------------------------

            with st.expander(
                "👁️ Preview Uploaded Data"
            ):

                st.dataframe(
                    df.head(20),
                    use_container_width=True,
                    hide_index=True
                )

            # -------------------------------------------------
            # ANALYZE BATCH
            # -------------------------------------------------

            if st.button(
                "🚀 Analyze All Failed Payments",
                use_container_width=True
            ):

                with st.spinner(
                    f"Analyzing {len(df):,} failed payments..."
                ):

                    try:

                        results = analyze_batch(
                            df
                        )

                        results, audit = (
                            execute_recovery_workflow(
                                results
                            )
                        )

                        st.session_state[
                            "batch_results"
                        ] = results

                        st.session_state[
                            "audit"
                        ] = audit

                        st.success(
                            f"Analysis completed for "
                            f"{len(results):,} unique payments."
                        )

                    except Exception as e:

                        st.error(
                            f"Batch analysis failed: {e}"
                        )

        except Exception as e:

            st.error(
                f"Unable to read uploaded CSV: {e}"
            )


# =========================================================
# BATCH RESULTS
# =========================================================

if (
    mode == "📊 Batch Recovery Analysis"
    and "batch_results" in st.session_state
):

    results = (
        st.session_state["batch_results"]
        .copy()
    )

    audit = st.session_state.get(
        "audit",
        pd.DataFrame()
    )

    if results.empty:

        st.warning(
            "No results were generated."
        )

        st.stop()

    st.divider()

    st.header(
        "📈 Recovery Operations Dashboard"
    )

    st.markdown(
        """
        <div class="decision-note">
            <b>How to read this dashboard:</b>
            <b>Expected Net Recovery</b> is the model-estimated opportunity after
            configured action costs. <b>Simulated Recovery</b> is an offline prototype
            outcome and is not confirmed revenue.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # DATA CLEANING
    # =====================================================

    results["amount_inr"] = pd.to_numeric(
        results["amount_inr"],
        errors="coerce"
    ).fillna(0)

    results["recovery_probability"] = pd.to_numeric(
        results["recovery_probability"],
        errors="coerce"
    ).fillna(0).clip(0, 1)

    results["expected_net_recovery_inr"] = pd.to_numeric(
        results["expected_net_recovery_inr"],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    if "simulated_recovery_amount_inr" not in results.columns:
        results["simulated_recovery_amount_inr"] = 0.0

    results["simulated_recovery_amount_inr"] = pd.to_numeric(
        results["simulated_recovery_amount_inr"],
        errors="coerce"
    ).fillna(0)

    if "simulated_net_recovery_inr" not in results.columns:
        results["simulated_net_recovery_inr"] = 0.0

    results["simulated_net_recovery_inr"] = pd.to_numeric(
        results["simulated_net_recovery_inr"],
        errors="coerce"
    ).fillna(0)

    if "simulated_recovered" not in results.columns:
        results["simulated_recovered"] = False

    # =====================================================
    # KPI CALCULATIONS
    # =====================================================

    total_payments = len(results)

    failed_value = (
        results["amount_inr"].sum()
    )

    expected_net_recovery = (
        results["expected_net_recovery_inr"].sum()
    )

    simulated_recovered_payments = (
        results["simulated_recovered"]
        .sum()
    )

    simulated_recovery_amount = (
        results["simulated_recovery_amount_inr"]
        .sum()
    )

    simulated_net_recovery = (
        results["simulated_net_recovery_inr"]
        .sum()
    )

    avg_probability = (
        results["recovery_probability"].mean()
        * 100
    )

    manual_review = int(
        results["recommended_action"]
        .astype(str)
        .str.lower()
        .eq("escalate")
        .sum()
    )

    if "execution_status" in results.columns:
        status = results["execution_status"].astype(str).str.lower()
        manual_review = max(
            manual_review,
            int(status.isin([
                "manual_review",
                "manual review",
                "review_required"
            ]).sum())
        )

    if "execution_result" in results.columns:
        execution_result = results["execution_result"].astype(str).str.lower()
        manual_review = max(
            manual_review,
            int(execution_result.str.contains(
                "manual|review", regex=True
            ).sum())
        )

    expected_opportunity_rate = (
        expected_net_recovery
        / failed_value
        * 100
        if failed_value > 0
        else 0
    )

    simulated_recovery_rate = (
        simulated_recovered_payments
        / total_payments
        * 100
        if total_payments > 0
        else 0
    )

    # =====================================================
    # KPI CARDS
    # =====================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Failed Payments",
            f"{total_payments:,}"
        )

    with c2:
        st.metric(
            "Failed Value",
            f"₹{failed_value:,.0f}"
        )

    with c3:
        st.metric(
            "Avg Recovery Probability",
            f"{avg_probability:.2f}%"
        )

    with c4:
        st.metric(
            "Expected Net Recovery",
            f"₹{expected_net_recovery:,.0f}"
        )

    with c5:
        st.metric(
            "Manual Review",
            f"{manual_review:,}"
        )

    # =====================================================
    # SIMULATION METRICS
    # =====================================================

    st.subheader(
        "🧪 Simulated Recovery Outcomes"
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric(
            "Simulated Recoveries",
            f"{simulated_recovered_payments:,}"
        )

    with s2:
        st.metric(
            "Simulated Recovery Rate",
            f"{simulated_recovery_rate:.2f}%"
        )

    with s3:
        st.metric(
            "Simulated Recovery Amount",
            f"₹{simulated_recovery_amount:,.0f}"
        )

    with s4:
        st.metric(
            "Simulated Net Recovery",
            f"₹{simulated_net_recovery:,.0f}"
        )

    st.caption(
        "Simulation only: these values do not represent "
        "real customer payments or confirmed revenue."
    )

    # =====================================================
    # EXPECTED OPPORTUNITY
    # =====================================================

    st.info(
        f"💡 The model estimates an expected net recovery "
        f"opportunity of **₹{expected_net_recovery:,.0f}**, "
        f"equivalent to **{expected_opportunity_rate:.2f}%** "
        f"of the failed transaction value."
    )

    # =====================================================
    # ACTION DISTRIBUTION
    # =====================================================

    st.subheader(
        "🎯 Recommended Action Distribution"
    )

    action_counts = (
        results["recommended_action"]
        .value_counts()
        .rename_axis("Action")
        .reset_index(
            name="Payments"
        )
    )

    action_counts["Action"] = (
        action_counts["Action"]
        .str.replace("_", " ")
        .str.title()
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        st.dataframe(
            action_counts,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        chart = (
            results["recommended_action"]
            .value_counts()
        )

        chart.index = (
            chart.index
            .str.replace("_", " ")
            .str.title()
        )

        st.bar_chart(
            chart
        )

    # =====================================================
    # EXPECTED VALUE BY ACTION
    # =====================================================

    st.subheader(
        "💰 Expected Net Recovery by Action"
    )

    recovery_by_action = (
        results
        .groupby(
            "recommended_action"
        )["expected_net_recovery_inr"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    recovery_by_action.index = (
        recovery_by_action.index
        .str.replace("_", " ")
        .str.title()
    )

    st.bar_chart(
        recovery_by_action
    )

    # =====================================================
    # PRIORITY QUEUE
    # =====================================================

    st.subheader(
        "🔥 Recovery Priority Queue"
    )

    results["recovery_priority_value"] = (
        results["expected_net_recovery_inr"]
    )

    priority = (
        results
        .sort_values(
            "recovery_priority_value",
            ascending=False
        )
        .head(100)
        .copy()
    )

    priority["Recommended Action"] = (
        priority["recommended_action"]
        .str.replace("_", " ")
        .str.title()
    )

    priority["Recovery Probability"] = (
        priority["recovery_probability"] * 100
    ).round(2).astype(str) + "%"

    priority["Amount"] = (
        priority["amount_inr"]
        .map(
            lambda x:
            f"₹{x:,.2f}"
        )
    )

    priority["Expected Net Recovery"] = (
        priority["expected_net_recovery_inr"]
        .map(
            lambda x:
            f"₹{x:,.2f}"
        )
    )

    priority["Priority Value"] = (
        priority["recovery_priority_value"]
        .round(2)
    )

    priority_display = priority[
        [
            "payment_id",
            "Amount",
            "failure_reason",
            "payment_method",
            "Recommended Action",
            "Recovery Probability",
            "Expected Net Recovery",
            "Priority Value"
        ]
    ]

    priority_display.columns = [
        "Payment ID",
        "Amount",
        "Failure Reason",
        "Payment Method",
        "Recommended Action",
        "Recovery Probability",
        "Expected Net Recovery",
        "Priority Value"
    ]

    st.dataframe(
        priority_display,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Priority Value ranks payments by expected "
        "net recovery opportunity."
    )

    # =====================================================
    # SIMULATED OUTCOME BY ACTION
    # =====================================================

    st.subheader(
        "🧪 Simulated Outcome by Action"
    )

    outcome = (
        results
        .groupby("recommended_action")
        .agg(
            Payments=(
                "payment_id",
                "count"
            ),
            Simulated_Recovered=(
                "simulated_recovered",
                "sum"
            ),
            Simulated_Recovery=(
                "simulated_recovery_amount_inr",
                "sum"
            ),
            Simulated_Net=(
                "simulated_net_recovery_inr",
                "sum"
            )
        )
        .reset_index()
    )

    outcome["recommended_action"] = (
        outcome["recommended_action"]
        .str.replace("_", " ")
        .str.title()
    )

    outcome.columns = [
        "Action",
        "Payments",
        "Simulated Recoveries",
        "Simulated Recovery (₹)",
        "Simulated Net Recovery (₹)"
    ]

    st.dataframe(
        outcome,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # AUDIT TRAIL
    # =====================================================

    st.subheader(
        "🧾 Recovery Audit Trail"
    )

    if not audit.empty:

        st.dataframe(
            audit.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Execution and recovery outcomes shown here "
            "are simulated for this prototype."
        )

    else:

        st.info(
            "No audit events were generated."
        )

    # =====================================================
    # FAILURE ANALYSIS
    # =====================================================

    st.subheader(
        "🔍 Failure Reason Analysis"
    )

    failure_analysis = (
        results
        .groupby("failure_reason")
        .agg(
            Payments=(
                "payment_id",
                "count"
            ),
            Failed_Value=(
                "amount_inr",
                "sum"
            ),
            Expected_Net_Recovery=(
                "expected_net_recovery_inr",
                "sum"
            ),
            Avg_Probability=(
                "recovery_probability",
                "mean"
            )
        )
        .reset_index()
    )

    failure_analysis[
        "Avg_Probability"
    ] = (
        failure_analysis[
            "Avg_Probability"
        ] * 100
    ).round(2)

    failure_analysis = (
        failure_analysis.round(2)
    )

    failure_analysis.columns = [
        "Failure Reason",
        "Payments",
        "Failed Value (₹)",
        "Expected Net Recovery (₹)",
        "Avg Recovery Probability (%)"
    ]

    st.dataframe(
        failure_analysis,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # PAYMENT METHOD ANALYSIS
    # =====================================================

    st.subheader(
        "💳 Payment Method Analysis"
    )

    method_analysis = (
        results
        .groupby("payment_method")
        .agg(
            Payments=(
                "payment_id",
                "count"
            ),
            Failed_Value=(
                "amount_inr",
                "sum"
            ),
            Expected_Net_Recovery=(
                "expected_net_recovery_inr",
                "sum"
            ),
            Avg_Probability=(
                "recovery_probability",
                "mean"
            )
        )
        .reset_index()
    )

    method_analysis[
        "Avg_Probability"
    ] = (
        method_analysis[
            "Avg_Probability"
        ] * 100
    ).round(2)

    method_analysis = (
        method_analysis.round(2)
    )

    method_analysis.columns = [
        "Payment Method",
        "Payments",
        "Failed Value (₹)",
        "Expected Net Recovery (₹)",
        "Avg Recovery Probability (%)"
    ]

    st.dataframe(
        method_analysis,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # HIGH VALUE OPPORTUNITIES
    # =====================================================

    st.subheader(
        "💎 High-Value Recovery Opportunities"
    )

    high_value = (
        results[
            results["amount_inr"] >= 50000
        ]
        .sort_values(
            "expected_net_recovery_inr",
            ascending=False
        )
        .head(50)
        .copy()
    )

    if high_value.empty:

        st.info(
            "No payments above ₹50,000 were found."
        )

    else:

        high_value[
            "Recommended Action"
        ] = (
            high_value[
                "recommended_action"
            ]
            .str.replace(
                "_",
                " "
            )
            .str.title()
        )

        high_value[
            "Recovery Probability (%)"
        ] = (
            high_value[
                "recovery_probability"
            ] * 100
        ).round(2)

        high_value[
            "Amount (₹)"
        ] = (
            high_value[
                "amount_inr"
            ]
            .map(
                lambda x:
                f"₹{x:,.2f}"
            )
        )

        high_value[
            "Expected Net Recovery (₹)"
        ] = (
            high_value[
                "expected_net_recovery_inr"
            ]
            .map(
                lambda x:
                f"₹{x:,.2f}"
            )
        )

        high_value_display = high_value[
            [
                "payment_id",
                "Amount (₹)",
                "failure_reason",
                "Recommended Action",
                "Recovery Probability (%)",
                "Expected Net Recovery (₹)"
            ]
        ]

        high_value_display.columns = [
            "Payment ID",
            "Amount (₹)",
            "Failure Reason",
            "Recommended Action",
            "Recovery Probability (%)",
            "Expected Net Recovery (₹)"
        ]

        st.dataframe(
            high_value_display,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # EXPORT
    # =====================================================

    st.subheader(
        "📥 Export Recovery Queue"
    )

    export_df = results.copy()

    export_df[
        "recommended_action"
    ] = (
        export_df[
            "recommended_action"
        ]
        .str.replace(
            "_",
            " "
        )
        .str.title()
    )

    export_df[
        "recovery_probability"
    ] = (
        export_df[
            "recovery_probability"
        ] * 100
    ).round(2)

    export_df[
        "expected_net_recovery_inr"
    ] = (
        export_df[
            "expected_net_recovery_inr"
        ].round(2)
    )

    export_df[
        "simulated_recovery_amount_inr"
    ] = (
        export_df[
            "simulated_recovery_amount_inr"
        ].round(2)
    )

    export_df[
        "simulated_net_recovery_inr"
    ] = (
        export_df[
            "simulated_net_recovery_inr"
        ].round(2)
    )

    csv = (
        export_df
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Recovery Recommendations",
        data=csv,
        file_name="recovery_recommendations.csv",
        mime="text/csv",
        use_container_width=True
    )

    # =====================================================
    # BUSINESS INTERPRETATION
    # =====================================================

    st.divider()

    st.subheader(
        "💡 Business Interpretation"
    )

    st.write(
        f"The AI agent analyzed **{total_payments:,} "
        f"unique failed payments** representing a "
        f"combined failed transaction value of "
        f"**₹{failed_value:,.2f}**."
    )

    st.write(
        f"The model estimates approximately "
        f"**₹{expected_net_recovery:,.2f}** in potential "
        f"net recovery value from actionable automated "
        f"recovery opportunities."
    )

    st.write(
        f"The offline simulation produced "
        f"**{simulated_recovered_payments:,} simulated "
        f"recoveries**, representing approximately "
        f"**₹{simulated_recovery_amount:,.2f}** in "
        f"simulated recovered transaction value."
    )

    st.write(
        f"**{manual_review:,} payments** were routed "
        f"toward manual review based on the configured "
        f"business guardrails."
    )

    st.info(
        "Important: Expected Net Recovery Value is a "
        "model estimate. Simulated Recovery is an offline "
        "prototype outcome. Neither represents confirmed "
        "real-world customer payments."
    )