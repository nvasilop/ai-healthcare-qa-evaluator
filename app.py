"""
AI Healthcare QA Evaluator
==========================

A simple Streamlit dashboard that demonstrates how a QA Analyst can test
AI healthcare agents using synthetic patient scenarios (test cases).

This app only *displays and analyses* test cases. It does NOT call any AI model
yet -- the goal is to show the QA testing workflow and metrics.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Our simple rule-based evaluator (no AI / ML inside).
from evaluator import evaluate_all, evaluate_response_set

# Paths to the JSON files. Using Path(__file__) makes the app work no matter
# where it is launched from.
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "test_cases.json"
RESPONSES_FILE = DATA_DIR / "agent_responses_v1.json"
RESPONSES_V2_FILE = DATA_DIR / "agent_responses_v2.json"
PROMPTS_FILE = DATA_DIR / "prompt_versions.json"
FAILURE_ANALYSIS_FILE = DATA_DIR / "failure_analysis.json"
THRESHOLDS_FILE = DATA_DIR / "quality_thresholds.json"
MONITORING_FILE = DATA_DIR / "monitoring_metrics.csv"
COMPLIANCE_FILE = DATA_DIR / "compliance_checklist.json"


@st.cache_data  # Cache the result so the file is only read once per session.
def load_json(path: Path) -> list[dict]:
    """Load a JSON file and return it as a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data  # Cache the result so the CSV is only read once per session.
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


@st.cache_data  # Cache the result so the file is only read once per session.
def load_test_cases(path: Path) -> pd.DataFrame:
    """Load the test cases from the JSON file into a pandas DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def main() -> None:
    # --- Page configuration ---
    st.set_page_config(page_title="AI Healthcare QA Evaluator", page_icon="🩺", layout="wide")

    st.title("🩺 AI Healthcare QA Evaluator")
    st.write(
        "A demonstration dashboard for QA Analysts to test AI healthcare agents "
        "using synthetic patient scenarios."
    )

    # --- Load data ---
    df = load_test_cases(DATA_FILE)

    # --- Key metrics ---
    # Count the totals we want to highlight at the top of the dashboard.
    total_cases = len(df)
    high_risk_cases = (df["risk_level"] == "High").sum()
    greek_cases = (df["language"] == "Greek").sum()

    # Show the metrics side by side in three columns.
    col1, col2, col3 = st.columns(3)
    col1.metric("Total test cases", total_cases)
    col2.metric("High-risk cases", int(high_risk_cases))
    col3.metric("Greek-language cases", int(greek_cases))

    st.divider()

    # --- Test cases table ---
    st.subheader("📋 Test cases")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Bar chart by category ---
    st.subheader("📊 Test cases by category")
    # Count how many test cases fall into each category, then plot them.
    category_counts = df["category"].value_counts()
    st.bar_chart(category_counts)

    st.divider()

    # --- QA Evaluation Results ---
    st.subheader("✅ QA Evaluation Results")
    st.write(
        "Here we run each sample AI agent response through the rule-based evaluator "
        "and check whether it satisfies the required checks for its test case."
    )

    # Load the raw test cases (as a list of dicts) and the agent responses,
    # then run the evaluator on them.
    test_cases_raw = load_json(DATA_FILE)
    agent_responses = load_json(RESPONSES_FILE)
    results = evaluate_all(test_cases_raw, agent_responses)
    results_df = pd.DataFrame(results)

    # Count how many tests passed and failed, and compute the pass rate.
    total_tests = len(results_df)
    passed_tests = (results_df["status"] == "PASS").sum()
    failed_tests = (results_df["status"] == "FAIL").sum()
    pass_rate = (passed_tests / total_tests * 100) if total_tests else 0

    # Show the QA summary metrics side by side.
    m1, m2, m3 = st.columns(3)
    m1.metric("Passed tests", int(passed_tests))
    m2.metric("Failed tests", int(failed_tests))
    m3.metric("Pass rate", f"{pass_rate:.0f}%")

    # Show the full results table. We join the list columns into readable strings.
    st.markdown("**All results**")
    display_df = results_df.copy()
    display_df["passed_checks"] = display_df["passed_checks"].apply(", ".join)
    display_df["failed_checks"] = display_df["failed_checks"].apply(", ".join)
    st.dataframe(
        display_df[
            ["test_case_id", "category", "risk_level", "status", "passed_checks", "failed_checks"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # --- Failed cases (shown separately for closer inspection) ---
    st.markdown("**❌ Failed cases**")
    failed_results = [r for r in results if r["status"] == "FAIL"]

    if not failed_results:
        st.success("No failed cases. All sample responses passed their checks!")
    else:
        # Build a quick lookup of test cases and responses by id so we can show
        # the user message, expected behaviour and agent response for each failure.
        test_cases_by_id = {tc["id"]: tc for tc in test_cases_raw}
        responses_by_id = {r["test_case_id"]: r["agent_response"] for r in agent_responses}

        for result in failed_results:
            tc_id = result["test_case_id"]
            test_case = test_cases_by_id[tc_id]
            # Each failed case can be expanded to see the details.
            with st.expander(f"{tc_id} — {result['category']} ({result['risk_level']} risk)"):
                st.markdown(f"**User message:** {test_case['user_message']}")
                st.markdown(f"**Expected behavior:** {test_case['expected_behavior']}")
                st.markdown(f"**Agent response:** {responses_by_id.get(tc_id, '(no response)')}")
                st.markdown(f"**Failed checks:** {', '.join(result['failed_checks'])}")

    st.divider()

    # --- Failure Analysis & Prompt Improvement ---
    st.subheader("🔧 Failure Analysis & Prompt Improvement")
    st.write(
        "When tests fail, a QA Analyst investigates the root cause and proposes a "
        "safer prompt. Below are the two prompt versions and the analysis of each failure."
    )

    # Load the two prompt versions and show each one in an expandable panel.
    prompts = load_json(PROMPTS_FILE)
    baseline = prompts["v1_baseline_prompt"]
    improved = prompts["v2_improved_prompt"]

    with st.expander(f"📄 {baseline['name']}"):
        st.caption(baseline["description"])
        st.code(baseline["prompt"], language="text")

    with st.expander(f"📄 {improved['name']}"):
        st.caption(improved["description"])
        st.code(improved["prompt"], language="text")

    # Load the failure analysis entries and show them as a summary table.
    failure_analysis = load_json(FAILURE_ANALYSIS_FILE)
    failure_df = pd.DataFrame(failure_analysis)

    st.markdown("**Failure analysis summary**")
    st.dataframe(
        failure_df[["test_case_id", "failure_summary", "business_or_safety_impact"]],
        use_container_width=True,
        hide_index=True,
    )

    # Show the full details of each failure in an expandable panel.
    st.markdown("**Detailed failure analysis**")
    for entry in failure_analysis:
        with st.expander(f"{entry['test_case_id']} — failure analysis"):
            st.markdown(f"**Failure summary:** {entry['failure_summary']}")
            st.markdown(f"**Possible root cause:** {entry['possible_root_cause']}")
            st.markdown(f"**Suggested prompt fix:** {entry['suggested_prompt_fix']}")
            st.markdown(f"**Retest plan:** {entry['retest_plan']}")
            st.markdown(f"**Business / safety impact:** {entry['business_or_safety_impact']}")

    st.divider()

    # --- QA Workflow ---
    st.subheader("🔁 QA Workflow")
    st.write("The end-to-end process a QA Analyst follows for a healthcare AI agent:")
    st.markdown(
        """
        1. **Define protocol** — agree on the safety rules the agent must follow.
        2. **Create test cases** — write synthetic patient scenarios with expected behaviour.
        3. **Run agent responses** — collect the AI agent's answers to each test case.
        4. **Evaluate pass/fail** — check each response against its required checks.
        5. **Analyze failures** — find the root cause of every failed test.
        6. **Improve prompt logic** — update the prompt to fix the safety gaps.
        7. **Re-test** — run the test suite again to confirm the fixes work.
        8. **Monitor after deployment** — keep watching real-world behaviour over time.
        """
    )

    st.divider()

    # --- Before / After Validation ---
    st.subheader("🆚 Before / After Validation")
    st.write(
        "This section shows how QA can validate whether a prompt improvement actually "
        "improved AI agent behavior."
    )

    # Load both response sets and evaluate them against the SAME test cases.
    # This is the core idea: only the responses change between v1 and v2.
    responses_v1 = load_json(RESPONSES_FILE)
    responses_v2 = load_json(RESPONSES_V2_FILE)
    results_v1 = evaluate_response_set(test_cases_raw, responses_v1)
    results_v2 = evaluate_response_set(test_cases_raw, responses_v2)

    def summarize(results: list[dict]) -> dict:
        """Compute total / passed / failed / pass rate for one result set."""
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = total - passed
        rate = (passed / total * 100) if total else 0
        return {"total": total, "passed": passed, "failed": failed, "pass_rate": rate}

    summary_v1 = summarize(results_v1)
    summary_v2 = summarize(results_v2)

    # Build a small comparison table between Agent v1 and Agent v2.
    comparison_df = pd.DataFrame(
        {
            "Metric": ["Total tests", "Passed tests", "Failed tests", "Pass rate (%)"],
            "Agent v1": [
                summary_v1["total"],
                summary_v1["passed"],
                summary_v1["failed"],
                round(summary_v1["pass_rate"]),
            ],
            "Agent v2": [
                summary_v2["total"],
                summary_v2["passed"],
                summary_v2["failed"],
                round(summary_v2["pass_rate"]),
            ],
        }
    )
    st.markdown("**Comparison table**")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Simple bar chart comparing the pass rate of the two versions.
    st.markdown("**Pass rate comparison**")
    pass_rate_chart = pd.DataFrame(
        {"Pass rate (%)": [summary_v1["pass_rate"], summary_v2["pass_rate"]]},
        index=["Agent v1", "Agent v2"],
    )
    st.bar_chart(pass_rate_chart)

    # Compare each test case between v1 and v2 by looking up status by id.
    status_v1 = {r["test_case_id"]: r["status"] for r in results_v1}
    status_v2 = {r["test_case_id"]: r["status"] for r in results_v2}

    # Test cases that went from FAIL (v1) to PASS (v2) -> the improvements.
    improved = [tc_id for tc_id in status_v1 if status_v1[tc_id] == "FAIL" and status_v2.get(tc_id) == "PASS"]
    # Test cases that still fail in v2 -> remaining work.
    still_failing = [tc_id for tc_id in status_v2 if status_v2[tc_id] == "FAIL"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**✅ Improved (FAIL → PASS)**")
        if improved:
            for tc_id in improved:
                st.write(f"- {tc_id}")
        else:
            st.write("None")
    with col_b:
        st.markdown("**❌ Still failing in v2**")
        if still_failing:
            for tc_id in still_failing:
                st.write(f"- {tc_id}")
        else:
            st.write("None — all tests pass in v2!")

    st.divider()

    # --- Quality Standards & Monitoring ---
    st.subheader("📈 Quality Standards & Monitoring")
    st.write(
        "Quality thresholds define the minimum acceptable behaviour for the agent. "
        "Monitoring tracks these metrics over time so we can catch regressions after deployment."
    )

    # Load the quality thresholds and show them in a table.
    thresholds = load_json(THRESHOLDS_FILE)
    thresholds_df = pd.DataFrame(thresholds)
    st.markdown("**Quality thresholds (minimum standards)**")
    st.dataframe(thresholds_df, use_container_width=True, hide_index=True)

    # Load the synthetic monitoring metrics (7 days) from the CSV file.
    monitoring_df = load_csv(MONITORING_FILE)
    st.markdown("**Monitoring metrics (last 7 days)**")
    st.dataframe(monitoring_df, use_container_width=True, hide_index=True)

    # Show line charts over time for three key metrics. We set the date as the
    # index so Streamlit draws the date on the x-axis.
    monitoring_indexed = monitoring_df.set_index("date")
    st.markdown("**Overall pass rate over time**")
    st.line_chart(monitoring_indexed["overall_pass_rate"])
    st.markdown("**Red-flag escalation rate over time**")
    st.line_chart(monitoring_indexed["red_flag_escalation_rate"])
    st.markdown("**Privacy protection rate over time**")
    st.line_chart(monitoring_indexed["privacy_protection_rate"])

    # Compare the LATEST monitoring values against each threshold.
    st.markdown("**Latest values vs thresholds**")
    latest = monitoring_df.iloc[-1]  # The most recent day (last row).
    comparison_rows = []
    for threshold in thresholds:
        metric = threshold["metric_name"]
        required = threshold["threshold_percentage"]
        # Some thresholds may not have a matching monitoring column; skip those.
        if metric not in monitoring_df.columns:
            continue
        latest_value = latest[metric]
        # A metric meets the threshold if it is at or above the required percentage.
        meets = latest_value >= required
        comparison_rows.append(
            {
                "metric_name": metric,
                "latest_value": latest_value,
                "threshold_percentage": required,
                "result": "Meets threshold" if meets else "Below threshold",
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    st.divider()

    # --- Compliance & Documentation Checklist ---
    st.subheader("📋 Compliance & Documentation Checklist")
    st.write(
        "A QA readiness checklist that tracks safety, privacy, validation, and "
        "documentation work across the project."
    )

    # Load the compliance checklist and show it as a table.
    compliance = load_json(COMPLIANCE_FILE)
    compliance_df = pd.DataFrame(compliance)
    st.dataframe(compliance_df, use_container_width=True, hide_index=True)

    # Count how many items are Done / In progress / Needs review.
    done_count = (compliance_df["status"] == "Done").sum()
    in_progress_count = (compliance_df["status"] == "In progress").sum()
    needs_review_count = (compliance_df["status"] == "Needs review").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Done", int(done_count))
    c2.metric("In progress", int(in_progress_count))
    c3.metric("Needs review", int(needs_review_count))

    # Important disclaimer: this is only a prototype, not formal compliance.
    st.info(
        "This checklist is a QA readiness prototype and does not represent formal "
        "regulatory compliance."
    )

    st.divider()

    # --- Why QA matters ---
    st.subheader("Why QA testing matters for healthcare AI agents")
    st.markdown(
        """
        Healthcare AI agents interact with vulnerable users and sensitive data, so
        quality assurance is critical. Structured QA testing helps ensure the agent:

        - **Catches red-flag symptoms** (e.g. chest pain, stroke signs) and escalates to
          emergency care instead of giving casual advice.
        - **Avoids unsafe behaviour** such as giving firm diagnoses or prescribing
          medications and dosages without a doctor.
        - **Protects privacy (GDPR)** by never requesting or storing sensitive personal
          identifiers like AMKA or ID numbers.
        - **Stays consistent and empathetic**, responding in the user's language and with
          appropriate care in stressful situations.

        By running synthetic scenarios like the ones above, QA Analysts can measure how
        safely and reliably an AI agent behaves *before* it reaches real patients.
        """
    )


if __name__ == "__main__":
    main()
