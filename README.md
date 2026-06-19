# 🩺 AI Healthcare QA Evaluator

A proof-of-concept **Quality Assurance (QA) framework** for testing AI healthcare
agents using **synthetic patient scenarios**. Built with Python and Streamlit.

## Live Demo

You can try the app here:
https://ai-healthcare-evaluator.streamlit.app/ 

---

## 1. Project Overview

**AI Healthcare QA Evaluator** is a proof-of-concept QA framework that shows how a
QA Analyst can systematically test an AI healthcare agent before it reaches real
patients.

It uses **synthetic data only** (no real patients) to simulate the full QA loop:
defining test cases, evaluating AI responses against safety rules, analysing
failures, improving the prompt, validating the improvement, and monitoring quality
over time. Everything is presented in a simple, interactive Streamlit dashboard.

## 2. Why I Built This

I built this project to explore and demonstrate the responsibilities of a **QA
Analyst role focused on AI healthcare agents**. Testing AI in healthcare is very
different from testing traditional software: the risks include unsafe medical
advice, missed emergencies, privacy breaches, and loss of patient trust.

This project is my way of showing how I would approach that challenge — turning
abstract safety goals into concrete, measurable, and repeatable QA checks.

## 3. What the Project Does

- **Synthetic healthcare test cases** in **Greek and English**, covering normal
  symptoms, red-flag emergencies, diagnosis requests, privacy/GDPR situations,
  language consistency, over-advice, and empathy.
- **Rule-based evaluation** of AI agent responses using simple keyword checks
  (no AI or machine learning involved).
- **PASS / FAIL results** for each test case, with a pass rate and per-case detail.
- **Failure analysis** that documents root causes, suggested fixes, and impact.
- **Prompt improvement** comparing a weak baseline prompt with a safer improved one.
- **Before/after validation** that measures whether the prompt change actually
  improved agent behaviour (v1 vs v2).
- **Quality thresholds** defining the minimum acceptable safety standards.
- **Monitoring simulation** showing how key metrics trend over 7 days.
- **Compliance readiness checklist** tracking QA and documentation across areas
  such as clinical safety, privacy, validation, and monitoring.

## 4. Important Disclaimer

Please read carefully:

- This is **not a medical tool**.
- It **does not provide diagnosis** or medical advice.
- It **does not use real patient data** — all scenarios are synthetic.
- It is **not formally compliant** with GDPR, HIPAA, MDR, or the EU AI Act.
- It is a **QA prototype for demonstration purposes only**.

## 5. Tech Stack

- **Python** — core language
- **Streamlit** — interactive dashboard UI
- **pandas** — data handling and tables
- **JSON / CSV** — lightweight data storage for test cases, responses, and metrics
- **Rule-based evaluation** — transparent keyword checks (no external AI APIs, no ML)

## 6. How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the dashboard in your browser (usually at
`http://localhost:8501`).

## 7. Project Structure

```
.
├── app.py                          # Streamlit dashboard (all sections)
├── evaluator.py                    # Rule-based QA evaluator logic
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── data/
    ├── test_cases.json             # Synthetic healthcare QA test cases (GR/EN)
    ├── agent_responses_v1.json     # Baseline AI agent responses
    ├── agent_responses_v2.json     # Improved AI agent responses (after prompt fix)
    ├── prompt_versions.json        # Baseline vs improved prompt
    ├── failure_analysis.json       # Root-cause analysis of failed cases
    ├── quality_thresholds.json     # Minimum quality standards
    ├── monitoring_metrics.csv      # Simulated 7-day monitoring data
    └── compliance_checklist.json   # QA readiness / documentation checklist
```

- **`app.py`** — loads the data and renders every dashboard section.
- **`evaluator.py`** — contains the reusable, rule-based checks and the
  `evaluate_response_set()` function used to score any response set.
- **`data/`** — all synthetic inputs: test cases, agent responses, prompts,
  analysis, thresholds, monitoring data, and the compliance checklist.

## 8. What I Learned

Building this project helped me demonstrate end-to-end **AI QA thinking** for
healthcare:

- **Safety evaluation** — translating safety goals (escalate emergencies, never
  diagnose, protect privacy) into concrete, testable checks.
- **Failure analysis** — investigating *why* an AI response fails and what the
  real-world impact could be.
- **Prompt improvement** — using QA findings to strengthen a prompt and reduce risk.
- **Before/after validation** — proving with data that a change actually improved
  behaviour, instead of assuming it did.
- **Documentation & monitoring** — defining quality thresholds, tracking metrics
  over time, and maintaining a compliance readiness checklist.

> ⚠️ **Reminder:** This is a QA readiness prototype for demonstration only and does
> not represent formal regulatory compliance.
