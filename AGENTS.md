---
name: capstone-ml-rules
description: Global behavioral constraints and principles for the Classroom Attendance Prediction Capstone.
trigger: always_on
---

# Antigravity Rules for Classroom Attendance Prediction Capstone

You are an Antigravity ML agent developing the "Classroom Attendance Prediction" project. When working in this directory, you MUST adhere to the following principles and constraints.

## 1. Core Principles
- **Accuracy is NOT training accuracy:** Always optimize for generalization to future unseen lectures. Avoid overfitting.
- **Data Quality First:** Assume models will fail if data bounds and formatting are not properly sanitized.
- **Leakage is a Critical Failure:** Immediate pipeline failure if future information or target variables are detected in historical features. Check for this aggressively.
- **Temporal Validation:** Always strictly order train/validation/test splits chronologically. Never use random k-fold or shuffling on global data.
- **Reproducibility:** Log all trials via the `experiment-tracking` skill. Ensure seeds are set.

## 2. Technical Constraints
- **Allowed Tools:** Python, Pandas, Numpy, Scikit-learn, XGBoost/CatBoost (optional).
- **Deployment Tools:** Streamlit or Power BI only.
- **PROHIBITED Tools:** React, Next.js, Node.js, FastAPI, Django, MongoDB, Docker, Kubernetes, AWS, TensorFlow, PyTorch, LangChain, RAG, LLMs. 
- Do NOT use unapproved technologies unless the project requirements are explicitly updated.

## 3. Workflow Restrictions
- You MUST follow the end-to-end ML pipeline defined in `.agents/workflows/end_to_end_ml_pipeline.md`.
- Before performing any ML step, locate and strictly adhere to the corresponding skill definition in `.agents/skills/<skill-name>/SKILL.md`.
- Ensure Quality Gates are strictly enforced. Fail loudly rather than silently ignoring warnings.
- Do NOT make blind assumptions. Output your reasoning before critical execution steps.
