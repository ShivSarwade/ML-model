---
name: streamlit-deployment
description: >-
  Builds the required interactive dashboard to visualize predictions and historical insights.
---

# STREAMLIT DEPLOYMENT

## PURPOSE
To fulfill the project requirement of deploying a working application (Streamlit or Power BI) for faculty to proactively assess attendance.

## WHEN TO USE
After final-model-validation passes.

## INPUTS
- prediction-pipeline
- Cleaned dataset (for historical visualizations).

## OUTPUTS
- Streamlit application code (`app.py`).

## RESPONSIBILITIES
- Implement required UI: predict upcoming lectures, identify low-attendance slots, identify poor-attendance subjects, and show holiday effects.
- Integrate the prediction-pipeline.

## RULES
- Keep the architecture simple (no external APIs/backends unless explicitly required).
- Adhere strictly to the visualization requirements specified in the PDF.

## VALIDATION
- Ensure the app launches successfully locally without missing dependency errors.
- Verify user inputs correctly update the prediction display.

## FAILURE CONDITIONS
- Application crashes on valid inputs.
- Extremely high latency on predictions making the UI unusable.

## EDGE CASES
- User inputs illogical combinations (e.g., 3 AM lecture on a Sunday): Provide UI warnings.

## EXAMPLES
- `st.sidebar.selectbox('Subject', subject_list)`

## DOWNSTREAM DEPENDENCIES
- project-documentation
