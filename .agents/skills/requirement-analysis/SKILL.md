---
name: requirement-analysis
description: >-
  Reads the project specification, extracts requirements (mandatory vs optional), deliverables, and creates a tracked checklist.
---

# REQUIREMENT ANALYSIS

## PURPOSE
To extract, categorize, and track all ML project requirements strictly from the authoritative specification document.

## WHEN TO USE
At the very beginning of the project, or whenever the project specification PDF/document is updated.

## INPUTS
- Project specification PDF or MD file.

## OUTPUTS
- Machine-readable requirement specification.
- Tracked requirement checklist distinguishing MANDATORY, OPTIONAL, and NOT SPECIFIED items.

## RESPONSIBILITIES
- Extract mandatory/optional algorithms, technical stacks, and metrics.
- Identify exact deliverables required for project completion.
- Track implementation status across the project lifecycle.

## RULES
- You MUST distinguish between MANDATORY, OPTIONAL, and NOT SPECIFIED.
- You MUST NEVER invent or hallucinate requirements.
- If a technology (e.g., React, Node.js, LLMs, Docker) is not mentioned, explicitly mark it as NOT SPECIFIED and prohibit its use.

## VALIDATION
- Cross-reference every item in the output checklist against a specific section of the original PDF.

## FAILURE CONDITIONS
- Source document cannot be parsed or read.
- Conflicting requirements detected without a clear resolution.

## EDGE CASES
- Ambiguous requirements: Mark as 'Requires User Clarification' rather than guessing.

## EXAMPLES
- Input text: "Optionally evaluate using XGBoost." -> Output: `OPTIONAL: XGBoost algorithm.`

## DOWNSTREAM DEPENDENCIES
- dataset-validation, model-training, streamlit-deployment, project-documentation
