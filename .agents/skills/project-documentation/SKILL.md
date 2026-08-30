---
name: project-documentation
description: >-
  Generates and synchronizes all final deliverables, READMEs, and the comprehensive final report.
---

# PROJECT DOCUMENTATION

## PURPOSE
To systematically compile all evidence, code, logs, and artifacts into the final portfolio required for capstone submission.

## WHEN TO USE
Continuously updated, with a final compilation pass after deployment is complete.

## INPUTS
- Requirement checklist.
- EDA findings, Feature dictionary.
- Experiment tracking ledger.
- Final test metrics.

## OUTPUTS
- README.md
- data_dictionary.md
- methodology.md
- final_report.pdf (or .md structure)

## RESPONSIBILITIES
- Compile the Problem Statement, Data Methodology, EDA Insights, Model Experiments, and Findings.
- Format the Experiment Table.
- Document deployment instructions.

## RULES
- NEVER claim that something was implemented if it wasn't.
- Documentation must remain perfectly synchronized with the actual codebase.

## VALIDATION
- Cross-reference the final report against the requirement-analysis checklist to ensure 100% coverage.

## FAILURE CONDITIONS
- Missing critical sections in the final report (e.g., no mention of data cleaning logic).

## EDGE CASES
- Findings contradict initial assumptions (e.g., Weather had zero impact): Document this transparently as a valid scientific finding.

## EXAMPLES
- Generating the Experiment Table markdown from `experiment_results.csv`.

## DOWNSTREAM DEPENDENCIES
- None. (End of workflow)
