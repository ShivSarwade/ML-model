---
name: no-cli-for-reading
description: Never use command line (python -c, grep, cat, etc.) for reading or inspecting files.
trigger: always_on
---

# No Command Line for File Reading

## Rule
NEVER use `python -c`, `cat`, `type`, `grep`, or any command-line tool to read, search, or inspect file contents. 

## Instead Use
- `view_file` — to read file contents
- `grep_search` — to find patterns in files
- `list_dir` — to list directory contents

## When Command Line IS Allowed
- Running actual scripts (e.g., `python retrain_all.py`)
- Running notebook execution (e.g., `jupyter nbconvert --execute`)
- Running the Streamlit app
- Running git commands
- Installing packages

## Why
The user explicitly prefers tool-based file interaction over command-line scripts. Command-line Python one-liners for reading files are unnecessary, harder to audit, and require user permission.
