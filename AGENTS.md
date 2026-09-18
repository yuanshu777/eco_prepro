# Project workflow

## Repository and completion

The user designated https://github.com/yuanshu777/eco_prepro.git as this project's repository.
After completing authorized work, run the relevant checks, commit the finished code and documentation,
and push `main` to `origin/main`. This standing instruction authorizes ordinary commits and pushes;
do not ask for permission again for each completed task. Report the commit and actual push outcome.
If authentication or remote changes block a push, preserve the local commit and explain the blocker.
Never force-push or overwrite remote work without explicit authorization.

## Publication boundaries

Keep raw datasets, processed videos, report ZIPs, credentials, and restricted dataset-derived QC images
local. `data/`, `outputs/`, `logs/`, and the virtual environment are not publication artifacts.
Unity-derived processed images must never be published. Other dataset restrictions are recorded in
`configs/datasets.yaml` and `docs/STAGE1.md`. Only the explicitly attributed EV9V figures already
approved in `docs/figures/` may be included; review the policy before adding any other data images.
Code, non-image reports, manifests, and tabular audit results can be committed after review.

The local `legacy/local-before-github` branch contains historical restricted QC images. Keep that
branch local: never push it, use `git push --all`, or merge it into the published history.
The public `main` history starts from the cleaned Stage 1 snapshot.

## Current scope

Preprocessing only. Stage 1 freezes the heterogeneous audit and adds acquisition reporting, early
pass-through, explicit statuses, preservation of uncertain cases, FPS provenance, and visual QC.
Do not introduce classifier training, downstream benchmarks, learned segmentation or new geometry
fitting without a new user instruction.

Use the existing `.venv`; tests: `.venv/bin/python -m pytest`; lint:
`.venv/bin/ruff check echoprep scripts tests`. Each audit run requires a new output directory.
