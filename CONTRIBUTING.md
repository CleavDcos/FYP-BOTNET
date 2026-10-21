# Contributing Guidelines: IoT Botnet Detection Project

Welcome to the team! This repository houses the Machine Learning-Based IoT Botnet Detection System developed using the UNSW BoT-IoT dataset.

---

## 1. Branching Strategy

To enable all team members to work in parallel without merge conflicts, adhere strictly to the feature-branch workflow:

- `main`: Production-ready, stable codebase. Direct commits to `main` are prohibited.
- `feature/<feature-name>`: Dedicated branches for new features, modules, or models (e.g., `feature/baseline-models`, `feature/shap-explainability`, `feature/dashboard-ui`).
- `fix/<bug-name>`: Targeted branches for fixing pipeline or evaluation bugs.
- `experiment/<experiment-name>`: Exploratory branches for hyperparameter searches or testing experimental algorithms.

### Creating a Branch
Always branch off the latest `main`:
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

---

## 2. Parallel Workstreams & Modular Ownership

To prevent overlapping edits, team tasks are modularized across distinct directories:

| Workstream | Directory Focus | Primary Responsibilities |
| :--- | :--- | :--- |
| **Model Development** | `src/models/` | Training baseline classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) using preprocessed datasets in `data/processed/`. |
| **Explainability** | `src/explainability/` | Integrating SHAP TreeExplainer / KernelExplainer to interpret feature importances. |
| **Risk & Mitigation** | `src/risk_assessment/` | Developing rule-based severity scoring and mitigation recommendations. |
| **Dashboard** | `app/dashboard/` & `app/components/` | Streamlit user interface displaying inferences, metrics, SHAP waterfall plots, and risk advisories. |
| **Testing** | `tests/` | Unit tests for data transforms, inference pipelines, and evaluation routines. |

---

## 3. Pull Request (PR) Workflow

1. **Commit Early and Clearly:**
   Use descriptive commit messages:
   - `feat(models): implement baseline Random Forest classifier`
   - `fix(preprocessing): handle missing port values gracefully`
   - `test(models): add unit tests for category evaluation metrics`
2. **Never Commit Large Datasets or Models:**
   - Raw CSVs (`data/raw/`) and Parquet splits (`data/processed/`) are ignored by `.gitignore`.
   - Never force-add (`git add -f`) large binary or dataset files.
3. **Verify Integrity Locally:**
   Before opening a PR, ensure all unit tests pass:
   ```bash
   python -m unittest discover tests
   ```
4. **Open a Pull Request:**
   - Push your branch to GitHub: `git push -u origin feature/your-feature-name`
   - Open a PR targeting `main`.
   - Provide a concise description of changes, metrics observed, and screenshots if adding UI elements.
   - Request review from at least one peer before merging.

---

## 4. Coding Standards

- **Python Version:** Python 3.10+
- **Styling:** Follow PEP 8 guidelines. Keep functions focused and docstrings descriptive.
- **Reproducibility:** Always pass `random_state=42` to any stochastic function, split, or estimator.
- **Leakage Prevention:** Never train transforms or encoders on validation or test sets. Fit strictly on the training fold.
