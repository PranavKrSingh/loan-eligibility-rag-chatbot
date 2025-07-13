
# RAG Q&A Chatbot for Loan Eligibility – **Enhanced**

This repository is your Week‑8 capstone integrating everything you've learned:

| Week | Concept | Implementation in this repo |
|------|---------|------------------------------|
| 1 | Python basics | clean, modular scripts |
| 2 | OOP in Python | classes for data pipeline & chatbot |
| 3 | Data Science with Python | `eda/` notebook + pandas |
| 4 | Feature Engineering | `feature_engineering/preprocess.py` |
| 5 | Regression (classification) | `models/train_classifier.py` (LogisticRegression) |
| 6 | Clustering | `clustering/segment.py` (K‑means) |
| 7 | SQL Basics | `sql/setup_db.py` – loads CSV into SQLite for ad‑hoc queries |
| 8 | GenAI (RAG) | `retriever/`, `llm/`, `app.py` |

## Quick Start

1. `pip install -r requirements.txt`
2. Copy the three CSVs into **data/**.
3. **Data prep**: `python feature_engineering/preprocess.py`
4. **Train model**: `python models/train_classifier.py`
5. **Index rows**: `python retriever/create_index.py`
6. **Chatbot**: `python app.py`

The chatbot can now:
* Explain dataset stats
* Run SQL-style lookups
* Retrieve similar applicants
* Predict loan approval (`Loan_Status`) for custom inputs
