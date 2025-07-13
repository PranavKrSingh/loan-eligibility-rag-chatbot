
<!--
README for: https://github.com/PranavKrSingh/loan-eligibility-rag-chatbot
-->

<h1 align="center">🏦 Loan Eligibility RAG Chatbot</h1>

<p align="center">
  <a href="https://loan-eligibility-rag-chatbot-pvaufufengqtonf3igvwic.streamlit.app/" target="_blank">
    <img alt="Streamlit App" src="https://img.shields.io/badge/Live%20Demo-Open-green?logo=streamlit&logoColor=white">
  </a>
  <a href="https://github.com/PranavKrSingh/loan-eligibility-rag-chatbot/blob/main/LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
  <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10+-yellow.svg">
</p>

> **Live Demo:**  
> 🌐 <https://loan-eligibility-rag-chatbot-pvaufufengqtonf3igvwic.streamlit.app/>

---

## 📜 Table of Contents
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Tech Stack](#tech-stack)  
4. [Quick Start](#quick-start)  
5. [Detailed Setup](#detailed-setup)  
6. [Project Structure](#project-structure)  
7. [Model Training & Retrieval Pipeline](#model-training--retrieval-pipeline)  
8. [Dataset](#dataset)  
9. [Demo Screenshots](#demo-screenshots)  
10. [Roadmap](#roadmap)  
11. [Contributing](#contributing)  
12. [License](#license)  

---

## Overview
This project combines **machine‑learning prediction** with **Retrieval‑Augmented Generation (RAG)** to create a conversational assistant that can:

1. **Predict** whether a home‑loan application will be **Approved** or **Rejected**  
2. **Explain** or answer questions about loan features and policy documents (RAG module)  
3. Provide both **single‑entry** and **batch CSV** scoring through a modern Streamlit UI  

The solution was built as the Week‑8 GenAI capstone in my Data Science internship at **Celebal Technologies**.

---

## Key Features
| Type | Feature |
|------|---------|
| 🔮 | Logistic‑Regression (class‑balanced + scaled) model |
| 🖥️ | Streamlit UI: two‑column form, batch‑upload tab, confidence bar |
| 🛠️ | Feature‑engineering pipeline with log‑transform, label encoding, imputation |
| 📚 | FAISS‑based semantic search (RAG) scaffold for future document Q&A |
| ☁️ | 1‑click deploy on Streamlit Cloud (live link above) |
| 🧪 | 74 %+ validation accuracy on imbalanced dataset |

---

## Tech Stack
- **Python 3.10+**  
- **Streamlit 1.35** – web interface  
- **Scikit‑learn** – preprocessing & Logistic Regression  
- **FAISS + Sentence‑Transformers** – semantic retrieval (RAG)  
- **Pandas / NumPy** – data wrangling  
- **Joblib** – model persistence  

---

## Quick Start
```bash
# 1. Clone
git clone https://github.com/PranavKrSingh/loan-eligibility-rag-chatbot.git
cd loan-eligibility-rag-chatbot

# 2. Create virtual env (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Win |  source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
streamlit run streamlit_app.py
````

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Detailed Setup

| Step                    | Command                                    |
| ----------------------- | ------------------------------------------ |
| **Data preprocessing**  | `python feature_engineering/preprocess.py` |
| **Model training**      | `python models/train_classifier.py`        |
| **Build FAISS index**   | `python retriever/create_index.py`         |
| **Launch CLI bot**      | `python app.py`                            |
| **Launch Streamlit UI** | `streamlit run streamlit_app.py`           |

Each step is modular—run only what you need. Model + encoders are saved under `models/` and `data/`.

---

## Project Structure

```
├── streamlit_app.py          # Front‑end
├── app.py                    # Terminal chatbot (optional)
├── data/
│   ├── Training Dataset.csv
│   ├── Test Dataset.csv
│   ├── train_fe.csv          # engineered
│   ├── label_encoders.pkl
├── feature_engineering/
│   └── preprocess.py         # FE pipeline
├── models/
│   ├── train_classifier.py
│   ├── loan_model.joblib     # saved pipeline
├── retriever/
│   ├── create_index.py
│   ├── retriever.py
│   └── faiss.index
├── clustering/               # bonus customer segmentation
└── README.md
```

---

## Model Training & Retrieval Pipeline

```mermaid
graph TD
A[Raw CSV Files] --> B[Preprocess\n(log‑transform, fillna,\nlabel‑encode)]
B --> C[train_fe.csv]
C --> D[Train ML Pipeline\n(StandardScaler +\nLogReg balanced)]
D --> E[joblib dump]
C --> F[Sentence‑Transformer\nEmbeddings]
F --> G[FAISS index]
```

---

## Dataset

* **Source:** Analytics Vidhya – Dream Housing Loan Prediction
* **Rows:** 614 training, 367 test
* **Target:** `Loan_Status (Y/N)`
* **Features:** Gender, Married, Dependents, Education, Income, LoanAmount, Credit\_History, Property\_Area, etc.

> Class distribution: 68.7 % approved vs 31.3 % rejected → mitigated with `class_weight='balanced'`.

