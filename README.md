# 🧠 AI Fake News Detector

A complete end-to-end fake news classification system powered by deep learning and NLP.  
This project combines a fine-tuned DistilRoBERTa model with an interactive Gradio interface and FastAPI backend.

---

## 📌 Features

- 🔍 Detects whether a news statement is **Fake** or **Real**
- 🧠 Fine-tuned **DistilRoBERTa** model on a merged dataset
- 🛠️ RESTful API using **FastAPI**
- 🎛️ User-friendly UI via **Gradio**
- 💬 Feedback & logging system
- 🧪 Pytest coverage for endpoints

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the backend

```bash
uvicorn backend:app --reload --port 8001
```

### 3. Launch the frontend

```bash
python frontend.py
```

### 4.Run tests (optional)

```bash
pytest test_main.py
```
