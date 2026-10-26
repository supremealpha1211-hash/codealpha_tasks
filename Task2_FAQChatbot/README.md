# CodeAlpha - Task 2: Chatbot for FAQs 🤖

An NLP-powered intelligent FAQ Chatbot built for the **CodeAlpha Artificial Intelligence Internship**.

## 📌 Features
- **FAQ Knowledge Base**: JSON-driven dataset covering CodeAlpha internship queries, guidelines, and core AI/NLP topics.
- **NLP Preprocessing Pipeline**: Tokenization, punctuation removal, stopword filtering, and WordNet lemmatization using NLTK.
- **Intent Matching**: TF-IDF (Term Frequency - Inverse Document Frequency) n-gram vectorization paired with Cosine Similarity scoring.
- **Thresholding & Confidence**: Automatically detects low-confidence questions and prompts user to rephrase.
- **Interactive Chat Interface**: Dark modern UI featuring real-time typing indicators, quick-question chips, and similarity match reporting.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- Dependencies: `Flask`, `nltk`, `scikit-learn`, `numpy`

### 2. Installation & Running
```bash
cd CodeAlpha_FAQChatbot
pip install -r requirements.txt
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5002
```

## 🛠 Tech Stack
- **Backend / NLP**: Python, NLTK, Scikit-Learn (TF-IDF, Cosine Similarity)
- **Web Server**: Flask
- **Frontend**: HTML5, Modern CSS3, JavaScript (Fetch API)

## 👤 Author
- **Intern Name**: Raiyan Kammal
- **Company**: CodeAlpha
