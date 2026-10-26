# CodeAlpha — Artificial Intelligence Internship Tasks 🤖🚀

Welcome to the **CodeAlpha Artificial Intelligence Internship** project repository! This repository contains the source code, interactive web applications, datasets, and documentation for **3 completed AI tasks**.

---

## 📌 Repository Overview

```
codealpha_tasks/
├── Task1_LanguageTranslationTool/   # Task 1: Real-time Multi-Language Translator with TTS
├── Task2_FAQChatbot/                # Task 2: NLP Intent Matching FAQ Support Chatbot
├── Task3_MusicGenerationAI/         # Task 3: Deep Learning LSTM Music Generator & MIDI Synthesizer
└── README.md
```

---

## 🛠 Tasks Summary

### 1️⃣ Task 1: Language Translation Tool
* **Directory**: [`Task1_LanguageTranslationTool/`](./Task1_LanguageTranslationTool)
* **Description**: Real-time neural machine translation tool supporting over 20 languages with auto-detection.
* **Key Features**:
  - Source and target language selection with instant language swapping.
  - Integration with Google Neural Translation API.
  - Interactive Text-to-Speech (TTS) voice playback for both input and translated text.
  - One-click copy to clipboard and dynamic character counting.
* **Tech Stack**: Python, Flask, Google Translate API, HTML5/CSS3, Web Speech API.
* **Run Locally**:
  ```bash
  cd Task1_LanguageTranslationTool
  pip install -r requirements.txt
  python app.py
  # Access at: http://127.0.0.1:5001
  ```

---

### 2️⃣ Task 2: Chatbot for FAQs
* **Directory**: [`Task2_FAQChatbot/`](./Task2_FAQChatbot)
* **Description**: An NLP-powered intelligent conversational agent designed to answer queries regarding CodeAlpha internship rules, evaluation criteria, and foundational AI/NLP concepts.
* **Key Features**:
  - Comprehensive NLP preprocessing pipeline: lowercasing, punctuation stripping, tokenization, stopword removal, and WordNet lemmatization.
  - TF-IDF n-gram vectorization with Cosine Similarity intent scoring.
  - Match confidence scoring and low-confidence fallback prompts.
  - Modern web chat UI with quick-question suggestion chips and real-time typing indicators.
* **Tech Stack**: Python, NLTK, Scikit-Learn, Flask, Modern CSS.
* **Run Locally**:
  ```bash
  cd Task2_FAQChatbot
  pip install -r requirements.txt
  python app.py
  # Access at: http://127.0.0.1:5002
  ```

---

### 3️⃣ Task 3: Music Generation with AI
* **Directory**: [`Task3_MusicGenerationAI/`](./Task3_MusicGenerationAI)
* **Description**: A Deep Learning Recurrent Neural Network (LSTM) trained to learn sequential note patterns, harmonic motifs, and compose original melodies across multiple styles (Classical, Jazz, Ambient).
* **Key Features**:
  - Multi-layer PyTorch LSTM neural network with embedding and dropout layers.
  - Temperature/creativity control and musical seed customization.
  - Symbolic note-to-MIDI synthesis converter exporting standard `.mid` files.
  - In-browser interactive piano roll visualizer and Web Audio API synthesizer.
* **Tech Stack**: Python, PyTorch, Mido, Flask, Web Audio API.
* **Run Locally**:
  ```bash
  cd Task3_MusicGenerationAI
  pip install -r requirements.txt
  python app.py
  # Access at: http://127.0.0.1:5003
  ```

---

## ⚙️ Quick Start (Run Any Task)

Clone the repository and install dependencies for the task of your choice:

```bash
git clone https://github.com/supremealpha1211-hash/codealpha_tasks.git
cd codealpha_tasks
```

Navigate to any task directory and launch its `app.py`.

---

## 👨‍💻 Author
- **Intern Name**: Raiyan Kammal
- **GitHub**: [@supremealpha1211-hash](https://github.com/supremealpha1211-hash)
- **Organization**: CodeAlpha
