"""
CodeAlpha - Task 2: Chatbot for FAQs
Features:
- NLTK & Scikit-Learn based NLP pipeline
- Text preprocessing (tokenization, lowercasing, punctuation stripping, stopwords, lemmatization)
- TF-IDF Vectorization and Cosine Similarity intent matching
- Confidence score thresholding and suggested follow-ups
- Clean interactive Chat UI with animated typing indicator and history
"""

import json
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template_string, request, jsonify

# Ensure NLTK resources are available
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

app = Flask(__name__)

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {"a", "an", "the", "in", "on", "is", "are", "and", "or", "to", "of", "it"}

# Load FAQs
FAQ_FILE = os.path.join(os.path.dirname(__file__), 'faqs.json')
with open(FAQ_FILE, 'r', encoding='utf-8') as f:
    FAQS = json.load(f)

def preprocess_text(text: str) -> str:
    """Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords, and lemmatize."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    tokens = text.split()
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    return ' '.join(processed_tokens)

# Prepare corpus and vectorizer
preprocessed_questions = [preprocess_text(faq['question']) for faq in FAQS]
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
faq_vectors = vectorizer.fit_transform(preprocessed_questions)

def find_best_answer(user_query: str, threshold: float = 0.25):
    processed_query = preprocess_text(user_query)
    if not processed_query.strip():
        # Fallback to direct raw query if all tokens were removed
        processed_query = user_query.lower()

    user_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(user_vector, faq_vectors).flatten()

    best_idx = similarities.argmax()
    best_score = float(similarities[best_idx])

    if best_score >= threshold:
        matched_faq = FAQS[best_idx]
        return {
            "answer": matched_faq["answer"],
            "matched_question": matched_faq["question"],
            "category": matched_faq.get("category", "General"),
            "confidence": round(best_score * 100, 1),
            "status": "matched"
        }
    else:
        return {
            "answer": "I'm sorry, I couldn't find an exact match for your question in our knowledge base. You can try rephrasing or asking about CodeAlpha, internship perks, submission guidelines, NLP, or AI concepts!",
            "matched_question": None,
            "category": None,
            "confidence": round(best_score * 100, 1),
            "status": "low_confidence"
        }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI FAQ Chatbot - CodeAlpha</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --bot-bubble: rgba(31, 41, 55, 0.9);
            --user-bubble: #4f46e5;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: radial-gradient(circle at 80% 20%, #1e1b4b 0%, #0b0f19 70%, #030712 100%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
        }

        .chat-wrapper {
            width: 100%;
            max-width: 800px;
            height: 90vh;
            max-height: 800px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 1.5rem;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        }

        .chat-header {
            padding: 1.25rem 1.75rem;
            background: rgba(15, 23, 42, 0.6);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .avatar {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            color: white;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
        }

        .header-title h2 {
            font-size: 1.15rem;
            font-weight: 700;
        }

        .status {
            font-size: 0.8rem;
            color: #10b981;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .status::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
        }

        .faq-badge {
            font-size: 0.75rem;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            background: rgba(99, 102, 241, 0.15);
            color: #c7d2fe;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        .quick-chips {
            padding: 0.75rem 1.75rem;
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
            white-space: nowrap;
            background: rgba(11, 15, 25, 0.4);
            border-bottom: 1px solid var(--border);
        }

        .chip {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s;
        }

        .chip:hover {
            color: white;
            background: rgba(99, 102, 241, 0.3);
            border-color: var(--primary);
        }

        .messages-container {
            flex: 1;
            padding: 1.5rem 1.75rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .message-row {
            display: flex;
            gap: 0.85rem;
            max-width: 85%;
        }

        .message-row.user {
            align-self: flex-end;
            flex-direction: row-reverse;
        }

        .message-row.bot {
            align-self: flex-start;
        }

        .msg-bubble {
            padding: 0.9rem 1.25rem;
            border-radius: 1.1rem;
            font-size: 0.95rem;
            line-height: 1.55;
            position: relative;
        }

        .message-row.user .msg-bubble {
            background: var(--user-bubble);
            color: white;
            border-bottom-right-radius: 0.2rem;
        }

        .message-row.bot .msg-bubble {
            background: var(--bot-bubble);
            color: #e5e7eb;
            border: 1px solid var(--border);
            border-bottom-left-radius: 0.2rem;
        }

        .confidence-tag {
            display: inline-block;
            margin-top: 0.5rem;
            font-size: 0.75rem;
            color: #94a3b8;
            background: rgba(0, 0, 0, 0.25);
            padding: 0.2rem 0.5rem;
            border-radius: 0.35rem;
        }

        .chat-input-bar {
            padding: 1rem 1.75rem;
            background: rgba(15, 23, 42, 0.6);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .chat-input {
            flex: 1;
            padding: 0.85rem 1.25rem;
            background: rgba(11, 15, 25, 0.8);
            border: 1px solid var(--border);
            border-radius: 9999px;
            color: white;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s;
        }

        .chat-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25);
        }

        .send-btn {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: var(--primary);
            color: white;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            transition: all 0.2s;
            flex-shrink: 0;
        }

        .send-btn:hover {
            background: var(--primary-hover);
            transform: scale(1.05);
        }

        .typing-indicator {
            display: none;
            align-self: flex-start;
            gap: 0.35rem;
            padding: 0.75rem 1.1rem;
            background: var(--bot-bubble);
            border-radius: 1rem;
            border: 1px solid var(--border);
        }

        .dot {
            width: 7px;
            height: 7px;
            background: #9ca3af;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="chat-wrapper">
        <div class="chat-header">
            <div class="header-left">
                <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="header-title">
                    <h2>CodeAlpha AI Support Bot</h2>
                    <div class="status">Online • TF-IDF & Cosine Similarity Matcher</div>
                </div>
            </div>
            <span class="faq-badge"><i class="fa-solid fa-graduation-cap"></i> CodeAlpha Task 2</span>
        </div>

        <div class="quick-chips">
            <span class="chip" onclick="askChip('What is CodeAlpha?')">What is CodeAlpha?</span>
            <span class="chip" onclick="askChip('What are the internship perks?')">Internship Perks</span>
            <span class="chip" onclick="askChip('How many tasks do I need to complete?')">Task Criteria</span>
            <span class="chip" onclick="askChip('How do I submit my completed tasks?')">Submission Guide</span>
            <span class="chip" onclick="askChip('What is Natural Language Processing?')">What is NLP?</span>
        </div>

        <div class="messages-container" id="messagesContainer">
            <div class="message-row bot">
                <div class="avatar" style="width: 32px; height: 32px; font-size: 0.85rem;"><i class="fa-solid fa-robot"></i></div>
                <div class="msg-bubble">
                    Hello! 👋 I am your <strong>CodeAlpha AI Assistant</strong>. Ask me anything about CodeAlpha internship guidelines, tasks, NLP, or AI technologies!
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>

        <div class="chat-input-bar">
            <input type="text" class="chat-input" id="userInput" placeholder="Ask a question about the internship, tasks, or AI..." autocomplete="off">
            <button class="send-btn" id="sendBtn" title="Send Question"><i class="fa-solid fa-paper-plane"></i></button>
        </div>
    </div>

    <script>
        const input = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const container = document.getElementById('messagesContainer');
        const indicator = document.getElementById('typingIndicator');

        function appendMessage(text, isUser = false, meta = null) {
            const row = document.createElement('div');
            row.className = `message-row ${isUser ? 'user' : 'bot'}`;

            if (!isUser) {
                const botAvatar = document.createElement('div');
                botAvatar.className = 'avatar';
                botAvatar.style.width = '32px';
                botAvatar.style.height = '32px';
                botAvatar.style.fontSize = '0.85rem';
                botAvatar.innerHTML = '<i class="fa-solid fa-robot"></i>';
                row.appendChild(botAvatar);
            }

            const bubble = document.createElement('div');
            bubble.className = 'msg-bubble';
            bubble.textContent = text;

            if (meta && meta.confidence && meta.status === 'matched') {
                const confTag = document.createElement('div');
                confTag.className = 'confidence-tag';
                confTag.innerHTML = `<i class="fa-solid fa-bullseye"></i> Match: ${meta.confidence}% • Matched: "${meta.matched_question}"`;
                bubble.appendChild(confTag);
            }

            row.appendChild(bubble);
            container.appendChild(row);
            container.scrollTop = container.scrollHeight;
        }

        async function handleSend() {
            const query = input.value.trim();
            if (!query) return;

            appendMessage(query, true);
            input.value = '';

            indicator.style.display = 'flex';
            container.scrollTop = container.scrollHeight;

            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await res.json();
                indicator.style.display = 'none';
                appendMessage(data.answer, false, data);
            } catch (err) {
                indicator.style.display = 'none';
                appendMessage("Sorry, I ran into an error processing your query. Please try again.", false);
            }
        }

        sendBtn.addEventListener('click', handleSend);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleSend();
        });

        function askChip(q) {
            input.value = q;
            handleSend();
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"answer": "Please ask a question!", "confidence": 0, "status": "empty"}), 400

    result = find_best_answer(query)
    return jsonify(result)

if __name__ == "__main__":
    print("CodeAlpha Task 2: Starting FAQ Chatbot at http://127.0.0.1:5002")
    app.run(host="127.0.0.1", port=5002, debug=False)
