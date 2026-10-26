"""
CodeAlpha - Task 1: Language Translation Tool
Features:
- Source & Target language selection with auto-detection
- Real-time translation via Google Translate API
- Text-to-Speech (TTS) integration with responsive voice playback
- Quick Copy to Clipboard, Character counter, and language swap
- Beautiful responsive UI with clean modern styling
"""

import json
import urllib.parse
import urllib.request
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

LANGUAGES = {
    "auto": "Auto Detect",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "ar": "Arabic",
    "zh-CN": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "id": "Indonesian"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Language Translation Tool - CodeAlpha</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #06b6d4;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: radial-gradient(circle at 15% 15%, #1e1b4b 0%, #0f172a 60%, #030712 100%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2.5rem 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 1000px;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            background: rgba(79, 70, 229, 0.2);
            color: #a5b4fc;
            font-size: 0.825rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            border: 1px solid rgba(79, 70, 229, 0.4);
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 30%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .translator-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 1.25rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }

        .controls-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border);
            gap: 1rem;
            background: rgba(15, 23, 42, 0.4);
        }

        .lang-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex: 1;
        }

        select {
            width: 100%;
            padding: 0.65rem 1rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border);
            border-radius: 0.6rem;
            color: var(--text-main);
            font-size: 0.95rem;
            font-weight: 500;
            outline: none;
            cursor: pointer;
            transition: all 0.2s;
        }

        select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.3);
        }

        .swap-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-muted);
            width: 42px;
            height: 42px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            flex-shrink: 0;
        }

        .swap-btn:hover {
            color: white;
            background: var(--primary);
            transform: rotate(180deg);
        }

        .panels-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            min-height: 280px;
        }

        @media (max-width: 768px) {
            .panels-grid {
                grid-template-columns: 1fr;
            }
            .controls-bar {
                flex-direction: column;
            }
            .swap-btn {
                transform: rotate(90deg);
            }
            .swap-btn:hover {
                transform: rotate(270deg);
            }
        }

        .panel {
            display: flex;
            flex-direction: column;
            padding: 1.5rem;
            position: relative;
        }

        .panel:first-child {
            border-right: 1px solid var(--border);
        }

        @media (max-width: 768px) {
            .panel:first-child {
                border-right: none;
                border-bottom: 1px solid var(--border);
            }
        }

        textarea {
            width: 100%;
            height: 200px;
            background: transparent;
            border: none;
            color: var(--text-main);
            font-size: 1.05rem;
            line-height: 1.6;
            resize: none;
            outline: none;
        }

        textarea::placeholder {
            color: #64748b;
        }

        .output-box {
            width: 100%;
            height: 200px;
            font-size: 1.05rem;
            line-height: 1.6;
            overflow-y: auto;
            color: #e2e8f0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .output-placeholder {
            color: #64748b;
        }

        .panel-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: auto;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }

        .char-count {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .action-btns {
            display: flex;
            gap: 0.5rem;
        }

        .tool-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 0.4rem 0.75rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .tool-btn:hover {
            color: white;
            background: rgba(255, 255, 255, 0.1);
        }

        .bottom-action-bar {
            padding: 1.25rem 1.5rem;
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 1rem;
            background: rgba(15, 23, 42, 0.6);
            border-top: 1px solid var(--border);
        }

        .primary-btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.75rem;
            border-radius: 0.6rem;
            font-weight: 600;
            font-size: 0.95rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .primary-btn:hover {
            background: var(--primary-hover);
            transform: translateY(-1px);
        }

        .primary-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #10b981;
            color: white;
            padding: 0.75rem 1.25rem;
            border-radius: 0.5rem;
            font-weight: 500;
            font-size: 0.9rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            display: none;
            animation: fadeIn 0.2s ease-out;
            z-index: 100;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="badge"><i class="fa-solid fa-brain"></i> CodeAlpha AI Internship — Task 1</div>
            <h1>Language Translation Tool</h1>
            <p>Real-time neural machine translation with Text-to-Speech and smart language detection</p>
        </div>

        <div class="translator-card">
            <div class="controls-bar">
                <div class="lang-group">
                    <label style="font-size: 0.85rem; color: var(--text-muted); font-weight: 600;">From:</label>
                    <select id="sourceLang">
                        {% for code, name in languages.items() %}
                            <option value="{{ code }}">{{ name }}</option>
                        {% endfor %}
                    </select>
                </div>

                <button class="swap-btn" id="swapBtn" title="Swap Languages">
                    <i class="fa-solid fa-arrow-right-arrow-left"></i>
                </button>

                <div class="lang-group">
                    <label style="font-size: 0.85rem; color: var(--text-muted); font-weight: 600;">To:</label>
                    <select id="targetLang">
                        {% for code, name in languages.items() if code != 'auto' %}
                            <option value="{{ code }}" {% if code == 'es' %}selected{% endif %}>{{ name }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>

            <div class="panels-grid">
                <div class="panel">
                    <textarea id="inputText" placeholder="Type or paste text here to translate... (e.g. 'Artificial Intelligence is revolutionizing modern technology')"></textarea>
                    <div class="panel-footer">
                        <span class="char-count" id="inputCount">0 / 5000 characters</span>
                        <div class="action-btns">
                            <button class="tool-btn" id="speakInputBtn" title="Listen"><i class="fa-solid fa-volume-high"></i> TTS</button>
                            <button class="tool-btn" id="clearBtn" title="Clear text"><i class="fa-solid fa-xmark"></i> Clear</button>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="output-box" id="outputText"><span class="output-placeholder">Translation will appear here...</span></div>
                    <div class="panel-footer">
                        <span class="char-count" id="detectedLang"></span>
                        <div class="action-btns">
                            <button class="tool-btn" id="speakOutputBtn" title="Listen Translation"><i class="fa-solid fa-volume-high"></i> Speak</button>
                            <button class="tool-btn" id="copyBtn" title="Copy Translation"><i class="fa-solid fa-copy"></i> Copy</button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="bottom-action-bar">
                <button class="primary-btn" id="translateBtn">
                    <i class="fa-solid fa-language"></i> Translate Now
                </button>
            </div>
        </div>
    </div>

    <div class="toast" id="toast">Copied to clipboard!</div>

    <script>
        const inputText = document.getElementById('inputText');
        const outputText = document.getElementById('outputText');
        const sourceLang = document.getElementById('sourceLang');
        const targetLang = document.getElementById('targetLang');
        const translateBtn = document.getElementById('translateBtn');
        const swapBtn = document.getElementById('swapBtn');
        const copyBtn = document.getElementById('copyBtn');
        const clearBtn = document.getElementById('clearBtn');
        const inputCount = document.getElementById('inputCount');
        const detectedLang = document.getElementById('detectedLang');
        const toast = document.getElementById('toast');
        const speakInputBtn = document.getElementById('speakInputBtn');
        const speakOutputBtn = document.getElementById('speakOutputBtn');

        let rawTranslation = "";

        function showToast(msg) {
            toast.textContent = msg;
            toast.style.display = 'block';
            setTimeout(() => { toast.style.display = 'none'; }, 2500);
        }

        inputText.addEventListener('input', () => {
            inputCount.textContent = `${inputText.value.length} / 5000 characters`;
        });

        clearBtn.addEventListener('click', () => {
            inputText.value = '';
            outputText.innerHTML = '<span class="output-placeholder">Translation will appear here...</span>';
            inputCount.textContent = '0 / 5000 characters';
            detectedLang.textContent = '';
            rawTranslation = '';
        });

        swapBtn.addEventListener('click', () => {
            if (sourceLang.value === 'auto') {
                showToast("Cannot swap when 'Auto Detect' is selected.");
                return;
            }
            const temp = sourceLang.value;
            sourceLang.value = targetLang.value;
            targetLang.value = temp;

            if (rawTranslation) {
                inputText.value = rawTranslation;
                doTranslate();
            }
        });

        async function doTranslate() {
            const text = inputText.value.trim();
            if (!text) {
                showToast("Please enter some text to translate!");
                return;
            }

            translateBtn.disabled = true;
            translateBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Translating...';

            try {
                const res = await fetch('/api/translate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        source: sourceLang.value,
                        target: targetLang.value
                    })
                });

                const data = await res.json();
                if (data.error) {
                    outputText.textContent = `Error: ${data.error}`;
                } else {
                    rawTranslation = data.translated_text;
                    outputText.textContent = data.translated_text;
                    if (data.detected_lang) {
                        detectedLang.textContent = `Detected: ${data.detected_lang.toUpperCase()}`;
                    }
                }
            } catch (err) {
                outputText.textContent = "Failed to communicate with translation API.";
            } finally {
                translateBtn.disabled = false;
                translateBtn.innerHTML = '<i class="fa-solid fa-language"></i> Translate Now';
            }
        }

        translateBtn.addEventListener('click', doTranslate);

        copyBtn.addEventListener('click', () => {
            if (!rawTranslation) {
                showToast("Nothing to copy!");
                return;
            }
            navigator.clipboard.writeText(rawTranslation).then(() => {
                showToast("Translation copied to clipboard!");
            });
        });

        function speak(text, lang) {
            if (!text) {
                showToast("No text to speak!");
                return;
            }
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                if (lang && lang !== 'auto') {
                    utterance.lang = lang;
                }
                window.speechSynthesis.speak(utterance);
            } else {
                showToast("Text-to-speech not supported in this browser.");
            }
        }

        speakInputBtn.addEventListener('click', () => {
            speak(inputText.value, sourceLang.value);
        });

        speakOutputBtn.addEventListener('click', () => {
            speak(rawTranslation, targetLang.value);
        });
    </script>
</body>
</html>
"""

def query_google_translate(text: str, source: str = "auto", target: str = "en") -> tuple[str, str]:
    url = (
        f"https://translate.googleapis.com/translate_a/single?client=gtx"
        f"&sl={urllib.parse.quote(source)}&tl={urllib.parse.quote(target)}&dt=t"
        f"&q={urllib.parse.quote(text)}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        result = json.loads(response.read().decode("utf-8"))
        
    translated_parts = [segment[0] for segment in result[0] if segment and segment[0]]
    translated_text = "".join(translated_parts)
    detected_lang = result[2] if len(result) > 2 and isinstance(result[2], str) else source
    return translated_text, detected_lang

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, languages=LANGUAGES)

@app.route("/api/translate", methods=["POST"])
def translate_api():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    source = data.get("source", "auto")
    target = data.get("target", "en")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        translated_text, detected = query_google_translate(text, source, target)
        return jsonify({
            "translated_text": translated_text,
            "detected_lang": detected,
            "source": source,
            "target": target
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("CodeAlpha Task 1: Starting Language Translation Tool at http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
