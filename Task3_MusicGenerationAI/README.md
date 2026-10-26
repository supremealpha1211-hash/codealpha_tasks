# CodeAlpha - Task 3: Music Generation with AI 🎵

A Deep Learning Recurrent Neural Network (LSTM) for musical sequence composition and MIDI synthesis, developed for the **CodeAlpha Artificial Intelligence Internship**.

## 📌 Features
- **Deep Learning Architecture**: PyTorch Multi-layer LSTM neural network trained to understand temporal note dependencies and multi-genre harmonic transitions (Classical, Jazz, Ambient).
- **Genre & Seed Customization**: Choose starting musical root/keys (C3–B5) and musical styles with dynamic temperature / creativity sampling.
- **MIDI Encoding & Export**: Converts generated symbolic note arrays directly into standard MIDI (`.mid`) files using `mido`.
- **Interactive UI & Visualizer**:
  - Live virtual piano roll animation
  - Web Audio API real-time synthesizer playback directly in browser
  - Direct 1-click `.MID` file downloader

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- Dependencies: `Flask`, `torch`, `mido`, `numpy`

### 2. Installation & Running
```bash
cd CodeAlpha_MusicGenerationAI
pip install -r requirements.txt
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5003
```

## 🛠 Tech Stack
- **Deep Learning Model**: PyTorch (LSTM + Embedding layers)
- **Audio & MIDI Engine**: Mido (MIDI generation) & Web Audio API (realtime browser synthesis)
- **Web Server**: Flask
- **Frontend**: HTML5, Modern CSS3, Interactive Audio JavaScript

## 👤 Author
- **Intern Name**: Raiyan Kammal
- **Company**: CodeAlpha
