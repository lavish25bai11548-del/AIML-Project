# 🎙️ Voice to Emotion Diary

> **Advanced AI-powered emotion detection from voice with 3D visualization**
> 
> Speak naturally — AI transcribes your speech and detects 7 distinct emotions in real-time, building a beautiful 3D emotional landscape of your day.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co)

### ✨ Features

- 🎤 **Offline Speech Recognition** - Powered by OpenAI Whisper (base model)
- 😊 **7-Class Emotion Detection** - Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
- 📊 **Real-time Visualization** - Live audio waveform and emotion confidence bars
- 🌌 **3D Emotion Landscape** - Interactive Plotly scatter plot showing valence-arousal-dominance
- 📝 **Daily Diary Summary** - Automatic emotional dashboard generation
- 💾 **JSON Export** - Save and download your emotional history
- 🎨 **Beautiful UI** - Gradient dark theme with smooth animations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Working microphone
- Windows / macOS / Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/voice-to-emotion-diary.git
cd voice-to-emotion-diary
Create virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
streamlit run advanced_emotion_diary.py
Open your browser
Navigate to http://localhost:8501

🎯 How It Works
Architecture Overview
text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│  Whisper (ASR)   │───▶│  Transcribed    │
│    Input        │    │  Speech-to-Text  │    │     Text        │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  3D Plotly      │◀───│  DistilRoBERTa   │◀───│  Emotion        │
│  Visualization  │    │  Classification  │    │  Detection      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
Models Used
Model	Purpose	Size	Source
Whisper (base)	Speech-to-Text	~142 MB	OpenAI
DistilRoBERTa	Emotion Classification	~500 MB	j-hartmann
Emotions Detected
Emotion	Emoji	Color	Description
Joy	😊🎉✨	#FFD700	Happiness, excitement, pleasure
Sadness	😢💔🌧️	#6495ED	Grief, disappointment, loneliness
Anger	😤🤬💢	#FF4500	Frustration, rage, irritation
Fear	😨😱👻	#9370DB	Anxiety, terror, nervousness
Surprise	😲🎁✨	#FF69B4	Astonishment, shock, amazement
Disgust	🤢🤮🐛	#556B2F	Revulsion, contempt, aversion
Neutral	😐📝⚖️	#A9A9A9	Balanced, factual, calm
📖 Usage Guide
Recording an Entry
Click the "🔴 Start Recording" button

Speak clearly into your microphone

Watch the waveform animate in real-time

Wait for transcription and emotion analysis

View your results with confidence scores

Viewing History
Recent entries appear in the right sidebar

Each entry shows emotion, confidence, timestamp, and text

Generating Summary
Click "✨ Generate Summary" in the sidebar

Get an AI-generated overview of your emotional day

3D Visualization
Scroll down to see the 3D Emotional Landscape

Each dot represents a voice entry

Colors correspond to emotions

Rotate, zoom, and hover for details

Exporting Data
Click "📥 Download Diary (JSON)" to save your history

JSON includes all entries with timestamps and emotion scores

🛠️ Configuration
Recording Duration
Adjust the recording length using the slider in the sidebar (3-15 seconds).

Using GPU Acceleration
If you have CUDA available, change device=-1 to device=0 in the emotion model loading:

python
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0  # Use GPU
)
Offline Mode
The app works completely offline after initial model download. No internet connection required!

📁 Project Structure
text
voice-to-emotion-diary/
│
├── advanced_emotion_diary.py    # Main application file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
│
└── exports/                      # Exported JSON files (created automatically)
    └── emotion_diary_*.json
🔧 Troubleshooting
Microphone Not Working
Windows:

Check Privacy Settings → Microphone → Allow apps to access microphone

Run sounddevice test: python -c "import sounddevice as sd; print(sd.query_devices())"

macOS:

System Preferences → Security & Privacy → Microphone → Allow Terminal/VS Code

Linux:

Install PortAudio: sudo apt-get install portaudio19-dev

Models Download Slowly
Set a Hugging Face token for faster downloads:

bash
huggingface-cli login
Get your free token at huggingface.co/settings/tokens

"No module named 'X'" Error
Make sure you've installed all requirements:

bash
pip install -r requirements.txt --upgrade
Streamlit Not Found
Install Streamlit globally or ensure your virtual environment is activated:

bash
pip install streamlit
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
OpenAI Whisper - Speech recognition

Hugging Face - Model hosting and transformers library

Streamlit - Web application framework

Plotly - Interactive visualizations

j-hartmann - Emotion classification model
