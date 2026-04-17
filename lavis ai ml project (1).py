"""
Advanced Voice to Emotion Diary
Run with: streamlit run advanced_emotion_diary.py

Features:
- Offline speech transcription (Whisper base) - NO FFMPEG REQUIRED
- 7-class emotion detection (DistilRoBERTa)
- Live waveform visualization
- 3D emotion scatter plot
- Daily diary summary
- Persistent JSON export
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from datetime import datetime
import json
import pandas as pd
import time
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="🎙️ Advanced Voice Diary",
    page_icon="🎭",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .emotion-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 28px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    .big-emoji {
        font-size: 100px;
        text-align: center;
        filter: drop-shadow(0 0 30px rgba(255, 215, 0, 0.5));
    }
    .emotion-bar {
        height: 10px;
        border-radius: 10px;
        margin: 6px 0;
        transition: width 0.4s cubic-bezier(0.2, 0.9, 0.4, 1);
        box-shadow: 0 0 8px currentColor;
    }
    .recording-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #ff3b3b;
        margin-right: 10px;
        animation: pulse 1.2s infinite;
        box-shadow: 0 0 15px #ff3b3b;
    }
    @keyframes pulse {
        0% { opacity: 0.6; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0.6; transform: scale(0.9); }
    }
    .history-entry {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-left: 4px solid;
        backdrop-filter: blur(8px);
    }
    h1, h2, h3, h4, p, span, div {
        color: #f0f0f0 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px 28px;
        font-weight: bold;
        font-size: 18px;
        transition: all 0.3s;
        box-shadow: 0 8px 20px rgba(106, 17, 203, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(37, 117, 252, 0.5);
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ffaa, #00aaff) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ML MODELS (CACHED) ----------
@st.cache_resource
def load_whisper_model():
    """Load Whisper base model (offline ASR)"""
    with st.spinner("🔊 Loading Whisper speech recognition model... (first time may take 30s)"):
        return whisper.load_model("base")

@st.cache_resource
def load_emotion_model():
    """Load 7-emotion classifier (DistilRoBERTa fine-tuned)"""
    with st.spinner("😊 Loading emotion detection model... (first time may take 20s)"):
        return pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1
        )

# Load models
try:
    whisper_model = load_whisper_model()
    emotion_pipeline = load_emotion_model()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ---------- EMOJI & COLOR MAPPINGS ----------
EMOJI_MAP = {
    'joy': '😊🎉✨🌈🎈',
    'sadness': '😢💔🌧️🥀😞',
    'anger': '😤🤬💢🔥👿',
    'fear': '😨😱👻🌑💀',
    'surprise': '😲🎁✨🤯🌟',
    'disgust': '🤢🤮🐛💩🙅',
    'neutral': '😐📝⚖️🌫️🤔'
}

COLOR_MAP = {
    'joy': '#FFD700',
    'sadness': '#6495ED',
    'anger': '#FF4500',
    'fear': '#9370DB',
    'surprise': '#FF69B4',
    'disgust': '#556B2F',
    'neutral': '#A9A9A9'
}

# ---------- SESSION STATE ----------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'summary_text' not in st.session_state:
    st.session_state.summary_text = ""

# ---------- AUDIO RECORDING & TRANSCRIPTION (NO FFMPEG) ----------
def record_and_transcribe(duration=6, sample_rate=16000):
    """Record audio and transcribe directly - no temp files, no ffmpeg"""
    
    # Visual recording indicator
    indicator_placeholder = st.empty()
    indicator_placeholder.markdown(
        '<div style="display: flex; align-items: center; margin: 20px 0;">'
        '<span class="recording-indicator"></span>'
        '<span style="font-size: 22px; font-weight: 500;">🎤 Recording... Speak clearly</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    
    # Animate progress
    for i in range(100):
        time.sleep(duration / 100)
        progress_bar.progress(i + 1)
        if i % 20 == 0:
            remaining = duration - int((i/100)*duration)
            status_text.text(f"⏳ {remaining} seconds remaining...")
    
    sd.wait()
    audio_data = recording.flatten()
    
    # Clean up UI elements
    progress_bar.empty()
    status_text.empty()
    indicator_placeholder.empty()
    
    # Show waveform
    fig_wave = go.Figure()
    downsample = max(1, len(audio_data) // 2000)
    fig_wave.add_trace(go.Scatter(
        y=audio_data[::downsample],
        mode='lines',
        line=dict(color='#00ffaa', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 170, 0.15)',
        name='Waveform'
    ))
    fig_wave.update_layout(
        title="🎵 Audio Waveform",
        height=160,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False
    )
    st.plotly_chart(fig_wave, width='stretch')
    
    # Transcribe directly from numpy array
    with st.spinner("🔊 Transcribing with Whisper..."):
        audio_data = audio_data.astype(np.float32)
        result = whisper_model.transcribe(audio_data, fp16=False)
        text = result["text"].strip()
    
    return text, audio_data

# ---------- DAILY SUMMARY ----------
def generate_daily_summary(entries):
    """Create summary from emotion statistics"""
    if not entries:
        return "📭 No entries yet today. Record your first voice memo!"
    
    from collections import Counter
    emotions = [e['dominant_emotion'] for e in entries]
    counts = Counter(emotions)
    top_emotion, top_count = counts.most_common(1)[0]
    
    summary = f"**📊 Today's Emotional Dashboard**  \n\n"
    summary += f"You recorded **{len(entries)}** voice entries.  \n"
    summary += f"Dominant mood: **{top_emotion.upper()}** ({top_count} occurrence{'s' if top_count>1 else ''}).  \n\n"
    
    if 'joy' in counts:
        summary += f"✨ Joyful moments: {counts['joy']}  \n"
    if 'sadness' in counts:
        summary += f"🌧️ Sad moments: {counts['sadness']}  \n"
    if 'anger' in counts:
        summary += f"🔥 Angry outbursts: {counts['anger']}  \n"
    if 'fear' in counts:
        summary += f"😨 Fearful moments: {counts['fear']}  \n"
    if 'surprise' in counts:
        summary += f"🎁 Surprises: {counts['surprise']}  \n"
    
    if top_emotion == 'joy':
        summary += "\n😊 *It was a bright, positive day!*"
    elif top_emotion == 'sadness':
        summary += "\n🤗 *Tomorrow is a new beginning.*"
    elif top_emotion == 'anger':
        summary += "\n🧘 *Breathe. Let it go.*"
    elif top_emotion == 'fear':
        summary += "\n💪 *You are stronger than your fears.*"
    
    return summary

# ---------- 3D PLOT ----------
def create_3d_emotion_plot(history):
    """Generate 3D scatter plot of emotions"""
    if not history:
        fig = go.Figure()
        fig.add_annotation(
            text="🎤 Record your first entry to see the 3D emotion landscape",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#aaa")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    # 3D coordinates based on valence-arousal-dominance model
    emotion_coords = {
        'joy':       (0.85, 0.60, 0.70),
        'sadness':   (-0.80, -0.30, -0.40),
        'anger':     (-0.60, 0.80, 0.50),
        'fear':      (-0.50, 0.70, -0.60),
        'surprise':  (0.40, 0.85, 0.20),
        'disgust':   (-0.70, 0.20, 0.10),
        'neutral':   (0.00, 0.00, 0.00)
    }
    
    df = pd.DataFrame(history)
    np.random.seed(42)
    df['x'] = df['dominant_emotion'].map(lambda e: emotion_coords[e][0] + np.random.normal(0, 0.04))
    df['y'] = df['dominant_emotion'].map(lambda e: emotion_coords[e][1] + np.random.normal(0, 0.04))
    df['z'] = df['dominant_emotion'].map(lambda e: emotion_coords[e][2] + np.random.normal(0, 0.04))
    df['size'] = df['confidence'] * 20 + 5
    
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='dominant_emotion',
        size='size',
        hover_data={'text': True, 'time': True, 'confidence': ':.0%', 'dominant_emotion': True},
        color_discrete_map=COLOR_MAP,
        title="🌌 3D Emotional Landscape (Valence · Arousal · Dominance)"
    )
    
    fig.update_traces(
        marker=dict(sizemode='diameter', sizeref=0.3, line=dict(width=1, color='white'))
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='Valence (Pleasantness)',
            yaxis_title='Arousal (Intensity)',
            zaxis_title='Dominance (Control)',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            zaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

# ---------- MAIN UI ----------
st.title("🎙️ Voice to Emotion · Advanced AI Diary")
st.markdown("### Speak your mind — AI detects 7 emotions and visualizes your mood in 3D")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Controls")
    duration = st.slider("⏱️ Recording length (seconds)", 3, 15, 6, help="Longer recordings = better accuracy")
    
    st.markdown("---")
    st.header("📝 Diary Summary")
    if st.button("✨ Generate Summary", width='stretch'):
        st.session_state.summary_text = generate_daily_summary(st.session_state.history)
    
    if st.session_state.summary_text:
        st.info(st.session_state.summary_text)
    
    st.markdown("---")
    st.header("💾 Export Data")
    if st.session_state.history:
        json_data = json.dumps(st.session_state.history, indent=2, default=str)
        st.download_button(
            label="📥 Download Diary (JSON)",
            data=json_data,
            file_name=f"emotion_diary_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            width='stretch'
        )
    else:
        st.caption("Record entries to enable export")
    
    st.markdown("---")
    st.header("📊 Emotion Stats")
    if st.session_state.history:
        from collections import Counter
        counts = Counter(e['dominant_emotion'] for e in st.session_state.history)
        total = len(st.session_state.history)
        for emo, count in counts.most_common():
            percentage = (count / total) * 100
            emoji = EMOJI_MAP.get(emo, '😐')[0]
            st.markdown(f"{emoji} **{emo}**: {count} ({percentage:.0f}%)")
    else:
        st.caption("No data yet")

# ---- MAIN COLUMNS ----
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("🎤 New Voice Entry")
    
    if st.button("🔴 Start Recording", width='stretch'):
        try:
            # Record and transcribe
            text, audio_wave = record_and_transcribe(duration)
            
            if not text:
                st.error("Could not transcribe audio. Please try again.")
                st.stop()
            
            st.success(f"**📝 You said:** *{text}*")
            
            # Emotion detection
            with st.spinner("😊 Analyzing emotions..."):
                scores = emotion_pipeline(text)[0]
                dominant = max(scores, key=lambda x: x['score'])
                dominant_label = dominant['label']
                dominant_score = dominant['score']
                emotion_scores_dict = {s['label']: s['score'] for s in scores}
                
                # Save entry
                entry = {
                    'text': text,
                    'dominant_emotion': dominant_label,
                    'confidence': dominant_score,
                    'all_emotions': emotion_scores_dict,
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.history.append(entry)
                
                # Display results
                emoji_icon = EMOJI_MAP.get(dominant_label, '😐')[0]
                color = COLOR_MAP.get(dominant_label, '#A9A9A9')
                
                st.markdown(f"""
                <div class="emotion-card" style="border-top: 5px solid {color};">
                    <div class="big-emoji">{emoji_icon}</div>
                    <h2 style="color: {color}; text-align: center; margin: 10px 0;">
                        {dominant_label.upper()} · {dominant_score:.0%}
                    </h2>
                    <div style="margin-top: 25px;">
                """, unsafe_allow_html=True)
                
                # Show all emotion scores
                for emo, score in sorted(scores, key=lambda x: x['score'], reverse=True):
                    bar_color = COLOR_MAP.get(emo, '#A9A9A9')
                    emoji = EMOJI_MAP.get(emo, '😐')[0]
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin: 8px 0;">
                        <span style="width: 90px; font-weight: 500;">{emoji} {emo}</span>
                        <div style="flex: 1; background: rgba(255,255,255,0.08); border-radius: 20px; margin: 0 15px;">
                            <div class="emotion-bar" style="width: {score*100}%; background: {bar_color}; box-shadow: 0 0 10px {bar_color};"></div>
                        </div>
                        <span style="min-width: 45px; text-align: right;">{score:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error during recording: {str(e)}")
            st.info("Make sure your microphone is connected and you've granted permission.")

with col_right:
    st.subheader("📋 Recent Diary Entries")
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-6:]):
            emoji = EMOJI_MAP.get(entry['dominant_emotion'], '😐')[0]
            color = COLOR_MAP.get(entry['dominant_emotion'], '#A9A9A9')
            st.markdown(f"""
            <div class="history-entry" style="border-left-color: {color};">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 28px;">{emoji}</span>
                    <div style="flex:1;">
                        <div style="font-weight: bold;">{entry['dominant_emotion']} · {entry['confidence']:.0%}</div>
                        <small style="opacity: 0.7;">{entry['time']}</small>
                    </div>
                </div>
                <div style="margin-top: 8px; font-style: italic; opacity: 0.9;">
                    "{entry['text'][:80]}{'...' if len(entry['text'])>80 else ''}"
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🎤 Click 'Start Recording' to begin your emotional journey")

# ---- BOTTOM: 3D VISUALIZATION ----
st.markdown("---")
st.subheader("🌌 3D Emotional Landscape")
fig_3d = create_3d_emotion_plot(st.session_state.history)
st.plotly_chart(fig_3d, width='stretch')

# ---- TIMELINE CHART ----
if st.session_state.history:
    st.subheader("📈 Emotion Intensity Timeline")
    df_timeline = pd.DataFrame(st.session_state.history)
    df_timeline['datetime'] = pd.to_datetime(df_timeline['timestamp'])
    
    fig_timeline = px.scatter(
        df_timeline, x='datetime', y='confidence',
        color='dominant_emotion',
        size='confidence',
        hover_data=['text'],
        color_discrete_map=COLOR_MAP,
        title="Confidence of Dominant Emotion Over Time",
        size_max=25
    )
    fig_timeline.update_traces(marker=dict(line=dict(width=1, color='white')))
    fig_timeline.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Time"),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickformat='.0%', title="Confidence"),
        height=350
    )
    st.plotly_chart(fig_timeline, width='stretch')

# ---- FOOTER ----
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7;">
    ✨ Built with OpenAI Whisper · DistilRoBERTa Emotion Model · Plotly 3D · Offline Ready ✨
</div>
""", unsafe_allow_html=True)