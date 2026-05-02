import streamlit as st
import requests
import io
import time
import base64
from datetime import datetime

# --- CONFIGURATION ---
API_BASE_URL = "http://localhost:8000"
PAGE_TITLE = "Customer Support Assistant"
PAGE_ICON = "🎧"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STYLING ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        
        * {
            font-family: 'Outfit', sans-serif;
        }
        
        .main {
            background: #0f172a;
            color: #f8fafc;
        }
        
        .stChatMessage {
            background: rgba(30, 41, 59, 0.7) !important;
            border-radius: 15px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            margin-bottom: 15px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        }
        
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.5rem 0;
            background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .status-online { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
        .status-offline { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
        
        /* Premium Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

# --- HELPERS ---
def get_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=1)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<div class='header-container'><h1>Customer Support Assistant</h1></div>", unsafe_allow_html=True)

health = get_api_health()
cols_header = st.columns([4, 1])
with cols_header[0]:
    # Language Switcher
    lang_choice = st.radio(
        "Preferred Language / भाषा पसंद:",
        ["English (Indian Accent)", "हिन्दी (Hindi)"],
        horizontal=True,
        index=0
    )
    lang_code = "hi" if "हिन्दी" in lang_choice else "en"

with cols_header[1]:
    if health and health.get("status") == "healthy":
        st.markdown("<div style='text-align: right;'><span class='status-badge status-online'>● SYSTEM READY</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align: right;'><span class='status-badge status-offline'>○ SYSTEM INITIALIZING...</span></div>", unsafe_allow_html=True)

# --- CHAT CONTAINER ---
chat_placeholder = st.container()

with chat_placeholder:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "audio" in message:
                st.audio(message["audio"], format="audio/mp3")
            if "timing" in message:
                st.caption(f"⏱️ Processed in {message['timing']}ms")

# --- INPUT HANDLING ---
st.markdown("---")
col1, col2 = st.columns([4, 1])

with col1:
    placeholder_text = "अपनी समस्या यहाँ लिखें..." if lang_code == "hi" else "Ask a question..."
    prompt = st.chat_input(placeholder_text)

with col2:
    audio_file = st.audio_input("Voice", label_visibility="collapsed")

# --- PROCESSING LOGIC ---
if prompt:
    # Handle Text
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_placeholder:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    with st.spinner("Thinking..." if lang_code == "en" else "सोच रहा हूँ..."):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/chat/text", 
                json={"text": prompt, "parameters": {"language": lang_code}}
            )
            if resp.status_code == 200:
                data = resp.json()
                ans_text = data["response_text"]
                
                # Get Audio
                audio_resp = requests.get(f"{API_BASE_URL}/chat/audio/{ans_text}", params={"language": lang_code}, timeout=30)
                audio_bytes = audio_resp.content if audio_resp.status_code == 200 else None
                
                msg_obj = {"role": "assistant", "content": ans_text}
                if audio_bytes:
                    msg_obj["audio"] = audio_bytes
                
                st.session_state.messages.append(msg_obj)
                st.rerun()
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

elif audio_file and audio_file != st.session_state.last_processed_audio:
    # Handle Audio
    st.session_state.last_processed_audio = audio_file
    
    with st.spinner("Processing voice..." if lang_code == "en" else "आवाज़ संसाधित कर रहा हूँ..."):
        try:
            files = {"audio": ("user_audio.wav", audio_file.getvalue(), "audio/wav")}
            resp = requests.post(f"{API_BASE_URL}/chat/audio", files=files, params={"language": lang_code})
            
            if resp.status_code == 200:
                data = resp.json()
                
                audio_bytes = base64.b64decode(data["audio_response"])
                transcript = data["transcript"]
                
                st.session_state.messages.append({"role": "user", "content": f"🎤 {transcript['user_input']}"})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": transcript["agent_response"],
                    "audio": audio_bytes,
                    "timing": data["processing_time_ms"]
                })
                
                st.rerun()
            else:
                st.error(f"Server Error: {resp.text}")
        except Exception as e:
            st.error(f"Failed to process audio: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/customer-support.png")
    st.title("Settings")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_processed_audio = None
        st.rerun()
    
    st.markdown("---")
    st.write("### Knowledge Base")
    st.caption("Now supporting Indian English & Hindi!")
    st.info("Language selection affects both the AI's understanding and its voice.")