<div align="center">

![Zangoh Audio Support Agent](docs/assets/banner.png)

# 🎙️ Zangoh GenAI Audio Support Agent
### *The Future of Intelligent, Multilingual Customer Interaction*

[![Maintained](https://img.shields.io/badge/Maintained%3F-yes-6366f1.svg?style=for-the-badge)](https://github.com/Ifrah27/Zangoh_GenAI/graphs/commit-activity)
[![Version](https://img.shields.io/badge/version-1.0.0-a855f7.svg?style=for-the-badge)](https://github.com/Ifrah27/Zangoh_GenAI/releases)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-10b981.svg?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b.svg?style=for-the-badge)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-3b82f6.svg?style=for-the-badge)](LICENSE)

---

<p align="center">
  <b>A state-of-the-art conversational AI pipeline orchestrating Speech-to-Text (STT), Retrieval-Augmented Generation (RAG), and Text-to-Speech (TTS) to deliver a premium, low-latency customer support experience.</b>
</p>

[Explore Docs](docs/RAG_IMPLEMENTATION_GUIDE.md) • [View Demo](#-demo-walkthrough) • [Report Bug](https://github.com/Ifrah27/Zangoh_GenAI/issues) • [Request Feature](https://github.com/Ifrah27/Zangoh_GenAI/issues)

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=header" width="100%"/>

## 💎 Features

<div align="center">

| 🗣️ Voice-First | 🧠 RAG Intelligence | 🌍 Multilingual (Hindi/English) |
| :---: | :---: | :---: |
| Real-time STT/TTS pipeline with Indian accent support | Deep contextual understanding powered by ChromaDB | Seamless switching between English and Hindi |
| <img src="https://img.icons8.com/fluency/48/000000/microphone.png"/> | <img src="https://img.icons8.com/fluency/48/000000/brain.png"/> | <img src="https://img.icons8.com/fluency/48/000000/translate.png"/> |

</div>

- **Dynamic RAG Pipeline**: Injects official company policy directly into LLM prompts for zero-hallucination support.
- **Hindi Localization**: Special handling for Devanagari script and MadhurNeural voice synthesis.
- **Low Latency**: Optimized asynchronous execution for near-instant responses.
- **Premium UI**: Dark-mode optimized Streamlit interface with real-time waveform visualization.

---

## 🏗 System Architecture

### 🌐 High-Level Data Flow
```mermaid
graph TD
    User([User]) -->|Voice/Text| Frontend[Streamlit Dashboard]
    Frontend -->|REST API| Backend[FastAPI Gateway]
    
    subgraph "Intelligent Pipeline"
        Backend --> STT[STT Engine: Whisper/Edge]
        STT --> Agent[LLM Agent: Groq/Gemini]
        Agent --> RAG[(ChromaDB Vector Store)]
        Agent --> TTS[TTS Engine: Edge-TTS]
    end
    
    TTS -->|Audio Stream| Backend
    Backend -->|Response| Frontend
    Frontend -->|Speech Playback| User
```

### 🛰 C4 Container Diagram
```mermaid
C4Container
    title Container diagram for Zangoh Audio Agent
    
    Person(customer, "Customer", "A user seeking support via voice or text.")
    System_Boundary(c1, "Audio Support System") {
        Container(web_app, "Streamlit UI", "Python/Streamlit", "Provides the interactive chat and audio recording interface.")
        Container(api, "FastAPI Server", "Python/FastAPI", "Orchestrates the conversion pipeline and handles session state.")
        ContainerDb(vector_db, "ChromaDB", "SQLite/Vector Store", "Stores company knowledge base embeddings.")
        Container(llm_service, "LLM Service", "Groq/Google Gemini", "Processes queries and generates responses using RAG context.")
    }
    
    Rel(customer, web_app, "Uses", "HTTPS")
    Rel(web_app, api, "API Requests", "JSON/Multipart")
    Rel(api, vector_db, "Query Context", "Vector Search")
    Rel(api, llm_service, "Generate Response", "HTTPS/API")
```

### 🔄 API Lifecycle & Sequence
```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant F as Frontend (Streamlit)
    participant B as Backend (FastAPI)
    participant S as STT (Speech-to-Text)
    participant A as AI Agent (LLM + RAG)
    participant T as TTS (Text-to-Speech)

    U->>F: Speak "How do I return items?"
    F->>B: POST /chat/audio (Binary Data)
    B->>S: transcribe(audio)
    S-->>B: "How do I return items?"
    B->>A: process_query(text)
    A->>A: Vector Search (ChromaDB)
    A-->>B: "You can return items within 30 days..."
    B->>T: synthesize(agent_response)
    T-->>B: audio_payload.mp3
    B-->>F: {audio, transcript, timing}
    F->>U: Play Audio + Display Text
```

---

## 📂 Project Structure

```text
e:/audio_support_agent/
├── 📂 data/                # Vector database persistence
├── 📂 docs/                # Technical documentation & assets
├── 📂 src/
│   ├── 📂 api/             # FastAPI endpoints & middleware
│   ├── 📂 llm/             # RAG logic & LLM Agent wrappers
│   ├── 📂 stt/             # Speech-to-Text implementations
│   ├── 📂 tts/             # Text-to-Speech engines
│   └── 📜 pipeline.py      # Core orchestration logic
├── 📜 streamlit_app.py     # Frontend application
├── 📜 .env                 # Environment secrets
└── 📜 requirements.txt     # Dependency manifest
```

---

## 🛠 Technology Stack

- **Core**: Python 3.10+
- **API**: FastAPI, Uvicorn
- **Frontend**: Streamlit, Custom CSS
- **AI Models**: 
  - **LLM**: Groq (Llama 3), Google Gemini
  - **STT**: OpenAI Whisper, SpeechRecognition
  - **TTS**: Microsoft Edge TTS, gTTS
- **Vector DB**: ChromaDB

---

## 🚦 Roadmap & Lifecycle

```mermaid
gantt
    title Development Roadmap 2026
    dateFormat  YYYY-MM-DD
    section Phase 1: Core
    STT/TTS Integration       :done,    des1, 2026-04-01, 2026-04-15
    FastAPI Gateway           :done,    des2, 2026-04-15, 2026-04-30
    section Phase 2: Intelligence
    ChromaDB RAG Layer        :active,  des3, 2026-05-01, 2026-05-15
    Hindi Localization        :active,  des4, 2026-05-10, 2026-05-20
    section Phase 3: Scale
    Multi-tenant Support      :         des5, 2026-06-01, 30d
    Deployment Optimization    :         des6, after des5, 20d
```

---

## 🚀 Deployment & CI/CD
```mermaid
graph LR
    Dev[Developer] -->|Push| GH[GitHub Repo]
    GH -->|Trigger| GHA[GitHub Actions]
    subgraph "CI/CD Pipeline"
        GHA --> Lint[Linting/Formatting]
        Lint --> Test[Unit Tests]
        Test --> Build[Docker Build]
    end
    Build -->|Deploy| Vercel[Vercel/Streamlit Cloud]
    Build -->|Deploy| AWS[AWS/EC2]
```

---

## 🔒 Security & Auth Flow
```mermaid
graph TD
    Client[Client] -->|API Key| Gateway[API Gateway]
    Gateway -->|Verify| Auth[Auth Service]
    Auth -->|Success| Pipeline[Audio Pipeline]
    Auth -->|Failure| Error[401 Unauthorized]
```

---

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
  <img src="https://img.icons8.com/fluency/48/000000/github.png"/>
  <br>
  Built with ❤️ by <b>Ifrah</b>
</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" width="100%"/>