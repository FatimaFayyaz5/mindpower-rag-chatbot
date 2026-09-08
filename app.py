"""
Mind Power Artists — AI Knowledge Assistant
RAG-based chatbot with modern UI
"""

import json
import os
import math
import re
import datetime
import streamlit as st
import requests

# ─── Configuration ───────────────────────────────────────────────────────────

KNOWLEDGE_FILE = "knowledge_base.json"
LOG_FILE = "unanswered_questions.json"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.15

# ─── TF-IDF Retrieval ───────────────────────────────────────────────────────

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_idf(corpus_tokens):
    n = len(corpus_tokens)
    df = {}
    for tokens in corpus_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (f + 1)) + 1 for t, f in df.items()}

def tfidf_vector(tokens, idf):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return {t: (1 + math.log(c)) * idf.get(t, 1.0) for t, c in tf.items()}

def cosine_sim(a, b):
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

class KnowledgeBase:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        self.corpus_tokens = [
            tokenize(e["question"] + " " + e["answer"]) for e in self.entries
        ]
        self.idf = build_idf(self.corpus_tokens)
        self.vectors = [tfidf_vector(t, self.idf) for t in self.corpus_tokens]

    def search(self, query, top_k=TOP_K):
        q_tokens = tokenize(query)
        q_vec = tfidf_vector(q_tokens, self.idf)
        scored = []
        for i, v in enumerate(self.vectors):
            sim = cosine_sim(q_vec, v)
            scored.append((self.entries[i], sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def add_entry(self, entry):
        self.entries.append(entry)
        tokens = tokenize(entry["question"] + " " + entry["answer"])
        self.corpus_tokens.append(tokens)
        self.idf = build_idf(self.corpus_tokens)
        self.vectors = [tfidf_vector(t, self.idf) for t in self.corpus_tokens]
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

# ─── LLM Generation ─────────────────────────────────────────────────────────

def call_groq(prompt, api_key):
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 512},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def call_gemini(prompt, api_key):
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

def generate_answer(query, context_chunks, provider, api_key):
    context = "\n\n".join(
        f"[Source: {c['category']}]\nQ: {c['question']}\nA: {c['answer']}"
        for c in context_chunks
    )
    prompt = f"""You are the helpful AI assistant for Mind Power Artists, a healing and mind sciences organization based in Islamabad, Pakistan.

Answer the user's question using ONLY the information provided below. Be warm, professional, and concise.
If the answer is not in the provided context, say: "I don't have that information right now. Please contact us at info@mindpowerartists.com or call +92 310 333 8452 for assistance."

Do NOT make up information. Do NOT answer from general knowledge.

--- KNOWLEDGE BASE ---
{context}
--- END ---

User question: {query}

Answer:"""
    try:
        if provider == "Groq":
            return call_groq(prompt, api_key)
        else:
            return call_gemini(prompt, api_key)
    except Exception as e:
        return f"Error calling {provider} API: {str(e)}"

def log_unanswered(question):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    logs.append({"question": question, "timestamp": datetime.datetime.now().isoformat()})
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

# ─── Streamlit Config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mind Power Artists — AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Full Custom CSS ─────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #f8f6ff;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0533 0%, #2d1154 40%, #3d1a6e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] label {
    color: #e0d4f5 !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #c4b5d9 !important;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08);
}

/* Sidebar buttons (sample questions) */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.06);
    color: #e0d4f5;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 400;
    text-align: left;
    width: 100%;
    transition: all 0.2s ease;
    margin-bottom: 2px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(155, 77, 202, 0.3);
    border-color: rgba(155, 77, 202, 0.5);
    color: #fff;
    transform: translateX(3px);
}

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #6B2D8B 0%, #8B3DC7 30%, #A855F7 60%, #C084FC 100%);
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(107, 45, 139, 0.25);
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-content { position: relative; z-index: 1; }

.hero h1 {
    color: #fff;
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}

.hero p {
    color: rgba(255,255,255,0.85);
    font-size: 0.92rem;
    margin: 0;
    font-weight: 400;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    font-size: 0.72rem;
    color: #fff;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    border: 1px solid rgba(255,255,255,0.2);
}

/* ── Chat Messages ── */
.msg-user {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin: 1rem 0;
    animation: slideIn 0.3s ease;
}

.msg-bot {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin: 1rem 0;
    animation: slideIn 0.4s ease;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.avatar {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    font-weight: 700;
}

.avatar-user {
    background: linear-gradient(135deg, #6B2D8B, #A855F7);
    color: #fff;
}

.avatar-bot {
    background: linear-gradient(135deg, #059669, #34D399);
    color: #fff;
}

.bubble {
    padding: 0.9rem 1.15rem;
    border-radius: 16px;
    max-width: 85%;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #1a1a2e;
}

.bubble-user {
    background: #ede5f7;
    border: 1px solid #d8c8ed;
    border-bottom-left-radius: 4px;
}

.bubble-bot {
    background: #ffffff;
    border: 1px solid #e8e8ee;
    border-bottom-left-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ── Source Tags ── */
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
    margin-left: 50px;
}

.src-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.src-high {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}

.src-mid {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}

.src-low {
    background: #fde2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #fff;
    border-radius: 14px;
    padding: 4px;
    border: 1px solid #e8e8ee;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
    background: transparent;
    border: none;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6B2D8B, #9B4DCA) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(107, 45, 139, 0.3);
}

.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Info Card ── */
.info-card {
    background: #fff;
    border: 1px solid #e8e8ee;
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin: 0.8rem 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}

.info-card h4 {
    margin: 0 0 0.3rem 0;
    font-size: 0.95rem;
    font-weight: 700;
    color: #1a1a2e;
}

.info-card p {
    margin: 0;
    font-size: 0.85rem;
    color: #64748b;
    line-height: 1.5;
}

/* ── Stats Row ── */
.stats-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
}

.stat-card {
    flex: 1;
    background: #fff;
    border: 1px solid #e8e8ee;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

.stat-num {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6B2D8B, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* ── Welcome Card ── */
.welcome-card {
    background: #fff;
    border: 1px solid #e8e8ee;
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

.welcome-icon {
    font-size: 3rem;
    margin-bottom: 0.8rem;
}

.welcome-card h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a1a2e;
}

.welcome-card p {
    color: #64748b;
    font-size: 0.88rem;
    margin: 0 auto;
    max-width: 500px;
    line-height: 1.6;
}

/* Quick-action chips */
.quick-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 1.2rem;
}

.qa-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 0.45rem 1rem;
    background: #f3e8ff;
    border: 1px solid #e0cffc;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #6B2D8B;
    cursor: default;
}

/* ── No-answer info bar ── */
.no-match-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #fef3c7;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.5rem 0 0.5rem 50px;
    font-size: 0.8rem;
    color: #92400e;
}

/* ── Knowledge base browser ── */
.kb-entry {
    background: #fff;
    border: 1px solid #e8e8ee;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    transition: border-color 0.2s;
}

.kb-entry:hover {
    border-color: #c4b5d9;
}

.kb-cat {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    background: #f3e8ff;
    color: #6B2D8B;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}

.kb-q {
    font-weight: 600;
    font-size: 0.88rem;
    color: #1a1a2e;
    margin: 0.2rem 0 0.3rem 0;
}

.kb-a {
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.55;
}

/* ── Form styling ── */
.stForm {
    background: #fff;
    border: 1px solid #e8e8ee;
    border-radius: 16px;
    padding: 1.5rem;
}

/* ── Main-area buttons ── */
.stMainBlockContainer .stButton > button {
    background: linear-gradient(135deg, #6B2D8B, #9B4DCA);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(107,45,139,0.2);
}

.stMainBlockContainer .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(107,45,139,0.35);
}

/* ── Metric override ── */
div[data-testid="stMetric"] {
    background: transparent;
}

/* ── Log entry ── */
.log-entry {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.7rem 1rem;
    background: #fff;
    border: 1px solid #e8e8ee;
    border-radius: 10px;
    margin: 0.4rem 0;
    font-size: 0.85rem;
}

.log-icon {
    width: 28px;
    height: 28px;
    background: #fef3c7;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    flex-shrink: 0;
}

.log-text {
    color: #1a1a2e;
    font-weight: 500;
    flex: 1;
}

.log-time {
    color: #94a3b8;
    font-size: 0.72rem;
    font-weight: 400;
}

</style>
""", unsafe_allow_html=True)

# ─── Hero Header ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-badge">⚡ RAG-Powered AI</div>
        <h1>Mind Power Artists — AI Assistant</h1>
        <p>Ask anything about our services, courses, booking, payments, and more</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 MPA Assistant")
    st.markdown("---")

    st.markdown("##### ⚙️ LLM Provider")
    provider = st.selectbox("Provider", ["Groq", "Gemini"], index=0, label_visibility="collapsed")

    if provider == "Groq":
        api_key = st.text_input("Groq API Key", type="password", help="Free at console.groq.com", label_visibility="collapsed", placeholder="Paste Groq API key...")
    else:
        api_key = st.text_input("Gemini API Key", type="password", help="Free at aistudio.google.com", label_visibility="collapsed", placeholder="Paste Gemini API key...")

    if not api_key:
        st.markdown("<p style='font-size:0.75rem; color:#fbbf24; margin-top:-0.5rem;'>⚠️ No key — using direct lookup mode</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='font-size:0.75rem; color:#34d399; margin-top:-0.5rem;'>✓ Connected to {provider}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 💬 Try asking")

    sample_questions = [
        "What services do you offer?",
        "How can I book an appointment?",
        "What is Aura Cleansing?",
        "What are the payment options?",
        "Do you offer corporate training?",
        "What courses are available?",
        "How can I cancel a session?",
        "What AI automation services do you have?",
    ]
    for q in sample_questions:
        if st.button(q, key=f"s_{q}", use_container_width=True):
            st.session_state["prefill"] = q

    st.markdown("---")
    st.markdown("##### 📊 Knowledge Base")
    kb_sidebar = KnowledgeBase(KNOWLEDGE_FILE)
    categories = sorted(set(e["category"] for e in kb_sidebar.entries))
    st.markdown(f"""
    <div style='display:flex; gap:8px; flex-wrap:wrap;'>
        <div style='background:rgba(255,255,255,0.08); border-radius:10px; padding:0.6rem 0.9rem; flex:1; min-width:80px; text-align:center;'>
            <div style='font-size:1.3rem; font-weight:800; color:#A855F7;'>{len(kb_sidebar.entries)}</div>
            <div style='font-size:0.65rem; color:#94a3b8; text-transform:uppercase; font-weight:600;'>Entries</div>
        </div>
        <div style='background:rgba(255,255,255,0.08); border-radius:10px; padding:0.6rem 0.9rem; flex:1; min-width:80px; text-align:center;'>
            <div style='font-size:1.3rem; font-weight:800; color:#34D399;'>{len(categories)}</div>
            <div style='font-size:0.65rem; color:#94a3b8; text-transform:uppercase; font-weight:600;'>Categories</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_chat, tab_admin, tab_log = st.tabs(["💬  Chat", "➕  Admin Panel", "📋  Unanswered Log"])

# ─── Chat Tab ────────────────────────────────────────────────────────────────

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome state
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🧠</div>
            <h3>Welcome to Mind Power Artists AI</h3>
            <p>I'm your intelligent assistant, powered by RAG (Retrieval-Augmented Generation). I answer from our verified knowledge base — no hallucinations, no guesswork. Ask me anything about our services.</p>
            <div class="quick-actions">
                <span class="qa-chip">🧘 Healing Services</span>
                <span class="qa-chip">📚 Courses</span>
                <span class="qa-chip">💳 Payments</span>
                <span class="qa-chip">📅 Booking</span>
                <span class="qa-chip">🤖 AI Services</span>
                <span class="qa-chip">❌ Cancellation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="avatar avatar-user">F</div>
                <div class="bubble bubble-user">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Convert markdown-style formatting to HTML
            content = msg["content"].replace("\n", "<br>")
            st.markdown(f"""
            <div class="msg-bot">
                <div class="avatar avatar-bot">✦</div>
                <div class="bubble bubble-bot">{content}</div>
            </div>
            """, unsafe_allow_html=True)
            # Source chips
            if msg.get("sources"):
                chips_html = ""
                for src in msg["sources"]:
                    score = src["score"]
                    cls = "src-high" if score > 0.3 else ("src-mid" if score > 0.2 else "src-low")
                    pct = f"{score:.0%}"
                    chips_html += f'<span class="src-chip {cls}">📎 {src["category"]}  ·  {pct}</span>'
                st.markdown(f'<div class="sources-row">{chips_html}</div>', unsafe_allow_html=True)

    # Input
    prefill = st.session_state.pop("prefill", "")
    user_input = st.chat_input("Ask about Mind Power Artists...")

    if prefill:
        user_input = prefill

    if user_input:
        # User message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f"""
        <div class="msg-user">
            <div class="avatar avatar-user">F</div>
            <div class="bubble bubble-user">{user_input}</div>
        </div>
        """, unsafe_allow_html=True)

        # Retrieve
        kb = KnowledgeBase(KNOWLEDGE_FILE)
        results = kb.search(user_input)
        best_score = results[0][1] if results else 0
        relevant = [(entry, score) for entry, score in results if score > SIMILARITY_THRESHOLD]

        if not api_key:
            if relevant:
                answer = relevant[0][0]["answer"]
                if len(relevant) > 1 and relevant[1][1] > 0.25:
                    answer += "\n\n" + relevant[1][0]["answer"]
            else:
                answer = "I don't have that information right now. Please contact us at info@mindpowerartists.com or call +92 310 333 8452 for assistance."
                log_unanswered(user_input)
        else:
            if relevant:
                context_entries = [entry for entry, _ in relevant]
                with st.spinner("Retrieving and generating..."):
                    answer = generate_answer(user_input, context_entries, provider, api_key)
            else:
                answer = "I don't have that information in our knowledge base right now. Please contact us at info@mindpowerartists.com or call +92 310 333 8452 for assistance."
                log_unanswered(user_input)

        # Bot message
        sources = [{"category": e["category"], "score": s} for e, s in relevant]
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        content_html = answer.replace("\n", "<br>")
        st.markdown(f"""
        <div class="msg-bot">
            <div class="avatar avatar-bot">✦</div>
            <div class="bubble bubble-bot">{content_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if sources:
            chips_html = ""
            for src in sources:
                score = src["score"]
                cls = "src-high" if score > 0.3 else ("src-mid" if score > 0.2 else "src-low")
                pct = f"{score:.0%}"
                chips_html += f'<span class="src-chip {cls}">📎 {src["category"]}  ·  {pct}</span>'
            st.markdown(f'<div class="sources-row">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-match-bar">
                <span>⚠️</span>
                <span>No matching entries found — this question has been logged for review.</span>
            </div>
            """, unsafe_allow_html=True)

# ─── Admin Tab ───────────────────────────────────────────────────────────────

with tab_admin:
    col_form, col_browse = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("""
        <div class="info-card">
            <h4>➕ Add New Knowledge</h4>
            <p>Add new FAQs, service descriptions, or any information. The chatbot will instantly use it.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("add_entry", clear_on_submit=True):
            new_category = st.selectbox("Category", ["About", "Services", "Courses", "Digital Services", "Payment", "Booking", "Cancellation", "General", "Corporate Training", "Social Media"])
            new_question = st.text_input("Question / Topic")
            new_answer = st.text_area("Answer / Information", height=120)
            submitted = st.form_submit_button("Add to Knowledge Base", use_container_width=True)

            if submitted and new_question and new_answer:
                kb = KnowledgeBase(KNOWLEDGE_FILE)
                new_id = f"custom_{len(kb.entries) + 1}"
                kb.add_entry({"id": new_id, "category": new_category, "question": new_question, "answer": new_answer})
                st.success(f"Added! Knowledge base now has {len(kb.entries)} entries.")
                st.balloons()

    with col_browse:
        st.markdown("""
        <div class="info-card">
            <h4>📂 Browse Knowledge Base</h4>
            <p>View all entries currently in the chatbot's knowledge.</p>
        </div>
        """, unsafe_allow_html=True)

        kb = KnowledgeBase(KNOWLEDGE_FILE)
        filter_cat = st.selectbox("Filter by category", ["All"] + sorted(set(e["category"] for e in kb.entries)), key="admin_filter")

        for entry in kb.entries:
            if filter_cat != "All" and entry["category"] != filter_cat:
                continue
            st.markdown(f"""
            <div class="kb-entry">
                <span class="kb-cat">{entry['category']}</span>
                <div class="kb-q">{entry['question']}</div>
                <div class="kb-a">{entry['answer'][:150]}{'...' if len(entry['answer']) > 150 else ''}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Log Tab ─────────────────────────────────────────────────────────────────

with tab_log:
    st.markdown("""
    <div class="info-card">
        <h4>📋 Unanswered Questions</h4>
        <p>Questions the chatbot couldn't answer confidently. Review these to improve the knowledge base — add answers in the Admin panel.</p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
        if logs:
            # Stats
            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-num">{len(logs)}</div>
                    <div class="stat-label">Unanswered</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{len(set(l['question'] for l in logs))}</div>
                    <div class="stat-label">Unique</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for log_entry in reversed(logs):
                ts = log_entry.get("timestamp", "")
                short_time = ts[:16].replace("T", " ") if ts else ""
                st.markdown(f"""
                <div class="log-entry">
                    <div class="log-icon">❓</div>
                    <span class="log-text">{log_entry['question']}</span>
                    <span class="log-time">{short_time}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            if st.button("🗑️  Clear All Logs"):
                os.remove(LOG_FILE)
                st.rerun()
        else:
            st.info("No unanswered questions yet — the chatbot is covering everything!")
    else:
        st.info("No unanswered questions yet — the chatbot is covering everything!")

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem 0; color:#94a3b8; font-size:0.78rem;'>
    Mind Power Artists AI Assistant · RAG Pipeline: TF-IDF Retrieval → LLM Generation · 
    <span style='color:#A855F7; font-weight:600;'>mindpowerartists.com</span>
</div>
""", unsafe_allow_html=True)
