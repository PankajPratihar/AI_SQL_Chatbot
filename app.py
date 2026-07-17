"""
Conversational SQL Chatbot — Modern UI Edition
================================================
Stack:
  - LangChain 0.3+ / langchain-core
  - LLM        : Groq (llama-3.3-70b-versatile, etc.)
  - Database   : SQLite via SQLAlchemy / langchain_community.SQLDatabase
  - UI         : Streamlit (custom themed)

Ask questions in plain English about a SQL database — the app generates
SQL, executes it, and explains the result in natural language.
"""

import os
import sqlite3
import tempfile
import time

import streamlit as st
import pandas as pd

# ── LangChain core ──────────────────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SQL Chatbot",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# THEME / CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #ecfeff 0%, #f8fafc 35%, #f1f5f9 100%);
    }

    #MainMenu, footer {visibility: hidden;}

    /* ── Hero header ─────────────────────────────────────────────────── */
    .hero {
        background: linear-gradient(120deg, #0891b2 0%, #0284c7 50%, #4f46e5 100%);
        border-radius: 20px;
        padding: 28px 34px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(8, 145, 178, 0.45);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -40%; right: -8%;
        width: 260px; height: 260px;
        background: rgba(255,255,255,0.12);
        border-radius: 50%;
    }
    .hero h1 {
        color: white;
        font-weight: 800;
        font-size: 1.9rem;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.88);
        font-size: 0.98rem;
        margin: 0;
    }

    /* ── Pills ─────────────────────────────────────────────────────────── */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .pill-green { background: #dcfce7; color: #166534; }
    .pill-amber { background: #fef3c7; color: #92400e; }
    .pill-blue  { background: #dbeafe; color: #1d4ed8; }
    .pill-red   { background: #fee2e2; color: #991b1b; }

    /* ── Cards ─────────────────────────────────────────────────────────── */
    .card {
        background: white;
        border-radius: 16px;
        padding: 20px 22px;
        border: 1px solid #eef0f4;
        box-shadow: 0 2px 10px -4px rgba(15, 23, 42, 0.06);
        margin-bottom: 16px;
    }

    /* ── Sidebar ───────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #082f49 0%, #1e3a8a 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0f2fe !important; }
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] textarea {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: #f8fafc !important;
        border-radius: 10px;
    }
    section[data-testid="stSidebar"] .stButton button {
        background: linear-gradient(120deg, #06b6d4, #0891b2);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: transform 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 14px -4px rgba(6, 182, 212, 0.6);
    }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }
    .sidebar-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #7dd3fc !important;
        margin-top: 14px;
        margin-bottom: 6px;
    }

    /* ── Chat bubbles ─────────────────────────────────────────────────── */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 10px 14px !important;
        margin-bottom: 10px;
        border: 1px solid #eef0f4;
        background: #ffffff !important;
    }
    .stChatMessage:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: #ecfeff !important;
    }
    .stChatMessage:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background: #eef2ff !important;
    }
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] li {
        font-size: 0.96rem;
        line-height: 1.55;
        color: #1e293b !important;
    }

    /* ── SQL code block styling ───────────────────────────────────────── */
    .sql-box {
        background: #0f172a;
        color: #7dd3fc;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        padding: 12px 16px;
        border-radius: 10px;
        white-space: pre-wrap;
        margin: 6px 0 10px 0;
    }

    /* ── Metric chips row ─────────────────────────────────────────────── */
    .metric-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
    .metric-chip {
        flex: 1;
        min-width: 140px;
        background: white;
        border: 1px solid #eef0f4;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px -4px rgba(15,23,42,0.05);
    }
    .metric-chip .label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        font-weight: 700;
    }
    .metric-chip .value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1e293b;
        margin-top: 2px;
    }

    /* ── File uploader ────────────────────────────────────────────────── */
    [data-testid="stFileUploaderDropzone"] {
        background: white;
        border: 2px dashed #a5f3fc;
        border-radius: 16px;
    }

    /* ── Table chips (schema view) ────────────────────────────────────── */
    .table-chip {
        display: inline-block;
        background: #f1f5f9;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 3px 10px;
        margin: 3px 4px 3px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }

    /* ── Main-area text colour override ──────────────────────────────────
       On dark theme, Streamlit's default headings/labels/expander text
       render white — invisible against our white cards. Force dark text
       everywhere in the main content area (sidebar keeps its own rules
       above, so this only targets the app view container). */
    section[data-testid="stAppViewContainer"] .block-container h1,
    section[data-testid="stAppViewContainer"] .block-container h2,
    section[data-testid="stAppViewContainer"] .block-container h3,
    section[data-testid="stAppViewContainer"] .block-container h4,
    section[data-testid="stAppViewContainer"] .block-container h5,
    section[data-testid="stAppViewContainer"] .block-container p,
    section[data-testid="stAppViewContainer"] .block-container span,
    section[data-testid="stAppViewContainer"] .block-container label,
    section[data-testid="stAppViewContainer"] .block-container li {
        color: #1e293b !important;   /* <-- change this hex for main-area text colour */
    }
    /* Expander header/summary text */
    section[data-testid="stAppViewContainer"] div[data-testid="stExpander"] summary,
    section[data-testid="stAppViewContainer"] div[data-testid="stExpander"] summary p,
    section[data-testid="stAppViewContainer"] div[data-testid="stExpander"] * {
        color: #1e293b !important;
    }
    /* File uploader helper text / button */
    section[data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] * {
        color: #1e293b !important;
    }
    /* Radio button labels */
    section[data-testid="stAppViewContainer"] .stRadio label p {
        color: #1e293b !important;
    }
    /* Re-assert pill/chip/sql-box text so the rule above never dims them */
    .pill, .table-chip, .sql-box, .metric-chip .label, .metric-chip .value,
    .hero h1, .hero p {
        color: inherit;
    }
    .pill-green { color: #166534 !important; }
    .pill-amber { color: #92400e !important; }
    .pill-blue  { color: #1d4ed8 !important; }
    .pill-red   { color: #991b1b !important; }
    .table-chip { color: #0f172a !important; }
    .sql-box { color: #7dd3fc !important; }
    .metric-chip .label { color: #94a3b8 !important; }
    .metric-chip .value { color: #1e293b !important; }
    .hero h1, .hero p { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>🗄️ Conversational SQL Chatbot</h1>
    <p>Ask questions in plain English — the bot writes the SQL, runs it, and explains the result.</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SAMPLE DATABASE  (auto-created so the app is usable out of the box)
# ════════════════════════════════════════════════════════════════════════════

SAMPLE_DB_PATH = os.path.join(tempfile.gettempdir(), "sql_chat_sample_company.db")


def ensure_sample_db(path: str) -> None:
    """Create the demo company.db (employees + sales) if it doesn't exist yet."""
    if os.path.exists(path):
        return

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS employees(
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales(
            sale_id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            amount INTEGER,
            sale_date TEXT
        )
    """)

    employees = [
        (1, "Alice", "HR", 50000),
        (2, "Bob", "IT", 75000),
        (3, "Charlie", "Sales", 65000),
        (4, "David", "IT", 80000),
        (5, "Eva", "HR", 52000),
        (6, "Frank", "Marketing", 60000),
        (7, "Grace", "Sales", 70000),
        (8, "Henry", "Finance", 90000),
        (9, "Ivy", "Finance", 85000),
        (10, "Jack", "Marketing", 62000),
    ]
    cur.executemany("INSERT OR REPLACE INTO employees VALUES (?,?,?,?)", employees)

    sales = [
        (1, 1, 12000, "2025-01-10"),
        (2, 2, 15000, "2025-01-11"),
        (3, 3, 22000, "2025-01-12"),
        (4, 4, 18000, "2025-01-13"),
        (5, 5, 17000, "2025-01-14"),
        (6, 6, 25000, "2025-01-15"),
        (7, 7, 27000, "2025-01-16"),
        (8, 8, 32000, "2025-01-17"),
        (9, 9, 21000, "2025-01-18"),
        (10, 10, 19000, "2025-01-19"),
    ]
    cur.executemany("INSERT OR REPLACE INTO sales VALUES (?,?,?,?)", sales)

    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — keys, model, database source
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at https://console.groq.com",
    )

    st.markdown('<div class="sidebar-title">Model</div>', unsafe_allow_html=True)
    groq_model = st.selectbox(
        "Groq LLM",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-title">Database source</div>', unsafe_allow_html=True)
    db_source = st.radio(
        "Database source",
        options=["Sample company.db", "Upload a .db / .sqlite file", "Custom connection URI"],
        label_visibility="collapsed",
    )

    uploaded_db = None
    custom_uri = ""
    if db_source == "Upload a .db / .sqlite file":
        uploaded_db = st.file_uploader(
            "Upload SQLite file",
            type=["db", "sqlite", "sqlite3"],
            label_visibility="collapsed",
        )
    elif db_source == "Custom connection URI":
        custom_uri = st.text_input(
            "SQLAlchemy URI",
            type="password",
            placeholder="postgresql+psycopg2://user:pass@host/dbname",
            help="Never share real credentials in shared/public apps.",
        )

    st.markdown('<div class="sidebar-title">Session</div>', unsafe_allow_html=True)
    session_id = st.text_input("Session ID", value="session_1", label_visibility="collapsed")

    if st.button("🗑️  Clear Chat History"):
        if session_id in st.session_state.get("store", {}):
            st.session_state.store[session_id] = []
        st.success("Chat history cleared!")

    st.markdown("---")
    st.markdown(
        "<small>Built with LangChain · Groq · SQLAlchemy · Streamlit</small>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# GUARD: API key required
# ════════════════════════════════════════════════════════════════════════════

if not groq_api_key:
    st.markdown("""
    <div class="card">
        <span class="pill pill-amber">⚠️ Setup required</span>
        <p style="margin-top:10px; color:#475569;">
        Enter your <b>Groq API key</b> in the sidebar to get started. It's free at
        <a href="https://console.groq.com" target="_blank">console.groq.com</a>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Session state init ───────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = {}

if "db_uri" not in st.session_state:
    st.session_state.db_uri = None

if "db_label" not in st.session_state:
    st.session_state.db_label = None

# ════════════════════════════════════════════════════════════════════════════
# RESOLVE DATABASE CONNECTION
# ════════════════════════════════════════════════════════════════════════════

db = None
db_error = None

try:
    if db_source == "Sample company.db":
        ensure_sample_db(SAMPLE_DB_PATH)
        db_uri = f"sqlite:///{SAMPLE_DB_PATH}"
        db_label = "company.db (sample)"
        db = SQLDatabase.from_uri(db_uri)

    elif db_source == "Upload a .db / .sqlite file":
        if uploaded_db is None:
            db_error = "Upload a SQLite file in the sidebar to continue."
        else:
            tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{uploaded_db.name}")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_db.getvalue())
            db_uri = f"sqlite:///{tmp_path}"
            db_label = uploaded_db.name
            db = SQLDatabase.from_uri(db_uri)

    else:  # Custom connection URI
        if not custom_uri:
            db_error = "Enter a SQLAlchemy connection URI in the sidebar to continue."
        else:
            db_label = "Custom database"
            db = SQLDatabase.from_uri(custom_uri)

except Exception as e:
    db_error = f"Could not connect to the database: {e}"

if db_error:
    st.markdown(f"""
    <div class="card">
        <span class="pill pill-red">⚠️ Database not connected</span>
        <p style="margin-top:10px; color:#475569;">{db_error}</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.session_state.db_label = db_label

# ════════════════════════════════════════════════════════════════════════════
# SCHEMA CARD
# ════════════════════════════════════════════════════════════════════════════

table_names = db.get_usable_table_names()
schema_text = db.get_table_info()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(f"#### 🗂️ Connected: {db_label}")
chips = "".join(f'<span class="table-chip">{t}</span>' for t in table_names)
st.markdown(chips, unsafe_allow_html=True)
with st.expander("View full schema"):
    st.markdown(f'<div class="sql-box">{schema_text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# BUILD NL → SQL CHAIN  (LCEL)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_chains(_db, groq_api_key: str, groq_model: str):
    """Returns (sql_chain, answer_chain). Cached per (key, model) combo."""

    llm = ChatGroq(model=groq_model, groq_api_key=groq_api_key, temperature=0)

    sql_prompt = ChatPromptTemplate.from_template("""
You are an expert SQL assistant.

Database Schema:
{schema}

Generate ONLY a valid SQL query for the schema above.

Rules:
- Return ONLY raw SQL.
- Do NOT use Markdown.
- Do NOT use ```sql.
- Do NOT explain anything.
- Output only the SQL statement.

Question:
{question}
""")

    sql_chain = sql_prompt | llm | StrOutputParser()

    answer_prompt = ChatPromptTemplate.from_template("""
Question:
{question}

SQL Query:
{sql}

SQL Result:
{result}

Answer naturally and concisely, as if speaking to a non-technical user.
""")

    answer_chain = answer_prompt | llm | StrOutputParser()

    return sql_chain, answer_chain


def clean_sql(raw_sql: str) -> str:
    """Strip stray markdown fences the LLM sometimes adds despite instructions."""
    text = raw_sql.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("sql"):
            text = text[3:]
    return text.strip()


def ask_database(question: str, sql_chain, answer_chain) -> dict:
    """Runs the full NL -> SQL -> execute -> NL answer pipeline."""
    sql = clean_sql(sql_chain.invoke({"schema": schema_text, "question": question}))

    try:
        result = db.run(sql)
        error = None
    except Exception as e:
        result = None
        error = str(e)

    if error:
        answer = f"I generated a query but it failed to run: {error}"
    else:
        answer = answer_chain.invoke({"question": question, "sql": sql, "result": result})

    return {"sql": sql, "result": result, "answer": answer, "error": error}


sql_chain, answer_chain = build_chains(db, groq_api_key, groq_model)

# ════════════════════════════════════════════════════════════════════════════
# CHAT UI
# ════════════════════════════════════════════════════════════════════════════

def get_history() -> list:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = []
    return st.session_state.store[session_id]


history = get_history()

st.markdown(f"""
<div class="metric-row">
    <div class="metric-chip">
        <div class="label">Database</div>
        <div class="value" style="font-size:0.95rem;">{db_label}</div>
    </div>
    <div class="metric-chip">
        <div class="label">Tables</div>
        <div class="value">{len(table_names)}</div>
    </div>
    <div class="metric-chip">
        <div class="label">Model</div>
        <div class="value" style="font-size:0.95rem;">{groq_model}</div>
    </div>
    <div class="metric-chip">
        <div class="label">Turns</div>
        <div class="value">{len(history)}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Suggested questions ──────────────────────────────────────────────────────
with st.expander("💡 Try one of these questions"):
    cols = st.columns(3)
    suggestions = [
        "How many employees are there in each department?",
        "Who has the highest salary?",
        "What is the total sales amount?",
        "List all employees in the IT department",
        "What is the average salary?",
        "Show top 3 sales by amount",
    ]
    picked = None
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"suggest_{i}"):
            picked = s

# ── Render existing chat history ─────────────────────────────────────────────
for turn in history:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(turn["answer"])
        if turn.get("sql"):
            with st.expander("🔍 Generated SQL"):
                st.markdown(f'<div class="sql-box">{turn["sql"]}</div>', unsafe_allow_html=True)
                if isinstance(turn.get("result"), str) and turn["result"] and not turn.get("error"):
                    st.caption(f"Raw result: {turn['result']}")

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your data…")
if picked:
    user_input = picked

if user_input:
    history = get_history()

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Writing and running SQL…"):
            out = ask_database(user_input, sql_chain, answer_chain)
        st.markdown(out["answer"])
        with st.expander("🔍 Generated SQL", expanded=bool(out["error"])):
            st.markdown(f'<div class="sql-box">{out["sql"]}</div>', unsafe_allow_html=True)
            if out["error"]:
                st.markdown(f'<span class="pill pill-red">Query error</span>', unsafe_allow_html=True)

    history.append({
        "question": user_input,
        "sql": out["sql"],
        "result": out["result"],
        "answer": out["answer"],
        "error": out["error"],
    })
    st.session_state.store[session_id] = history
    st.rerun()
