"""
pages/8_AI_Chatbot.py - Groq AI Academic Advisor & Local Fallback Chatbot
AI Student Performance & Academic Guidance System
"""

import os
import streamlit as st
from utils import (
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_MUTED,
    apply_custom_css,
    format_chat_history_export_txt,
    generate_chat_response,
    get_active_df,
    get_badge_html,
    init_session_state,
    render_footer,
    render_header,
    render_sidebar_status,
    render_top_navbar,
)

# ---------------------------------------------------------
# Page Configuration & Global Shell
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Chatbot | AI Student Performance",
    layout="wide",
)

init_session_state()
apply_custom_css()
render_top_navbar("pages/8_AI_Chatbot.py")
render_sidebar_status()

render_header(
    title="AI Academic Chatbot",
    subtitle="Ask questions across study planning, attendance improvement, ML concepts, and general educational guidance.",
    badge_text="Module 08",
    badge_variant="indigo",
)

active_df, _ = get_active_df()
model_info = {
    "model_type": st.session_state.get("model_type"),
    "model_task": st.session_state.get("model_task"),
    "model_metrics": st.session_state.get("model_metrics", {}),
}

# Auto-detect GROQ_API_KEY from environment if not yet stored in session state
env_api_key = os.environ.get("GROQ_API_KEY", "").strip()
if not st.session_state.get("groq_api_key") and env_api_key:
    st.session_state["groq_api_key"] = env_api_key


def _reset_chat_history():
    """Resets session chat history and clears the active source indicator."""
    st.session_state["chat_history"] = [
        {
            "role": "assistant",
            "content": (
                "Academic Advisor initialized! I have live visibility into the student cohort telemetry. "
                "Ask me any question about study planning, attendance recovery, machine learning formulas, or cohort benchmarks."
            ),
        }
    ]
    st.session_state["last_chat_source"] = None
    st.session_state["_last_fallback_text"] = None
    st.rerun()


# ---------------------------------------------------------
# Sidebar: Groq API Key & Chat Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("#### Groq AI Settings")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.get("groq_api_key", ""),
        help="Provide a Groq API key (or set GROQ_API_KEY environment variable) to query hosted Llama-3.3-70B. Leave blank to use the smart dynamic fallback advisor.",
    )
    if api_key_input != st.session_state.get("groq_api_key"):
        st.session_state["groq_api_key"] = api_key_input

    groq_model_choice = st.selectbox(
        "Groq Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0,
    )

    is_online = bool((st.session_state.get("groq_api_key") or "").strip() or env_api_key)
    mode_badge = (
        get_badge_html("Groq API Active (Llama-3.3-70B)", "emerald")
        if is_online
        else get_badge_html("Smart Fallback Engine Active", "amber")
    )

    st.markdown(
        f"""
        <div style="background-color: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.75rem; margin-top: 0.5rem; margin-bottom: 0.75rem;">
            <div style="font-size: 0.7rem; text-transform: uppercase; color: {COLOR_MUTED}; font-weight: 600; margin-bottom: 0.35rem;">
                Advisor Mode
            </div>
            <div>{mode_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Chat Controls
    transcript_txt = format_chat_history_export_txt(st.session_state["chat_history"])
    st.download_button(
        label="Download Chat (.txt)",
        data=transcript_txt,
        file_name="academic_advisor_chat.txt",
        mime="text/plain",
        key="sidebar_download_chat",
        width="stretch",
    )
    if st.button("Clear Chat", key="sidebar_clear_chat", width="stretch"):
        _reset_chat_history()

# ---------------------------------------------------------
# Supported Advisory Topics & Quick Prompts
# ---------------------------------------------------------
st.markdown("#### Supported Advisory Topics")
topic_cols = st.columns(5)
quick_prompts = [
    ("1. Greetings & Advice", "Hello advisor! How can you help me navigate my academic goals this semester?"),
    ("2. Attendance Recovery", "How does attendance impact final grades and how can a student recover above the 75% gate?"),
    ("3. Study & Marks", "Provide a structured 4-hour daily study schedule and assignment improvement strategy."),
    ("4. ML Concepts", "Explain Linear vs Logistic Regression formulas, R2, MAE, and feature beta weights."),
    ("5. Cohort Analytics", "How many students are in the dataset, what is the class average, and who is the topper?"),
]

selected_quick_prompt = None
for idx, (col, (title, prompt_text)) in enumerate(zip(topic_cols, quick_prompts)):
    with col:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if st.button("Ask Topic", key=f"chat_topic_{idx}", width="stretch"):
                selected_quick_prompt = prompt_text

# ---------------------------------------------------------
# Conversational Interface & Chat History Controls
# ---------------------------------------------------------
c_hdr1, c_hdr2, c_hdr3 = st.columns([2.0, 1.0, 1.0])
with c_hdr1:
    st.markdown("#### Conversation")
with c_hdr2:
    st.download_button(
        label="Download Chat (.txt)",
        data=format_chat_history_export_txt(st.session_state["chat_history"]),
        file_name="academic_advisor_chat.txt",
        mime="text/plain",
        key="main_download_chat",
        width="stretch",
    )
with c_hdr3:
    if st.button("Clear Chat", key="main_clear_chat", width="stretch"):
        _reset_chat_history()

last_idx = len(st.session_state["chat_history"]) - 1
last_source = st.session_state.get("last_chat_source")
for idx, msg in enumerate(st.session_state["chat_history"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if idx == last_idx and msg["role"] == "assistant" and last_source:
            st.caption(f"Source: {last_source}")

typed_prompt = st.chat_input("Ask about study schedules, attendance recovery, ML formulas, or cohort stats...")
active_prompt = selected_quick_prompt or typed_prompt

if active_prompt:
    # 1. Accept user question & append to session state
    st.session_state["chat_history"].append({"role": "user", "content": active_prompt})

    # 2. Generate AI response (Groq API or smart dynamic fallback)
    reply_text, source_mode = generate_chat_response(
        prompt=active_prompt,
        df=active_df,
        model_info=model_info,
        api_key=st.session_state.get("groq_api_key", ""),
        groq_model=groq_model_choice,
        history=st.session_state["chat_history"][:-1],
    )

    # 3. Persist response & source, then rerun so Download Chat reflects the latest turn immediately
    st.session_state["chat_history"].append({"role": "assistant", "content": reply_text})
    st.session_state["last_chat_source"] = source_mode
    st.rerun()

render_footer()
