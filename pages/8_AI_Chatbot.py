"""
pages/8_AI_Chatbot.py - Groq AI Academic Advisor & Local Fallback Chatbot
AI Student Performance & Academic Guidance System
"""

import streamlit as st
from utils import (
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_MUTED,
    apply_custom_css,
    format_chat_history_export,
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


def _reset_chat_history():
    """Resets session chat history and clears the active source indicator."""
    st.session_state["chat_history"] = [
        {
            "role": "assistant",
            "content": (
                "Chat history cleared. Ask a question about study planning, attendance, "
                "machine learning concepts, or general educational guidance."
            ),
        }
    ]
    st.session_state["last_chat_source"] = None
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
        help="Provide a Groq API key to query hosted LLMs. Leave blank to use the local dataset/rule-based advisor.",
    )
    if api_key_input != st.session_state.get("groq_api_key"):
        st.session_state["groq_api_key"] = api_key_input

    groq_model_choice = st.selectbox(
        "Groq Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0,
    )

    is_online = bool((st.session_state.get("groq_api_key") or "").strip())
    mode_badge = (
        get_badge_html("Groq API Active", "emerald")
        if is_online
        else get_badge_html("Local Fallback Active", "amber")
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

    # Chat Management Controls (Clear Chat & Download Chat)
    transcript_md = format_chat_history_export(st.session_state["chat_history"])
    st.download_button(
        label="Download Chat",
        data=transcript_md,
        file_name="academic_advisor_chat.md",
        mime="text/markdown",
        width="stretch",
    )
    if st.button("Clear Chat", width="stretch"):
        _reset_chat_history()

# ---------------------------------------------------------
# Supported 4 Guidance Areas + Quick Prompts
# ---------------------------------------------------------
st.markdown("#### Supported Advisory Topics")
topic_cols = st.columns(4)
quick_prompts = [
    ("1. Study Planning", "Create a structured weekly study plan and learning strategy to improve exam performance."),
    ("2. Attendance & Improvement", "How does attendance affect academic performance and how can a student recover above 75%?"),
    ("3. ML Concepts", "Explain Linear Regression vs Logistic Regression and the evaluation metrics R2, MAE, and F1-score."),
    ("4. Educational Guidance", "What general habits and dataset indicators matter most for strong academic outcomes?"),
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
c_hdr1, c_hdr2, c_hdr3 = st.columns([2.2, 0.9, 0.9])
with c_hdr1:
    st.markdown("#### Conversation")
with c_hdr2:
    st.download_button(
        label="Download Chat",
        data=format_chat_history_export(st.session_state["chat_history"]),
        file_name="academic_advisor_chat.md",
        mime="text/markdown",
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

typed_prompt = st.chat_input("Ask about study planning, attendance improvement, ML concepts, or guidance...")
active_prompt = selected_quick_prompt or typed_prompt

if active_prompt:
    # 1. Accept user question & append to session state
    st.session_state["chat_history"].append({"role": "user", "content": active_prompt})

    # 2. Generate AI response (Groq API or local fallback)
    reply_text, source_mode = generate_chat_response(
        prompt=active_prompt,
        df=active_df,
        model_info=model_info,
        api_key=st.session_state.get("groq_api_key", ""),
        groq_model=groq_model_choice,
    )

    # 3. Persist response & source, then rerun so Download Chat reflects the latest turn immediately
    st.session_state["chat_history"].append({"role": "assistant", "content": reply_text})
    st.session_state["last_chat_source"] = source_mode
    st.rerun()

render_footer()
