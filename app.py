from pathlib import Path

import streamlit as st

from agents.executor import _asks_about_visuals
from agents.retrieval_agent import run_rag
from agents.retriever_worker import LLMQuotaExceeded
from config import DOCUMENTS_DIR, MAX_DOCUMENT_BYTES
from memory.conversation_memory import (
    clear_conversation,
    create_new_conversation,
    list_conversations,
    load_history,
    save_message,
)
from tools.document_ingestion import (
    ensure_documents_indexed,
    get_vector_store_stats,
    save_uploaded_documents,
)
from tools.document_store import (
    SUPPORTED_EXTENSIONS,
    delete_document_chunks,
    invalidate_document_catalog_cache,
)
from tools.visual_processing import collect_visual_assets, safe_image_paths
from workflow_ui import render_workflow_inspector


PROJECT_NAME = "Nexus"


def _image_paths_for_answer(question: str, result: dict) -> list[str]:
    if not _asks_about_visuals(question):
        return []
    paths: list[str] = []
    for asset in result.get("visual_assets") or []:
        p = str(asset.get("path") or asset.get("image_path") or "")
        if p and p not in paths:
            paths.append(p)
    for asset in collect_visual_assets(result.get("context_blocks") or result.get("chunks") or []):
        p = str(asset.get("path") or "")
        if p and p not in paths:
            paths.append(p)
    return paths

st.set_page_config(
    page_title=f"{PROJECT_NAME} | Research",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --nexus-bg: #f5f7f6;
            --nexus-panel: #ffffff;
            --nexus-panel-hover: #f2f7f5;
            --nexus-border: #e2e8e5;
            --nexus-text: #17211d;
            --nexus-muted: #66746d;
            --nexus-accent: #118263;
        }
        html, body, #root, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
            background: var(--nexus-bg) !important;
            color: var(--nexus-text);
        }
        [data-testid="stHeader"] {background: transparent !important;}
        .block-container {max-width: 900px; padding: 2rem 1.5rem 7rem;}
        [data-testid="stSidebar"] {
            min-width: 300px;
            max-width: 300px;
            background: #ffffff !important;
            border-right: 1px solid var(--nexus-border);
            color: var(--nexus-text) !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.25rem;
            background: #ffffff !important;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: var(--nexus-text) !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--nexus-muted) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            letter-spacing: -0.04em;
            margin-bottom: 0;
        }
        [data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid var(--nexus-border);
        }
        [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
            padding: 0.55rem 0.65rem;
            font-weight: 600;
        }
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 10px;
            border-color: var(--nexus-border);
            transition: background 160ms ease, border-color 160ms ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(17, 130, 99, 0.55);
            background: var(--nexus-panel-hover);
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(110deg, #087458, #15946c);
            border-color: transparent;
            color: #fff;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #ffffff;
            border-color: var(--nexus-border);
            border-radius: 10px;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid transparent;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin: 0.8rem 0;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #e9f2ee;
            border-color: #dce8e2;
        }
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
            color: var(--nexus-text);
            line-height: 1.75;
        }
        [data-testid="stChatInput"] > div {
            background: #ffffff;
            border: 1px solid #dce4e0;
            border-radius: 18px;
            box-shadow: 0 12px 36px rgba(25, 55, 42, 0.1);
        }
        [data-testid="stChatInput"] > div:focus-within {
            border-color: rgba(17, 130, 99, 0.6);
            box-shadow: 0 0 0 3px rgba(17, 130, 99, 0.1);
        }
        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: var(--nexus-text) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {color: #77837d;}
        [data-testid="stChatInput"] button {
            background: #118263;
            border-radius: 10px;
        }
        [data-testid="stChatInput"] button:hover {background: #0c6d52;}
        .app-header {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin: 0.15rem 0 2.2rem;
            color: #34443c;
            font-size: 0.92rem;
            font-weight: 600;
        }
        .app-mark {
            display: grid;
            width: 2rem;
            height: 2rem;
            place-items: center;
            border: 1px solid rgba(17, 130, 99, 0.2);
            border-radius: 10px;
            background: rgba(17, 130, 99, 0.08);
            color: var(--nexus-accent);
        }
        .welcome {
            max-width: 700px;
            margin: 11vh auto 2rem;
            text-align: center;
        }
        .welcome-kicker {
            color: var(--nexus-accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .welcome h1 {
            margin: 0.75rem 0;
            color: var(--nexus-text);
            font-size: clamp(2rem, 5vw, 3.2rem);
            font-weight: 650;
            letter-spacing: -0.055em;
        }
        .welcome p {
            max-width: 520px;
            margin: 0 auto;
            color: var(--nexus-muted);
            font-size: 1.04rem;
            line-height: 1.7;
        }
        .prompt-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
            max-width: 650px;
            margin: 2rem auto 0;
            text-align: left;
        }
        .prompt-card {
            padding: 0.9rem 1rem;
            border: 1px solid var(--nexus-border);
            border-radius: 14px;
            background: #ffffff;
            box-shadow: 0 6px 20px rgba(20, 45, 33, 0.04);
            color: #34443c;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .prompt-card span {display: block; margin-bottom: 0.25rem; color: var(--nexus-accent);}
        @media (max-width: 640px) {
            .block-container {padding: 1.25rem 1rem 6rem;}
            .app-header {margin-bottom: 1rem;}
            .welcome {margin-top: 8vh;}
            .prompt-grid {grid-template-columns: 1fr;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if "session_id" not in st.session_state:
    st.session_state.session_id = "local_user"


def _uploaded_documents() -> list[Path]:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        (
            path
            for path in DOCUMENTS_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def _render_document_uploader() -> None:
    st.subheader("Documents")
    st.caption(
        "Upload PDF, JSON, DOCX, TXT, Markdown, CSV, or image files "
        "(PNG, JPG, WEBP, GIF). Image/chart understanding uses the configured vision model."
    )
    uploads = st.file_uploader(
        "Choose one or more documents",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        max_upload_size=MAX_DOCUMENT_BYTES // (1024 * 1024),
        key="document_upload",
    )
    if st.button(
        "Upload and index",
        type="primary",
        use_container_width=True,
        disabled=not uploads,
        key="upload_and_index",
    ):
        try:
            upload_report = save_uploaded_documents(uploads or [])
            for error in upload_report["errors"]:
                st.error(f"{error['name']}: {error['error']}")

            if upload_report["saved"] or upload_report["unchanged"]:
                with st.spinner("Indexing documents in the local vector database..."):
                    ingestion_report = ensure_documents_indexed()
                    stats = get_vector_store_stats()
                if ingestion_report["errors"]:
                    for error in ingestion_report["errors"]:
                        st.error(f"{Path(error['path']).name}: {error['error']}")
                else:
                    invalidate_document_catalog_cache()
                    st.success(
                        f"Indexed {len(upload_report['saved'])} new file(s); "
                        f"{stats['chunks']} chunks are searchable."
                    )
                    st.rerun()
        except Exception as exc:
            st.error(f"Upload or indexing failed: {exc}")

    files = _uploaded_documents()
    st.caption(f"Document library · {len(files)} file(s)")
    if files:
        for path in files:
            column, action = st.columns([4, 1])
            column.write(f"📄 {path.name}")
            if action.button("Delete", key=f"delete_{path.name}"):
                try:
                    deleted_chunks = delete_document_chunks(path)
                    st.success(f"Deleted {path.name} and {deleted_chunks} indexed chunk(s).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete {path.name}: {exc}")
    else:
        st.info("No documents uploaded yet.")


with st.sidebar:
    st.markdown(f"## 📚 {PROJECT_NAME}")
    st.caption("YOUR DOCUMENT RESEARCH SPACE")
    if st.button("＋  New chat", type="primary", use_container_width=True, key="new_conversation"):
        st.session_state.session_id = create_new_conversation()
        st.session_state.latest_workflow = None
        st.rerun()

    st.markdown("#### Recent chats")
    conversation_ids = list_conversations()
    current_id = st.session_state.session_id
    options = list(dict.fromkeys([current_id, *conversation_ids]))
    selected_id = st.selectbox(
        "Conversation",
        options=options,
        index=options.index(current_id),
        format_func=lambda value: "Current conversation" if value == "local_user" else value,
        label_visibility="collapsed",
        key="conversation_selector",
    )
    if selected_id != st.session_state.session_id:
        st.session_state.session_id = selected_id
        st.session_state.latest_workflow = None
        st.rerun()
    if st.button("Clear this conversation", use_container_width=True, key="clear_conversation"):
        clear_conversation(st.session_state.session_id)
        st.session_state.latest_workflow = None
        st.rerun()

    st.markdown("---")
    tab_docs, tab_workflow = st.tabs(["📁 Library", "⚡ Workflow"])
    with tab_docs:
        _render_document_uploader()
    with tab_workflow:
        conversation_history = load_history(st.session_state.session_id)
        latest_wf = st.session_state.get("latest_workflow")
        render_workflow_inspector(latest_wf, conversation_history=conversation_history)

st.markdown(
    f'<div class="app-header"><span class="app-mark">✦</span><span>{PROJECT_NAME} Research</span></div>',
    unsafe_allow_html=True,
)

history = load_history(st.session_state.session_id)
if not history:
    st.markdown(
        """
        <section class="welcome">
            <div class="welcome-kicker">Your personal research assistant</div>
            <h1>What would you like to discover?</h1>
            <p>Ask questions across your documents and get clear answers grounded in your own library.</p>
            <div class="prompt-grid">
                <div class="prompt-card"><span>✦ Summarize</span>Give me the key takeaways from my documents</div>
                <div class="prompt-card"><span>⌕ Find an answer</span>What do my files say about a specific topic?</div>
                <div class="prompt-card"><span>⇄ Compare</span>Compare the approaches in my research papers</div>
                <div class="prompt-card"><span>▤ Explore</span>Show me the most useful sources for my question</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

for message in history:
    role = message.get("role")
    content = message.get("content", "")
    if role in {"user", "assistant"} and content:
        with st.chat_message(role):
            st.markdown(content)
            for image_path in safe_image_paths(message.get("image_paths")):
                st.image(
                    image_path,
                    caption=image_path.name,
                    use_container_width=True,
                )

question = st.chat_input("Ask a question about your documents...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.spinner("Searching your documents and preparing an answer..."):
            session_id = st.session_state.session_id
            history = load_history(session_id)
            result = run_rag(
                question,
                history=history,
            )
            answer = result["answer"]
            image_paths = _image_paths_for_answer(question, result)
            st.session_state.latest_workflow = result
            save_message(session_id, "user", question)
            save_message(
                session_id,
                "assistant",
                answer,
                image_paths=image_paths,
                workflow=result,
            )
        st.rerun()
    except LLMQuotaExceeded as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Could not complete the research request: {exc}")
