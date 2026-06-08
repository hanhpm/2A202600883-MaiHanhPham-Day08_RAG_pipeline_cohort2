"""Streamlit RAG chatbot UI inspired by ChatGPT and NotebookLM."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from group_project.document_store import ingest_upload, storage_status
from group_project.search_service import load_recent_logs, run_chat_answer, run_search


st.set_page_config(page_title="Drug Law RAG", layout="wide")

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp {
        background: #f7f7f4;
        color: #1f1f1f;
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.1rem;
        padding-bottom: 1.5rem;
    }
    [data-testid="stSidebar"] {
        background: #f2f2ee;
        border-right: 1px solid #dfdfd6;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #202020;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #ffffff !important;
        border-color: #d8d8cf !important;
        color: #1f1f1f !important;
    }
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.55rem 0;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        line-height: 1.58;
    }
    div[data-testid="stChatInput"] {
        border-top: 1px solid #deded6;
        padding-top: 0.7rem;
    }
    .app-shell {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 380px;
        gap: 18px;
        align-items: start;
    }
    .chat-surface {
        background: #ffffff;
        border: 1px solid #dfdfd6;
        border-radius: 8px;
        min-height: 74vh;
        padding: 18px 24px 8px 24px;
    }
    .source-panel {
        background: #ffffff;
        border: 1px solid #dfdfd6;
        border-radius: 8px;
        min-height: 74vh;
        padding: 16px;
        position: sticky;
        top: 1rem;
    }
    .brand-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid #ecece5;
        padding-bottom: 12px;
        margin-bottom: 12px;
    }
    .brand-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 0;
        margin: 0;
    }
    .brand-subtitle {
        color: #686861;
        font-size: 13px;
        margin: 2px 0 0 0;
    }
    .status-pill {
        display: inline-block;
        border: 1px solid #d2d2c8;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 12px;
        color: #3b3b36;
        background: #fbfbf8;
        margin: 0 4px 6px 0;
        white-space: nowrap;
    }
    .quick-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        margin: 12px 0 4px 0;
    }
    .quick-card {
        border: 1px solid #e0e0d8;
        border-radius: 8px;
        padding: 10px 11px;
        background: #fbfbf8;
        color: #30302b;
        min-height: 74px;
        font-size: 13px;
        line-height: 1.35;
    }
    .source-card {
        border: 1px solid #e0e0d8;
        border-radius: 8px;
        padding: 11px 12px;
        background: #fbfbf8;
        margin: 9px 0;
    }
    .source-title {
        font-weight: 650;
        font-size: 13px;
        margin-bottom: 5px;
        overflow-wrap: anywhere;
    }
    .source-body {
        color: #4b4b46;
        font-size: 12.5px;
        line-height: 1.45;
    }
    .muted {
        color: #6f6f68;
        font-size: 13px;
    }
    .section-title {
        font-size: 14px;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .empty-state {
        border: 1px dashed #d6d6cc;
        border-radius: 8px;
        padding: 18px;
        background: #fcfcfa;
        color: #65655e;
        font-size: 14px;
        line-height: 1.45;
    }
    @media (max-width: 1080px) {
        .app-shell { grid-template-columns: 1fr; }
        .source-panel { position: static; min-height: auto; }
        .quick-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_source_card(item: dict, index: int, compact: bool = True) -> None:
    metadata = item.get("metadata", {})
    source = metadata.get("source", "unknown")
    doc_type = metadata.get("type", "unknown")
    stage = item.get("retrieval_stage") or item.get("method") or item.get("source", "retrieval")
    score = float(item.get("score", 0.0))
    content = " ".join(item.get("content", "").split())
    max_chars = 520 if compact else 900
    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-title">S{index}. {source}</div>
            <span class="status-pill">{doc_type}</span>
            <span class="status-pill">{stage}</span>
            <span class="status-pill">score {score:.3f}</span>
            <div class="source-body">{content[:max_chars]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

with st.sidebar:
    st.markdown("### Library")
    st.caption("Upload source documents, then embed them into the RAG store.")
    uploaded_files = st.file_uploader(
        "Add sources",
        type=["txt", "md", "pdf", "docx", "csv", "json"],
        accept_multiple_files=True,
    )
    doc_type = st.selectbox("Document type", ["legal", "news", "upload"], index=0)
    if st.button("Embed sources", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one file first.")
        else:
            for file in uploaded_files:
                result = ingest_upload(file.name, file.getvalue(), doc_type=doc_type)
                if result.get("ok"):
                    st.success(
                        f"{result['filename']}: {result['chunks']} chunks | "
                        f"PG={result['postgres']['ok']} | ES={result['elasticsearch']['ok']}"
                    )
                else:
                    st.error(result.get("message", "Upload failed."))

    st.divider()
    st.markdown("### Retrieval")
    top_k = st.slider("Top K", 3, 10, 5)
    rerank_method = st.selectbox(
        "Reranker",
        ["cross_encoder", "jina_api", "qwen_local", "mmr", "rrf"],
        index=0,
    )
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.pending_prompt = ""
        st.rerun()

    st.divider()
    st.markdown("### Runtime")
    status = storage_status()
    pg_status = "online" if status["postgres_pgvector"]["ok"] else "offline"
    es_status = "online" if status["elasticsearch"]["ok"] else "offline"
    st.caption(f"Embedding: {status['embedding_model']}")
    st.caption(f"PostgreSQL/pgvector: {pg_status}")
    st.caption(f"Elasticsearch: {es_status}")
    st.caption(f"Local upload chunks: {status['local_upload_chunks']}")

st.markdown('<div class="app-shell">', unsafe_allow_html=True)

left, right = st.columns([0.68, 0.32], gap="large")

with left:
    st.markdown('<div class="chat-surface">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="brand-row">
            <div>
                <p class="brand-title">Drug Law RAG</p>
                <p class="brand-subtitle">Ask questions, keep follow-ups in context, and inspect citations.</p>
            </div>
            <div>
                <span class="status-pill">chat</span>
                <span class="status-pill">citations</span>
                <span class="status-pill">hybrid retrieval</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                Start with a legal question or upload your own source files from the Library.
                Answers are grounded in retrieved context and linked to source cards on the right.
            </div>
            <div class="quick-grid">
                <div class="quick-card">Dieu 249 Bo luat Hinh su quy dinh gi ve tang tru trai phep chat ma tuy?</div>
                <div class="quick-card">Luat Phong chong ma tuy 2021 cam nhung hanh vi nao?</div>
                <div class="quick-card">Cai nghien ma tuy bat buoc duoc quy dinh nhu the nao?</div>
                <div class="quick-card">So sanh thong tin ve vu viec Chi Dan trong cac bai bao.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask about drug law, decrees, cases, or uploaded documents...")
    active_prompt = prompt or st.session_state.pending_prompt
    if active_prompt:
        st.session_state.pending_prompt = ""
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and drafting a cited answer..."):
                result = run_chat_answer(
                    active_prompt,
                    history=st.session_state.messages[:-1],
                    top_k=top_k,
                    rerank_method=rerank_method,
                )
            st.markdown(result["answer"])
            st.caption(f"Retrieval query: {result['standalone_query'][:220]}")
            st.caption(f"Elapsed: {result['elapsed_ms']} ms")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.session_state.last_sources = result.get("sources", [])
        st.rerun()

with right:
    st.markdown('<div class="source-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Sources</p>', unsafe_allow_html=True)
    if st.session_state.last_sources:
        for i, item in enumerate(st.session_state.last_sources, 1):
            render_source_card(item, i, compact=True)
    else:
        st.markdown(
            """
            <div class="empty-state">
                Source cards will appear here after the first answer. Use them to verify citations and inspect retrieved chunks.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Retrieval comparison", expanded=False):
        query = st.text_input(
            "Compare query",
            value="Dieu 249 Bo luat Hinh su quy dinh gi ve tang tru ma tuy?",
        )
        if st.button("Run comparison", use_container_width=True):
            with st.spinner("Running semantic, BM25, uploaded-doc retrieval, and rerank..."):
                comparison_result = run_search(query, top_k=top_k, rerank_method=rerank_method)
            st.info(comparison_result["comparison"]["summary"])
            st.dataframe(
                [
                    {"stage": stage, **values}
                    for stage, values in comparison_result["comparison"]["stages"].items()
                ],
                use_container_width=True,
            )
            st.markdown("**Top reranked sources**")
            for i, item in enumerate(comparison_result["reranked"][:3], 1):
                render_source_card(item, i, compact=True)

    with st.expander("Recent logs", expanded=False):
        st.dataframe(load_recent_logs(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
