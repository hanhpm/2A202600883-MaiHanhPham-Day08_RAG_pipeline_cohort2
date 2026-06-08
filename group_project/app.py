"""Streamlit Search Engine for Group Option A."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from group_project.search_service import load_recent_logs, run_generation, run_search


st.set_page_config(page_title="Drug Law Search", page_icon="Search", layout="wide")

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp { background: #ffffff; color: #111111; }
    [data-testid="stSidebar"] { background: #0f0f0f; color: #ffffff; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: #ffffff; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] [role="combobox"],
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="popover"] * {
        color: #111111 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: #ffffff !important;
    }
    .result-box {
        border: 1px solid #111;
        border-radius: 6px;
        padding: 14px;
        margin: 10px 0;
        background: #fff;
    }
    .score-pill {
        display: inline-block;
        border: 1px solid #111;
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 12px;
        margin-right: 6px;
    }
    .small-muted { color: #555; font-size: 13px; }
    .black-band {
        background: #111;
        color: white;
        padding: 18px 20px;
        border-radius: 6px;
        margin-bottom: 18px;
    }
    .black-band h1 { color: white; margin: 0; font-size: 28px; }
    .black-band p { color: #ddd; margin: 6px 0 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_result(item: dict, index: int) -> None:
    metadata = item.get("metadata", {})
    source = metadata.get("source", "unknown")
    doc_type = metadata.get("type", "unknown")
    score = float(item.get("score", 0.0))
    content = " ".join(item.get("content", "").split())
    st.markdown(
        f"""
        <div class="result-box">
            <span class="score-pill">#{index}</span>
            <span class="score-pill">score {score:.3f}</span>
            <span class="score-pill">{doc_type}</span>
            <div class="small-muted">source: {source}</div>
            <p>{content[:900]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="black-band">
        <h1>Drug Law & News Search Engine</h1>
        <p>Hybrid search, reranking, source display, score comparison, and citation answer demo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Controls")
    top_k = st.slider("Top K", 3, 10, 5)
    rerank_method = st.selectbox(
        "Reranker",
        ["cross_encoder", "jina_api", "qwen_local", "mmr", "rrf"],
        index=0,
    )
    run_citation = st.checkbox("Generate citation answer", value=True)

query = st.text_input(
    "Search query",
    value="Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?",
)

submitted = st.button("Search", type="primary")

if submitted and query.strip():
    with st.spinner("Running hybrid search and reranking..."):
        result = run_search(query.strip(), top_k=top_k, rerank_method=rerank_method)

    tab_results, tab_compare, tab_generation, tab_logs = st.tabs(
        ["Reranked Results", "Comparison", "Generation Citation", "Logs"]
    )

    with tab_results:
        st.subheader("Final reranked output")
        st.caption(f"Elapsed: {result['elapsed_ms']} ms")
        for i, item in enumerate(result["reranked"], 1):
            render_result(item, i)

    with tab_compare:
        st.subheader("Retrieval comparison")
        comparison = result["comparison"]
        st.info(comparison["summary"])
        st.dataframe(
            [
                {"stage": stage, **values}
                for stage, values in comparison["stages"].items()
            ],
            use_container_width=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Semantic top results**")
            for i, item in enumerate(result["semantic"][:3], 1):
                render_result(item, i)
        with col2:
            st.markdown("**BM25 top results**")
            for i, item in enumerate(result["lexical"][:3], 1):
                render_result(item, i)
        with col3:
            st.markdown("**Hybrid RRF top results**")
            for i, item in enumerate(result["hybrid"][:3], 1):
                render_result(item, i)

    with tab_generation:
        if run_citation:
            with st.spinner("Generating answer with citation..."):
                answer = run_generation(query.strip(), top_k=top_k)
            st.subheader("Answer with citation")
            st.write(answer.get("answer", ""))
            st.caption(f"Retrieval source: {answer.get('retrieval_source')} | elapsed: {answer.get('elapsed_ms')} ms")
            st.markdown("**Sources used**")
            for i, item in enumerate(answer.get("sources", []), 1):
                render_result(item, i)
        else:
            st.write("Citation generation is disabled in the sidebar.")

    with tab_logs:
        st.subheader("Recent input logs")
        st.dataframe(load_recent_logs(20), use_container_width=True)

else:
    st.caption("Enter a query and press Search.")
    st.markdown("Example queries:")
    st.code(
        "Điều 249 Bộ luật Hình sự quy định gì về tàng trữ ma túy?\n"
        "Nghệ sĩ nào bị điều tra liên quan ma túy?\n"
        "Luật Phòng chống ma túy 2021 quy định gì về cai nghiện?",
        language="text",
    )
