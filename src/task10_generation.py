"""
Task 10 - Generation With Citation.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

# top_k=5 gives enough evidence without overloading the context. top_p=0.9 keeps
# LLM answers natural while temperature=0.3 keeps RAG generation factual.
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Source, Year]).
If the information is not explicitly stated in the provided context, state
'I cannot verify this information' rather than guessing."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Put the strongest chunk first and the second strongest near the end to reduce
    the lost-in-the-middle effect. Example: [1,2,3,4,5] -> [1,3,5,4,2].
    """
    if len(chunks) <= 2:
        return list(chunks)

    odd_positions = [chunks[i] for i in range(0, len(chunks), 2)]
    even_positions_reversed = [chunks[i] for i in range(len(chunks) - 1, 0, -1) if i % 2 == 1]
    return odd_positions + even_positions_reversed


def format_context(chunks: list[dict]) -> str:
    """Format chunks with source labels for citation."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source") or chunk.get("source") or f"Source {i}"
        doc_type = metadata.get("type", "unknown")
        score = float(chunk.get("score", 0.0))
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citation.

    Returns a dict containing answer, sources and retrieval_source.
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or ""
        except Exception:
            answer = _extractive_answer(query, reordered)
    else:
        answer = _extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    """Local answer fallback that cites retrieved chunks."""
    if not chunks:
        return "I cannot verify this information"

    sentences = []
    for chunk in chunks[:3]:
        content = " ".join(chunk.get("content", "").split())
        if not content:
            continue
        excerpt = content[:260].rstrip()
        source = chunk.get("metadata", {}).get("source", "Source")
        citation = _citation_label(source)
        sentences.append(f"{excerpt} [{citation}].")

    if not sentences:
        return "I cannot verify this information"
    return " ".join(sentences)


def _citation_label(source: str) -> str:
    stem = source.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
    year = next((part for part in stem.split() if part.isdigit() and len(part) == 4), "")
    return f"{stem}, {year}" if year else stem


if __name__ == "__main__":
    result = generate_with_citation("Hinh phat tang tru ma tuy?")
    print(result["answer"])
