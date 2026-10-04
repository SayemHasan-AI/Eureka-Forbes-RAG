import json
import re

import numpy as np
import faiss
import streamlit as st
import torch

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Eureka Forbes Financial Intelligence",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE / UI
# ---------------------------------------------------------

st.title("📊 Eureka Forbes Financial Intelligence")

st.markdown("""
### Enterprise Document Intelligence & RAG Assistant

Ask questions about the **Eureka Forbes Integrated Annual Report 2025–26**.

The system uses:

- 🔎 Hybrid semantic + keyword retrieval
- 🎯 Cross-encoder reranking
- 🤖 Local language model
- 📄 Page-level source attribution
- 🛡️ Grounded answers
- ❌ Refusal of unsupported questions
""")

st.divider()


# ---------------------------------------------------------
# LOAD SYSTEM
# ---------------------------------------------------------

@st.cache_resource
def load_system():

    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index("eureka_forbes.index")
    embeddings = np.load("eureka_embeddings.npy")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32
    )

    return (
        chunks,
        index,
        embeddings,
        embedding_model,
        bm25,
        reranker,
        tokenizer,
        llm
    )


# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------

def hybrid_retrieve(
    query,
    chunks,
    index,
    embedding_model,
    bm25,
    k=20
):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    faiss_scores, faiss_indices = index.search(
        query_embedding,
        len(chunks)
    )

    faiss_scores = faiss_scores[0]

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    faiss_min = faiss_scores.min()
    faiss_max = faiss_scores.max()

    if faiss_max > faiss_min:
        faiss_norm = (
            (faiss_scores - faiss_min)
            / (faiss_max - faiss_min)
        )
    else:
        faiss_norm = faiss_scores

    bm25_min = bm25_scores.min()
    bm25_max = bm25_scores.max()

    if bm25_max > bm25_min:
        bm25_norm = (
            (bm25_scores - bm25_min)
            / (bm25_max - bm25_min)
        )
    else:
        bm25_norm = bm25_scores

    combined_scores = (
        0.6 * faiss_norm
        + 0.4 * bm25_norm
    )

    top_indices = np.argsort(
        combined_scores
    )[::-1][:k]

    results = []

    for idx in top_indices:

        result = chunks[int(idx)].copy()

        result["score"] = float(
            combined_scores[idx]
        )

        results.append(result)

    return results


# ---------------------------------------------------------
# RERANKING
# ---------------------------------------------------------

def reranked_retrieve(
    query,
    chunks,
    index,
    embedding_model,
    bm25,
    reranker,
    initial_k=20,
    final_k=5
):

    candidates = hybrid_retrieve(
        query,
        chunks,
        index,
        embedding_model,
        bm25,
        k=initial_k
    )

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    for candidate, score in zip(
        candidates,
        scores
    ):
        candidate["rerank_score"] = float(score)

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return candidates[:final_k]


# ---------------------------------------------------------
# SAFETY / EVIDENCE CHECKS
# ---------------------------------------------------------

def extract_years(query):
    """
    Pulls out any year the user is asking about, in either
    full form ("2035") or the report's own shorthand ("FY35").
    Returns a list of full 4-digit years as strings, e.g. ["2035"].
    """

    full_years = re.findall(
        r"\b20\d{2}\b",
        query
    )

    fy_years = re.findall(
        r"\bFY\s?0?(\d{2})\b",
        query,
        re.IGNORECASE
    )
    fy_years_full = [f"20{y}" for y in fy_years]

    return list(set(full_years + fy_years_full))


def future_year_requested(query):
    """
    The report covers FY2025-26.
    Any explicitly requested year after 2026 is unsupported.
    """

    years = extract_years(query)

    for year in years:
        if int(year) > 2026:
            return True

    return False


def evidence_supports_query(query, results):
    """
    Basic evidence gate.

    For a specific unsupported year, the requested year must
    actually appear in the retrieved evidence.
    """

    if not results:
        return False

    years = extract_years(query)

    for year in years:

        if year in ["2025", "2026"]:
            continue

        if not any(
            year in result["text"]
            for result in results
        ):
            return False

    return True


# ---------------------------------------------------------
# LLM ANSWER
# ---------------------------------------------------------

def generate_llm_answer(
    query,
    results,
    tokenizer,
    llm
):

    evidence = "\n\n".join(
        f"""
Page {result["page"]}:

{result["text"]}
"""
        for result in results
    )

    prompt = f"""
You are an enterprise document intelligence assistant.

Answer the user's question using ONLY the evidence provided below.

Rules:

1. Use only the provided evidence.
2. Do not use outside knowledge.
3. Do not invent or estimate information.
4. If the evidence is insufficient, say:
"The information is not available in the provided report."
5. Give a concise answer.
6. Mention the relevant page number.

USER QUESTION:

{query}

EVIDENCE:

{evidence}

ANSWER:
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.no_grad():

        output = llm.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False
        )

    answer = tokenizer.decode(
        output[0][
            inputs["input_ids"].shape[1]:
        ],
        skip_special_tokens=True
    )

    return answer.strip()


# ---------------------------------------------------------
# QUESTION BOX
# ---------------------------------------------------------

query = st.text_input(
    "🔍 Ask a question",
    placeholder="e.g. What was the FY26 revenue?"
)

ask = st.button(
    "Ask",
    type="primary"
)


# ---------------------------------------------------------
# RUN RAG
# ---------------------------------------------------------

if ask:

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    # -----------------------------------------------------
    # HARD FUTURE-YEAR SAFETY GATE
    # -----------------------------------------------------

    elif future_year_requested(query):

        st.error(
            "❌ Information not available"
        )

        st.write(
            "The provided Eureka Forbes FY25–26 Annual Report "
            "does not contain information for the requested year."
        )

    else:

        # Load models only after a valid question
        with st.spinner(
            "Loading the document intelligence system..."
        ):

            (
                chunks,
                index,
                embeddings,
                embedding_model,
                bm25,
                reranker,
                tokenizer,
                llm
            ) = load_system()

        # -------------------------------------------------
        # RETRIEVAL
        # -------------------------------------------------

        with st.spinner(
            "Searching the annual report..."
        ):

            results = reranked_retrieve(
                query,
                chunks,
                index,
                embedding_model,
                bm25,
                reranker,
                initial_k=10,
                final_k=3
            )

        # -------------------------------------------------
        # EVIDENCE GATE
        # -------------------------------------------------

        if not evidence_supports_query(
            query,
            results
        ):

            st.error(
                "❌ Information not available"
            )

            st.write(
                "The provided Eureka Forbes FY25–26 Annual Report "
                "does not contain sufficient evidence to answer "
                "this question."
            )

            st.subheader(
                "🔎 Retrieved evidence"
            )

            for i, result in enumerate(
                results,
                1
            ):

                st.markdown(
                    f"**Source {i} — Page {result['page']}**"
                )

                st.write(
                    result["text"]
                )

            st.stop()

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        with st.spinner(
            "Generating grounded answer..."
        ):

            answer = generate_llm_answer(
                query,
                results,
                tokenizer,
                llm
            )

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        st.subheader(
            "💡 Answer"
        )

        st.write(
            answer
        )

        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        st.subheader(
            "📄 Sources"
        )

        pages = sorted(
            set(
                result["page"]
                for result in results
            )
        )

        st.write(
            "Eureka Forbes Integrated Annual Report "
            "2025–26 — Pages "
            + ", ".join(
                str(page)
                for page in pages
            )
        )

        # -------------------------------------------------
        # EVIDENCE
        # -------------------------------------------------

        st.subheader(
            "🔎 Supporting Evidence"
        )

        for i, result in enumerate(
            results,
            1
        ):

            with st.expander(
                f"Source {i} — Page {result['page']}"
            ):

                st.write(
                    result["text"]
                )

                st.caption(
                    "Reranker score: "
                    f"{result['rerank_score']:.3f}"
                )