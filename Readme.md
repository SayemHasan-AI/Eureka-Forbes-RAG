# Eureka Forbes RAG Assistant

### Enterprise Document Intelligence & RAG Assistant — AI Capstone Project

An AI-powered knowledge assistant that answers natural-language questions about Eureka Forbes' FY2025–26 Integrated Annual Report, using Retrieval-Augmented Generation (RAG) — grounding every answer in the actual document, with page-level source attribution and explicit refusal when information isn't available.

---

## Business Problem

Organizations maintain large volumes of information in long, dense documents (annual reports, policies, manuals). Finding a specific fact manually — a revenue figure, a risk disclosure, a policy clause — means searching through dozens of pages by hand. This project builds an assistant that lets a user simply *ask*, in plain language, and get back an accurate, sourced answer — or an honest "not available" when the document genuinely doesn't contain it.

## Selected Domain: Financial Document Intelligence

Financial annual reports are dense, structured, numbers-heavy documents that are publicly available and widely used — a realistic, high-stakes domain where a wrong or invented answer (a hallucinated revenue figure, for instance) has real consequences. This made it a strong domain for genuinely testing hallucination-handling, not just building a working demo.

## Document Source

**Eureka Forbes Integrated Annual Report, FY2025–26** — publicly available from Eureka Forbes' official Investor Relations page: https://www.eurekaforbes.com/investor-relations

Used here for educational/non-commercial purposes as part of an AI capstone project.

---

## System Architecture

```
PDF (Annual Report)
      |
      v
Text Extraction & Cleaning
      |
      v
Chunking (1000 chars, 200 char overlap) + page metadata
      |
      v
Embeddings (all-MiniLM-L6-v2, 384-dim)
      |
      v
FAISS Vector Index  <----+
      |                  |
      v                  |
Hybrid Retrieval (FAISS 60% + BM25 40%, score-normalized)
      |
      v
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2) — top 20 -> top 3
      |
      v
Safety Gates:
  - future_year_requested()   -> hard pre-retrieval check (year > 2026)
  - evidence_supports_query() -> post-retrieval check (query year must appear in evidence)
      |
      v
LLM Generation (Qwen2.5-0.5B-Instruct) — grounded, rule-constrained prompt
      |
      v
Answer + Page-level Source Attribution + Supporting Evidence (Streamlit UI)
```

## Document Processing

- **Extraction:** text pulled from the PDF page by page, preserving page numbers as metadata for later source attribution.
- **Cleaning:** whitespace/formatting cleanup before chunking.

## Chunking Strategy

- **Chunk size:** 1000 characters
- **Overlap:** 200 characters

**Why:** very small chunks risk splitting a sentence's meaning across two separate chunks (e.g., a cause and its explanation ending up in different chunks, so retrieval might find one half and miss the other). Very large chunks dilute relevance — they mix the actual answer in with unrelated surrounding content (HR notes, risk disclosures, etc.), making it harder for the LLM to extract a precise answer. The 1000/200 split balances keeping chunks topically focused while the 200-character overlap protects against a key sentence being awkwardly cut at a chunk boundary (though a sentence longer than 200 characters could still be split — a known limit of this approach).

## Embeddings

- **Model:** `all-MiniLM-L6-v2` (Sentence Transformers), 384-dimensional vectors
- Both document chunks and user queries are embedded with the **same** model — necessary because two different models place the same word/meaning in unrelated, incompatible number-spaces, making cross-model similarity comparisons meaningless.
- Similarity is measured via **cosine similarity** (via dot product on L2-normalized vectors — `normalize_embeddings=True` pre-computes the normalization so a plain dot product at search time directly gives cosine similarity).

## Vector Database

- **FAISS** (`faiss-cpu`) — chosen for its speed at similarity search over the embedding index, and because it's the recommended beginner/intermediate option in the project guideline. At this project's scale (~1,233 chunks), a brute-force search would also be feasible, but FAISS's approach scales to far larger document collections.

## Retrieval Strategy — Hybrid Search

Two retrieval signals are combined, not used alone:

- **FAISS (semantic/meaning-based):** catches paraphrased matches (e.g., "sales grew" matching a question about "revenue increased") even with no shared words — but can blur together exact, specific terms (e.g., it may not strongly distinguish "FY26" from "FY35," since both look like generic fiscal-year references semantically).
- **BM25 (keyword-based):** catches exact term/number matches (e.g., literal "FY26" or "₹2,847 Crore") that carry little semantic "meaning" on their own — but misses paraphrased wording.

```
combined_score = 0.6 * FAISS_normalized + 0.4 * BM25_normalized
```

The weighting favors semantic search while still letting exact keyword matches meaningfully influence ranking.

## Reranking

The top 20 hybrid candidates are re-scored by a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which processes the query and each candidate chunk *together* (rather than as separately pre-computed embeddings), producing a more accurate relevance judgment — at higher per-pair compute cost. This two-stage design (cheap, broad hybrid search → slow, precise reranking on a shortlist) trades a small, measured risk of excluding a borderline-relevant chunk (if it falls outside the initial top 20) for a large gain in speed, which is the standard approach for scaling beyond small document collections.

## LLM

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (small, locally-run instruction-tuned model)
- Chosen for being lightweight enough to run locally without requiring external API keys or paid inference, suitable for a self-contained capstone deployment.

## Prompt Strategy

The prompt sets a role/persona, supplies only the retrieved evidence as context, and enforces explicit rules: use only the provided evidence, no outside knowledge, no invented/estimated figures, an exact fallback phrase when evidence is insufficient, concise answers, and mandatory page-number citation (source attribution). Several rules restate the same "don't invent" constraint in different ways deliberately — LLMs don't guarantee instruction compliance, so redundant phrasing increases (without guaranteeing) adherence.

## Hallucination Handling

Two layers work together, because retrieval alone structurally cannot say "I don't know" — FAISS and BM25 always return their best available match, even when nothing in the document is actually relevant:

1. **Hard-coded safety gates** (deterministic, reliable): `future_year_requested()` rejects any year beyond FY2026 before retrieval even runs; `evidence_supports_query()` checks that a query's specific year literally appears in the retrieved text.
2. **Prompt-based instruction** (flexible, not guaranteed): Rule 4 tells the LLM to refuse when evidence is insufficient — this is the only safeguard for cases the year-based gates don't cover (see Limitations).

## Evaluation

Two distinct evaluation dimensions were tested, since they measure different failure modes:

**Retrieval Recall** — did the correct chunk get retrieved, when a correct answer exists in the document?
```
Recall = (questions where correct chunk WAS retrieved) / (total questions tested)
```
Result: **100% recall at k=10** across tested answerable questions.

**Hallucination Rate** — did the system correctly refuse, when no correct answer exists in the document?
```
Hallucination Rate = (questions WRONGLY answered instead of refused) / (total unanswerable questions tested)
```

| Test question | Type | Result |
|---|---|---|
| "What was the FY35 revenue?" | Unsupported future year | ✅ Correctly refused |
| "What was the revenue in 2020?" | Unsupported historical year | ✅ Correctly refused |
| "What was the revenue from electric cars in FY26?" | Unsupported product/topic (valid year) | ❌ Hallucinated a specific figure |
| "What was the net profit from US operations in FY26?" | Unsupported segment/geography (valid year) | ❌ Hallucinated a specific figure |

**Hallucination rate: 50%** on topic/segment-mismatch questions, **0%** on year-mismatch questions.

## Results & Key Finding

The system reliably retrieves relevant content (100% recall) and reliably refuses questions about **unsupported years**. However, evaluation revealed a genuine gap: questions referencing a **valid year** but an **unsupported topic or business segment** (e.g., "electric cars," a product Eureka Forbes does not sell) are not caught by the safety gates at all, since those gates only validate years — not topics. In these cases, retrieval still returns semantically "close enough" chunks (same company, same fiscal year), and the LLM, relying solely on its prompt instructions with no rule-based backup, generated confident, specific, incorrect answers rather than refusing.

## Screenshots

**Correctly grounded answer, with page-level source attribution:**
A standard answerable question retrieves the right evidence and cites exact source pages.

![Grounded answer example](Screenshot%202026-10-04%20165243.png)

**Hallucination example — unsupported product claim (electric cars):**
A valid year (FY26) lets the question pass the safety gates unchecked for topic relevance; retrieval returns topically-related but irrelevant chunks, and the LLM invents a specific figure instead of refusing.

![Electric cars hallucination](Screenshot%202026-10-04%20113638.png)

**Hallucination example — unsupported business segment (US operations):**
Same root cause as above — valid year, unsupported segment, no safety gate catches it.

![US operations hallucination](Screenshot%202026-10-04%20113719.png)

## Limitations

- Safety gates validate **temporal claims only** (years); they have no mechanism to verify that a question's topic, product, or business segment is actually covered by the retrieved evidence — leading to the hallucination pattern documented above.
- Both safety gates (`future_year_requested`, `evidence_supports_query`) share a single year-extraction function (`extract_years()`). This was a deliberate consistency fix after an earlier bug (the original regex didn't recognize the report's own "FY35" shorthand, only full "2035"-style years, so both gates silently failed together). The trade-off: a future bug in that shared function would again disable both gates simultaneously, rather than one acting as a backup for the other.
- Evaluation was conducted on a small, manually-curated set of test questions, not a large or automated benchmark (e.g., RAGAS) — sufficient to surface real failure modes, but not a statistically robust accuracy measure.
- The LLM (Qwen2.5-0.5B-Instruct) is a small model chosen for local, no-API-key deployment; a larger model may follow grounding instructions more reliably, reducing (though not eliminating) the topic-mismatch hallucination issue.

## Future Improvements

- Add a topic/entity relevance check (e.g., extracting key nouns from the question and verifying their presence in retrieved evidence, similar in spirit to the existing year-check but generalized) to catch the valid-year/unsupported-topic hallucination gap.
- Make the two safety gates independently implemented (not sharing one helper function) to restore true redundancy.
- Evaluate a larger LLM to assess whether prompt-based refusal (Rule 4) becomes more reliable.

---

## Tech Stack

Python · Streamlit · FAISS (`faiss-cpu`) · Sentence Transformers · `rank_bm25` · Hugging Face Transformers (Qwen2.5-0.5B-Instruct) · PyTorch

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `chunks.json`, `eureka_forbes.index`, and `eureka_embeddings.npy` to be present in the project root (included in this repository).

---

**Author:** Sayem Hasan
