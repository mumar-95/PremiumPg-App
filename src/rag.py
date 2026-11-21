"""Section-aware policy RAG with Groq + lightweight local retrieval.

Uses token/TF overlap over policy sections (no torch/FAISS on startup).
Heavy LangChain/Groq imports are deferred until a question is asked.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any

from src.data_loader import load_policy_sections
from src.privacy import (
    ID_REFUSAL,
    MANAGER_REFERRAL,
    OTHER_RESIDENT_REFUSAL,
    SYSTEM_PROMPT,
    looks_like_govt_id_request,
    looks_like_other_resident_request,
    sanitize_output,
)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_GROQ_MODEL = "llama-3.3-70b-versatile"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RagAnswer:
    answer: str
    citations: list[str]
    referred_to_manager: bool


@dataclass
class PolicyIndex:
    """In-memory policy sections with pre-tokenized text for fast retrieval."""

    sections: list[dict[str, Any]]

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return self.sections[:k]

        q_set = set(q_tokens)
        q_tf = _tf(q_tokens)
        scored: list[tuple[float, dict[str, Any]]] = []
        for sec in self.sections:
            tokens: list[str] = sec["_tokens"]
            if not tokens:
                continue
            # Jaccard overlap + TF cosine-ish boost on title hits
            t_set = set(tokens)
            overlap = len(q_set & t_set) / max(len(q_set | t_set), 1)
            title_hits = sum(1 for t in q_set if t in sec["_title_tokens"])
            tf = _tf(tokens)
            dot = sum(q_tf.get(t, 0.0) * tf.get(t, 0.0) for t in q_set)
            norm = math.sqrt(sum(v * v for v in q_tf.values()) * sum(v * v for v in tf.values())) or 1.0
            cosine = dot / norm
            score = overlap + 0.5 * cosine + 0.35 * title_hits
            if score > 0:
                scored.append((score, sec))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            # Fallback: return first k sections so Groq can still attempt / refer manager
            return self.sections[:k]
        return [s for _, s in scored[:k]]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0.0) + 1.0
    n = float(len(tokens)) or 1.0
    return {t: c / n for t, c in counts.items()}


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:
        return None


def get_groq_api_key() -> str | None:
    return _secret("GROQ_API_KEY")


def get_groq_model() -> str:
    return _secret("GROQ_MODEL") or DEFAULT_GROQ_MODEL


def build_policy_vectorstore() -> PolicyIndex:
    """Load and tokenize policy sections (fast — no model download)."""
    sections = load_policy_sections()
    prepared: list[dict[str, Any]] = []
    for sec in sections:
        citation = f"{sec['section_id']}: {sec['section_title']}"
        content = sec["content"]
        title = sec["section_title"]
        prepared.append(
            {
                "section_id": sec["section_id"],
                "section_title": title,
                "citation": citation,
                "content": content,
                "page_content": f"{citation}\n\n{content}",
                "_tokens": _tokenize(f"{title} {content}"),
                "_title_tokens": set(_tokenize(title)),
            }
        )
    return PolicyIndex(sections=prepared)


def _format_context(docs: list[dict[str, Any]]) -> str:
    blocks = []
    for d in docs:
        cite = d.get("citation", "Policy")
        blocks.append(f"[{cite}]\n{d.get('page_content', d.get('content', ''))}")
    return "\n\n---\n\n".join(blocks)


def _citations(docs: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for d in docs:
        c = d.get("citation") or d.get("section_id", "")
        if c and c not in seen:
            seen.append(c)
    return seen


def _make_llm(api_key: str, model: str):
    from langchain_groq import ChatGroq

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0.1,
        max_tokens=1024,
    )


def answer_policy_question(
    question: str,
    vectorstore: PolicyIndex,
    *,
    api_key: str | None = None,
    k: int = 4,
) -> RagAnswer:
    """Retrieve policy sections and answer with Groq, or refer to manager."""
    q = (question or "").strip()
    if not q:
        return RagAnswer(
            answer="Please enter a policy or FAQ question.",
            citations=[],
            referred_to_manager=False,
        )

    if looks_like_govt_id_request(q):
        return RagAnswer(answer=ID_REFUSAL, citations=[], referred_to_manager=False)

    if looks_like_other_resident_request(q):
        return RagAnswer(
            answer=OTHER_RESIDENT_REFUSAL,
            citations=[],
            referred_to_manager=False,
        )

    key = api_key or get_groq_api_key()
    if not key:
        return RagAnswer(
            answer=(
                "Groq API key is not configured. Set GROQ_API_KEY in the environment "
                "or in .streamlit/secrets.toml, then try again."
            ),
            citations=[],
            referred_to_manager=False,
        )

    docs = vectorstore.search(q, k=k)
    if not docs:
        return RagAnswer(
            answer=MANAGER_REFERRAL,
            citations=[],
            referred_to_manager=True,
        )

    context = _format_context(docs)
    cites = _citations(docs)

    from langchain_core.messages import HumanMessage, SystemMessage

    user_prompt = (
        f"POLICY CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{q}\n\n"
        "Instructions: Use only the POLICY CONTEXT. Cite section(s) used. "
        "If the context is insufficient, say so and refer to the PG manager."
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    preferred = get_groq_model()
    models = [preferred]
    if FALLBACK_GROQ_MODEL not in models:
        models.append(FALLBACK_GROQ_MODEL)

    last_error: Exception | None = None
    raw = ""
    for model in models:
        try:
            llm = _make_llm(key, model)
            response = llm.invoke(messages)
            raw = str(response.content)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            raw = ""

    if not raw:
        return RagAnswer(
            answer=(
                f"I could not reach Groq ({last_error}). "
                f"{MANAGER_REFERRAL}"
            ),
            citations=cites,
            referred_to_manager=True,
        )

    lower = raw.lower()
    referred = False
    if any(
        p in lower
        for p in (
            "not mentioned in the policy",
            "not in the policy",
            "no information in the policy",
            "cannot find",
            "could not find",
            "don't have information",
            "do not have information",
            "contact the pg manager",
        )
    ):
        referred = True

    return RagAnswer(
        answer=sanitize_output(raw),
        citations=cites,
        referred_to_manager=referred,
    )


def describe_index(vectorstore: PolicyIndex) -> dict[str, Any]:
    return {
        "num_sections": len(vectorstore.sections),
        "retrieval": "lightweight-token-overlap",
    }
