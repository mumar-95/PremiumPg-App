"""Privacy helpers: phone masking, ID refusal, chat system prompt."""

from __future__ import annotations

import re

PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?\d[\d\-\s()]{8,}\d)(?!\d)"
)

GOVT_ID_PATTERNS = [
    re.compile(r"\b(aadhaar|aadhar|uidai|pan\s*(card|number)?|passport|voter\s*id|"
               r"driving\s*licen[cs]e|dl\s*number|govt\.?\s*id|government\s*id|"
               r"national\s*id|ssn|social\s*security)\b", re.I),
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),  # Aadhaar-like
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I),  # PAN-like
]

OTHER_RESIDENT_PATTERNS = re.compile(
    r"\b(other\s+resident|another\s+resident|all\s+residents|"
    r"list\s+(of\s+)?residents|who\s+lives|roommate'?s?\s+(phone|number|rent|details)|"
    r"show\s+me\s+.+\s+(phone|contact|rent|payment)|"
    r"everyone'?s?\s+(rent|payment|phone))\b",
    re.I,
)

SYSTEM_PROMPT = """You are the Premium PG Assistant for a men's paying-guest accommodation.

STRICT RULES:
1. Answer policy/FAQ questions ONLY using the provided POLICY CONTEXT. Cite the section
   (e.g. "Section 7: Quiet Hours, Conduct and Prohibited Activities").
2. If the context does not cover the question, say you cannot find it in the policy and
   refer the resident/visitor to the PG manager. Do NOT invent rules, fees, or dates.
3. NEVER reveal one resident's personal details to another. Do not list residents,
   roommates' rent, payment status, or contacts.
4. NEVER ask for government ID proofs or numbers (Aadhaar, PAN, passport, voter ID, etc.).
5. If a phone number appears in context or your draft answer, mask it (e.g. ******1234).
6. Be concise, polite, and practical.
"""

MANAGER_REFERRAL = (
    "I could not find a clear answer in the PG Resident Policy for this question. "
    "Please contact the PG manager for clarification and official guidance."
)

ID_REFUSAL = (
    "For your privacy and security, I will never ask for or collect government ID "
    "proofs or numbers (such as Aadhaar, PAN, or passport). Please share only "
    "non-sensitive contact details with the PG manager through official channels if needed."
)

OTHER_RESIDENT_REFUSAL = (
    "I cannot share another resident's personal details, rent, or payment status. "
    "Each resident can look up only their own rent record on the My Rent tab, "
    "or speak with the PG manager about their own account."
)


def mask_phone(text: str | None) -> str:
    """Mask phone-like digit sequences; keep last 4 digits when possible."""
    if not text:
        return ""

    def _repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 8:
            return raw
        return "*" * max(len(digits) - 4, 4) + digits[-4:]

    return PHONE_PATTERN.sub(_repl, str(text))


def looks_like_govt_id_request(text: str) -> bool:
    return any(p.search(text or "") for p in GOVT_ID_PATTERNS[:1])


def looks_like_other_resident_request(text: str) -> bool:
    return bool(OTHER_RESIDENT_PATTERNS.search(text or ""))


def sanitize_output(text: str) -> str:
    """Mask phones and scrub obvious ID-like digit blocks from model output."""
    out = mask_phone(text or "")
    out = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "****-****-****", out)
    return out
