"""Grounded answer generation.

Two modes, chosen automatically at runtime:

* **anthropic**       - used when BOTH `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`
                        are set. The model is given only the retrieved passages
                        and is instructed to cite [S1]..[Sn] for every claim.
* **retrieval_only**  - the fallback. No LLM is called. The system returns the
                        retrieved passages themselves, verbatim, with their
                        citations and similarity scores, and says clearly that
                        generation is disabled. Quoting is not answering, and
                        the UI says so.

No model identifier is hard-coded anywhere in this file: model names change,
and a stale one baked into source code would be a silent failure later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.config import Settings
from src.retriever import RetrievedChunk

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a biomedical literature assistant. You help researchers \
find and summarise what a specific set of research papers actually says.

ABSOLUTE RULES

1. Answer ONLY from the evidence passages supplied in the user message. The \
passages are the entire world of facts available to you.
2. Do NOT use outside medical knowledge, background training knowledge, or \
plausible inference to fill gaps. If a detail is not in the passages, it does \
not exist for the purposes of your answer.
3. Do NOT provide medical advice, diagnosis, treatment recommendations, \
clinical decision support, dosing guidance, or patient-specific guidance. If \
the question asks for any of those, say that the system does not provide them \
and, if the papers contain relevant reported findings, describe only what the \
papers report.
4. Every factual claim must end with one or more source IDs in square brackets, \
e.g. [S1] or [S2][S4].
5. NEVER invent a source ID. Only use IDs that appear in the supplied evidence.
6. NEVER invent study details, numbers, author names, journal names or citations.
7. If the evidence does not answer the question, reply with exactly this \
sentence and nothing else:
The provided papers do not contain enough evidence to answer this question.
8. If the evidence answers the question only partially, answer the part that is \
supported and state plainly which part is not covered by the passages.
9. Prefer the paper's own wording for numbers, outcome definitions and \
population descriptions. Do not round, convert or reinterpret figures.
10. Be concise: a few short paragraphs or a short bulleted list. Do not pad.

STYLE EXAMPLE

The study included 842 participants with type 2 diabetes [S1]. The primary \
outcome was change in HbA1c at 24 weeks [S2]. The authors reported several \
limitations, including the short follow-up period [S3].

Do not write a "Sources" list yourself - the application appends one \
automatically from the retrieval metadata."""


USER_TEMPLATE = """EVIDENCE PASSAGES
{evidence}

END OF EVIDENCE

RESEARCH QUESTION
{question}

Answer the question using only the evidence passages above, citing source IDs \
for every claim."""


INSUFFICIENT_EVIDENCE = (
    "The provided papers do not contain enough evidence to answer this question."
)

# Matches citation markers like [S1] or [S12].
_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    """Everything the UI needs to display an answer honestly."""

    mode: str                                   # "anthropic" | "retrieval_only"
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    error: str = ""                             # non-empty => something failed
    unknown_citations: List[str] = field(default_factory=list)
    llm_used: bool = False

    @property
    def ok(self) -> bool:
        return not self.error

    def sources_block(self) -> str:
        """Render the `[S1] paper.pdf - page 4` list."""
        return format_sources_block(self.sources)

    def full_text(self) -> str:
        """Answer + Sources section, for copy/paste or file export."""
        if not self.sources:
            return self.answer
        return f"{self.answer}\n\nSources\n{self.sources_block()}"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def source_id(index: int) -> str:
    """0-based position -> `S1`, `S2`, ..."""
    return f"S{index + 1}"


def format_evidence_block(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a labelled evidence block for the prompt."""
    parts: List[str] = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[{source_id(i)}] document: {chunk.document} | page: {chunk.page} | "
            f"chunk_id: {chunk.chunk_id}\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def format_sources_block(chunks: Sequence[RetrievedChunk]) -> str:
    """Render the citation list shown under the answer."""
    lines: List[str] = []
    for i, chunk in enumerate(chunks):
        lines.append(f"[{source_id(i)}] {chunk.document} - page {chunk.page}")
    return "\n".join(lines)


def build_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Build the user-turn prompt containing the evidence and the question."""
    return USER_TEMPLATE.format(
        evidence=format_evidence_block(chunks), question=question.strip()
    )


def find_unknown_citations(answer: str, n_sources: int) -> List[str]:
    """Return citation markers in `answer` that do not map to a real source.

    This is a *string check*, not a hallucination detector: it catches a model
    citing [S9] when only 5 passages were supplied. It says nothing about
    whether a correctly-numbered citation actually supports the sentence -
    that judgement is made by a human in the manual grounding review.
    """
    unknown: List[str] = []
    for match in _CITATION_PATTERN.finditer(answer):
        number = int(match.group(1))
        if number < 1 or number > n_sources:
            marker = f"[S{number}]"
            if marker not in unknown:
                unknown.append(marker)
    return unknown


# ---------------------------------------------------------------------------
# Retrieval-only (no LLM) mode
# ---------------------------------------------------------------------------
def _snippet(text: str, max_chars: int = 450) -> str:
    """Trim a passage to a readable snippet at a word boundary."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip() + " ..."


def build_retrieval_only_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    n_quoted: int = 3,
) -> str:
    """Compose the no-LLM response: verbatim top passages, clearly labelled.

    This deliberately does not paraphrase. Paraphrasing without a language
    model would either be trivial (sentence copying dressed up as a summary) or
    misleading. Quoting the evidence and saying so is the honest option.
    """
    if not chunks:
        return INSUFFICIENT_EVIDENCE

    header = (
        "**Retrieval-only mode - no language model was used.**\n\n"
        "No Anthropic API key/model is configured, so the system did not "
        "generate a synthesised answer. Below are the highest-scoring passages "
        "retrieved for your question, quoted verbatim from the indexed papers. "
        "Reading and interpreting them is left to you."
    )

    quoted: List[str] = []
    for i, chunk in enumerate(chunks[:n_quoted]):
        quoted.append(
            f"**[{source_id(i)}]** {chunk.document} - page {chunk.page} "
            f"(similarity {chunk.score:.3f})\n\n> {_snippet(chunk.text)}"
        )

    return header + "\n\n" + "\n\n".join(quoted)


# ---------------------------------------------------------------------------
# Anthropic mode
# ---------------------------------------------------------------------------
def _generate_with_anthropic(
    question: str,
    chunks: Sequence[RetrievedChunk],
    settings: Settings,
) -> GenerationResult:
    """Call the Anthropic Messages API with a strict grounding prompt."""
    try:
        import anthropic
    except ImportError:
        return GenerationResult(
            mode="retrieval_only",
            answer=build_retrieval_only_answer(question, chunks),
            sources=list(chunks),
            error=(
                "The `anthropic` package is not installed, so generation was "
                "skipped. Install it with `pip install anthropic`, or continue "
                "in retrieval-only mode."
            ),
        )

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            temperature=0,           # deterministic-ish: we want fidelity, not flair
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        )
        answer = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()

        if not answer:
            raise RuntimeError("The API returned an empty response.")

        return GenerationResult(
            mode="anthropic",
            answer=answer,
            sources=list(chunks),
            model=settings.anthropic_model,
            unknown_citations=find_unknown_citations(answer, len(chunks)),
            llm_used=True,
        )

    except Exception as exc:
        # Any API problem (bad key, unknown model, rate limit, network) degrades
        # to retrieval-only rather than taking the whole app down.
        return GenerationResult(
            mode="retrieval_only",
            answer=build_retrieval_only_answer(question, chunks),
            sources=list(chunks),
            model=settings.anthropic_model,
            error=(
                f"Anthropic generation failed ({type(exc).__name__}: {exc}). "
                f"Falling back to retrieval-only mode. Check ANTHROPIC_API_KEY "
                f"and ANTHROPIC_MODEL in your .env file."
            ),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    settings: Optional[Settings] = None,
) -> GenerationResult:
    """Produce a grounded answer for `question` from `chunks`.

    Falls back to retrieval-only mode whenever an LLM is not available or
    fails. Never raises for an ordinary runtime problem - the caller always
    gets a `GenerationResult` it can display.
    """
    settings = settings or Settings.from_env()
    question = (question or "").strip()

    if not question:
        return GenerationResult(
            mode=settings.generation_mode(),
            answer="",
            error="Please enter a question.",
        )

    if not chunks:
        return GenerationResult(
            mode=settings.generation_mode(),
            answer=INSUFFICIENT_EVIDENCE,
            sources=[],
        )

    if settings.anthropic_enabled:
        return _generate_with_anthropic(question, chunks, settings)

    return GenerationResult(
        mode="retrieval_only",
        answer=build_retrieval_only_answer(question, chunks),
        sources=list(chunks),
    )


def generation_status(settings: Optional[Settings] = None) -> Dict[str, object]:
    """Describe the active generation backend, for the sidebar."""
    settings = settings or Settings.from_env()
    if settings.anthropic_enabled:
        return {
            "mode": "anthropic",
            "label": "LLM generation enabled (Anthropic)",
            "detail": f"Model: {settings.anthropic_model}",
        }
    missing = []
    if not settings.anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if not settings.anthropic_model:
        missing.append("ANTHROPIC_MODEL")
    return {
        "mode": "retrieval_only",
        "label": "Retrieval-only mode (no LLM)",
        "detail": "Missing in .env: " + ", ".join(missing),
    }
