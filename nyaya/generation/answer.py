"""Generates grounded legal answers using the local vLLM server (Qwen2.5-14B).

Takes a query + retrieved chunks → formats a prompt → calls vLLM at
settings.llm_base_url (OpenAI-compatible endpoint) → returns a structured
answer with inline citations.

The openai Python client is used purely as a wire protocol — zero calls
to OpenAI's servers. All inference runs locally on CHAMP's A100.

Usage (once vLLM is running on CHAMP):
    from nyaya.generation.answer import LegalAnswerer
    answerer = LegalAnswerer()
    result = answerer.answer("What is the punishment for murder?", chunks)
    print(result.answer)
    print(result.citations)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from nyaya.config import settings
from nyaya.schema import Chunk

log = logging.getLogger("nyaya.generation")

_SYSTEM_PROMPT = """You are Nyaya, an AI legal assistant specialising in Indian law.

RULES:
1. Answer ONLY from the provided statute sections — do not use outside knowledge.
2. Cite every legal claim with [Act §Section] inline, e.g. [BNS §103] or [IPC §302].
3. India replaced IPC with BNS, CrPC with BNSS, and IEA with BSA on 1 July 2024.
   When sections from both eras are provided, mention which law applies and when.
4. If the provided sections do not contain enough information to answer, say so clearly.
5. End every answer with: "For free legal aid, contact NALSA: 15100 or nalsa.gov.in"
6. Never fabricate section numbers, act names, or case citations.
"""


def _format_context(chunks: list[Chunk]) -> str:
    """Format retrieved chunks as numbered context blocks for the prompt."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        act = chunk.act or chunk.doc_id
        section = f"§{chunk.section}" if chunk.section else ""
        era_tag = f"[{chunk.era.value}]"
        lines.append(f"[{i}] {act} {section} {era_tag}")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines)


@dataclass
class AnswerResult:
    answer: str
    citations: list[str] = field(default_factory=list)
    chunks_used: list[str] = field(default_factory=list)  # chunk IDs
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _extract_citations(text: str) -> list[str]:
    """Pull inline citations like [BNS §103] or [IPC §302] from answer text."""
    return list(dict.fromkeys(re.findall(r"\[[A-Z]{2,6}\s+§\d+[A-Z]*\]", text)))


class LegalAnswerer:
    """Wraps the local vLLM server for legal Q&A."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        self._model = model or settings.generation_model
        self._client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key="local",  # vLLM doesn't check the key, but client requires one
        )
        log.info("LegalAnswerer pointing at %s (model=%s)", settings.llm_base_url, self._model)

    def answer(
        self,
        question: str,
        chunks: list[Chunk],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AnswerResult:
        """Generate a grounded answer from the provided chunks.

        Args:
            question:   The user's legal question.
            chunks:     Retrieved statute sections (from HybridRetriever).
            max_tokens: Max new tokens the model should generate.
            temperature: Lower = more deterministic. Keep low for legal answers.
        """
        if not chunks:
            return AnswerResult(
                answer="I could not find relevant statute sections to answer your question. "
                       "Please consult a qualified lawyer. "
                       "For free legal aid, contact NALSA: 15100 or nalsa.gov.in"
            )

        context = _format_context(chunks)
        user_message = f"STATUTE SECTIONS:\n{context}\n\nQUESTION: {question}"

        log.info("calling vLLM: model=%s, chunks=%d, question=%r", self._model, len(chunks), question[:60])

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            log.error("vLLM call failed: %s", exc)
            return AnswerResult(
                answer=f"LLM unavailable ({exc}). The retrieved sections are:\n\n{context}"
            )

        answer_text = response.choices[0].message.content or ""
        usage = response.usage

        return AnswerResult(
            answer=answer_text,
            citations=_extract_citations(answer_text),
            chunks_used=[c.id for c in chunks],
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
