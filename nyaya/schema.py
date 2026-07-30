"""Data models shared across all pipeline stages.

All pipeline stages (download → extract → parse → chunk → index) use
these models. Add new fields here, not in individual pipeline files.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Era(StrEnum):
    """Criminal law era for a chunk of text.

    From 1 July 2024, IPC→BNS, CrPC→BNSS, IEA→BSA came into effect.
    Acts that weren't part of this transition (RTI, Consumer Protection, etc.)
    get NEUTRAL so they show up regardless of which era you filter by.
    """

    OLD_CODE = "old_code"
    NEW_CODE = "new_code"
    NEUTRAL = "neutral"


ERA_CUTOVER = date(2024, 7, 1)


class License(StrEnum):
    PUBLIC = "public"  # Government acts / judgments: no copyright bar
    PERMISSIVE = "permissive"  # e.g. CC-BY datasets — record attribution
    RESTRICTED = "restricted"  # must NOT enter the corpus; flag and stop


class Subject(StrEnum):
    CRIMINAL = "criminal"
    CRIMINAL_PROCEDURE = "criminal_procedure"
    EVIDENCE = "evidence"
    CONSTITUTIONAL = "constitutional"
    CONSUMER = "consumer"
    CYBER = "cyber"
    FAMILY = "family"
    CONTRACT = "contract"
    PROPERTY = "property"
    LABOUR = "labour"
    TRANSPARENCY = "transparency"
    FINANCIAL = "financial"
    TRANSPORT = "transport"
    WOMEN_SAFETY = "women_safety"
    CIVIL_PROCEDURE = "civil_procedure"
    LEGAL_AID = "legal_aid"


class JudgmentCatalogEntry(BaseModel):
    """One row of configs/judgments_catalog.yaml — a judgment we intend to fetch.

    Case titles are real, public case names (like act titles in the acts
    catalog). url + citation are filled MANUALLY from the actual e-SCR
    document — never guessed. The downloader refuses any entry without a
    verified https url.
    """

    id: str = Field(pattern=r"^[a-z0-9_]+$", description="slug, e.g. 'sc_kesavananda_1973'")
    title: str = Field(description="case name as printed, 'X v. Y'")
    subject: Subject
    year: int | None = Field(default=None, ge=1947, le=2100)
    source: str = "e-SCR"
    url: str | None = Field(
        default=None, description="direct PDF URL from e-SCR — fill manually"
    )
    citation: str | None = None  # reporter citation, filled from the document
    ik_tid: int | None = Field(
        default=None,
        description="Indian Kanoon doc id — pin the exact judgment when the "
        "name is ambiguous (e.g. from indiankanoon.org/doc/<tid>/)",
    )
    notes: str = ""

    @field_validator("url")
    @classmethod
    def _url_scheme(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("https://"):
            raise ValueError("judgment source URLs must be https")
        return v


class Judgment(BaseModel):
    """A parsed court judgment — metadata extracted, never invented.

    Every optional field is None/empty when the source text does not state it
    (project rule: unverifiable → flag, never fabricate). Chunk ids follow
    '<id>:p<NNN>' per the Chunk docstring (e.g. 'sc_1973_kesavananda:p041').
    """

    id: str = Field(pattern=r"^[a-z0-9_]+$", description="e.g. 'sc_1973_kesavananda'")
    title: str | None = None  # "Petitioner v. Respondent" as printed
    citation: str | None = None  # reporter citation as printed, e.g. "AIR … SC …"
    judgment_date: date | None = None
    bench: list[str] = Field(default_factory=list)  # judge names as printed
    paragraphs: list[str] = Field(default_factory=list)
    held_paras: list[int] = Field(
        default_factory=list,
        description="indexes of paragraphs containing holding language",
    )
    source: str = ""
    url: str | None = None


class QueryType(StrEnum):
    """User-intent classes the query router assigns (PLAN M2, ADR-003).

    Drives index selection (statutes now, judgments in M2) and answer style.
    """

    SECTION = "section"  # asks about a specific statutory section
    RIGHTS = "rights"  # asks what the user is entitled to / protected from
    PROCEDURE = "procedure"  # asks how to do something (file, appeal, apply)
    CASE = "case"  # asks about judgments / case law (served fully in M2)
    GENERAL = "general"  # none of the above — plain hybrid retrieval


class RouteDecision(BaseModel):
    """Deterministic routing verdict for one query — explainable by design."""

    qtype: QueryType
    era: Era  # resolved era for the retrieval filter
    era_source: str = Field(
        description="how the era was chosen: 'explicit' | 'keyword' | 'default'"
    )
    sections: list[str] = Field(
        default_factory=list,
        description="section/article numbers named in the query, e.g. ['302', '120B']",
    )


class CatalogEntry(BaseModel):
    """One row of configs/acts_catalog.yaml — a document we intend to fetch."""

    id: str = Field(pattern=r"^[a-z0-9_]+$", description="stable slug, e.g. 'bns_2023'")
    title: str
    year: int = Field(ge=1800, le=2100)
    era: Era
    subject: Subject
    priority: int = Field(ge=1, le=2, description="1 = must-have for v1, 2 = nice-to-have")
    source: str = Field(description="human-readable origin, e.g. 'India Code'")
    url: str | None = Field(
        default=None,
        description="direct PDF URL, manually verified against indiacode.nic.in",
    )
    notes: str = ""

    @field_validator("url")
    @classmethod
    def _url_scheme(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("https://"):
            raise ValueError("corpus URLs must be https")
        return v


class DocumentMeta(BaseModel):
    """Sidecar metadata written next to every raw file as <id>.meta.json."""

    id: str
    title: str
    source_url: str
    source: str
    era: Era
    subject: Subject
    license: License = License.PUBLIC
    fetch_date: datetime
    sha256: str
    bytes: int
    content_type: str
    catalog_priority: int


class Chunk(BaseModel):
    """A single retrieval unit — one section for statutes, one paragraph window for judgments."""

    id: str = Field(description="e.g. 'bns_2023:s103' or 'sc_1973_kesavananda:p041'")
    text: str = Field(min_length=1)
    doc_id: str
    act: str | None = None
    section: str | None = None
    chapter: str | None = None
    era: Era
    subject: Subject
    lang: str = "en"
    source: str
    license: License = License.PUBLIC
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    def header(self) -> str:
        """Metadata header prepended to text at embedding time, e.g.
        '[Bharatiya Nyaya Sanhita 2023 | Section 103 | new_code]'."""
        section = f"Section {self.section}" if self.section else None
        parts = [p for p in (self.act, section, self.era.value) if p]
        return "[" + " | ".join(parts) + "]"

    def embed_text(self) -> str:
        return f"{self.header()} {self.text}"
