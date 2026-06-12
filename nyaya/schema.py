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
