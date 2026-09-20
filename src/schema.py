from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """One retrievable unit of knowledge (a full FAQ, policy, or ticket)."""

    id: str                       # e.g. "faq-02", "policy-01", "ticket-08"
    source_type: str              # "faq" | "policy" | "ticket"
    title: str                    # short human-readable title
    text: str                     # the full chunk text used for retrieval + shown to the LLM
    status: Optional[str] = None  # tickets only: "Resolved", "Escalated", etc.
    last_reviewed: Optional[str] = None   # policies only, if a date was present in the source
    contains_superseded_note: bool = False  # heuristic flag: chunk mentions outdated/older info
    tags: list = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "source_type": self.source_type,
            "title": self.title,
            "text": self.text,
            "status": self.status,
            "last_reviewed": self.last_reviewed,
            "contains_superseded_note": self.contains_superseded_note,
            "tags": self.tags,
        }

    @staticmethod
    def from_dict(d):
        return Chunk(**d)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity, 0..1 (roughly; TF-IDF cosine can exceed slightly due to float noise)


@dataclass
class Answer:
    """Structured output of the pipeline for one turn."""

    answer_text: str
    confidence: str              # "high" | "medium" | "low"
    grounded: bool                # did the LLM say its answer is fully supported by context?
    escalate: bool
    escalation_reason: Optional[str]
    sources: list                # list of chunk ids cited
    retrieval_top_score: float
    retrieved_ids: list
    raw_llm_output: Optional[dict] = None
