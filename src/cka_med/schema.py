"""Schemas, validation, and input-isolation guards for CKA-MedVQA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


ALLOWED_SPLITS = ("train", "dev", "validation", "test")
ALLOWED_TASK_TYPES = ("V", "J", "C")
ALLOWED_CONDITIONS = ("K0", "Khelp", "Kirrel", "Knear", "Kfalse", "O-H")
ALLOWED_REVIEWER_TYPES = ("agent_audited", "human_audited", "synthetic_fixture")
ALLOWED_PROVENANCE_QUALITY = ("accepted", "questionable", "excluded", "synthetic_fixture")

# Keys that MUST NEVER enter the model prompt or retriever input at inference time.
FORBIDDEN_MODEL_INPUT_KEYS = frozenset(
    {
        "answer",
        "original_answer",
        "normalized_answer",
        "gold_answer",
        "gold_entity",
        "gold_finding",
        "triple",
        "location",
        "report",
        "patient_report",
        "role",
        "context_role",
        "condition",
        "premise_label",
        "applicability_label",
        "task_eligibility_rationale",
        "provenance_quality",
        "oracle_context",
    }
)


class SchemaValidationError(ValueError):
    """Raised when a record or manifest violates the PoC schema."""


class InputLeakageError(RuntimeError):
    """Raised when forbidden gold/role/oracle fields attempt to enter model input."""


class DuplicatePredictionError(RuntimeError):
    """Raised when duplicate (run_id, sample_id) prediction records are written."""


@dataclass
class DataRecord:
    sample_id: str
    group_id: str
    image_id: str
    img_name: str
    source_dataset: str
    source_version: str
    source_qid: str
    image_path: str
    image_sha256: str
    image_dhash: str
    modality: str
    original_question: str
    original_answer: str
    normalized_answer: str
    task_type: str
    task_eligibility_rationale: str
    split: str
    condition: str
    context_pack_id: str
    context_text: str
    context_source_entries: list[dict[str, Any]]
    rendering_version: str
    gold_invariant: bool
    provenance_quality: str
    reviewer_type: str
    exclusion_reason: str | None = None
    patient_id: str | None = None
    premise_label: str | None = None
    verification: str = "not_established"
    oracle_context: bool = False
    synthetic_fixture: bool = False
    content_type: str | None = None
    cxr_candidate: bool = False
    question_family: str | None = None
    answer_choices: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.sample_id or not self.group_id or not self.image_id:
            raise SchemaValidationError("sample_id, group_id, and image_id must be non-empty.")
        if self.split not in ALLOWED_SPLITS:
            raise SchemaValidationError(f"Invalid split: {self.split}")
        if self.task_type not in ALLOWED_TASK_TYPES:
            raise SchemaValidationError(f"Invalid task_type: {self.task_type}")
        if self.condition not in ALLOWED_CONDITIONS:
            raise SchemaValidationError(f"Invalid condition: {self.condition}")
        if self.reviewer_type not in ALLOWED_REVIEWER_TYPES:
            raise SchemaValidationError(f"Invalid reviewer_type: {self.reviewer_type}")
        if self.provenance_quality not in ALLOWED_PROVENANCE_QUALITY:
            raise SchemaValidationError(f"Invalid provenance_quality: {self.provenance_quality}")
        if self.condition == "K0" and self.context_text.strip():
            raise SchemaValidationError("K0 condition must have empty context_text.")
        if self.condition == "O-H" and not self.oracle_context:
            raise SchemaValidationError("O-H condition must set oracle_context=True.")

    def to_model_input_dict(self, *, allow_oracle: bool = False) -> dict[str, Any]:
        """Return ONLY the fields permitted to reach the model/prompt renderer."""
        if self.oracle_context and not allow_oracle:
            raise InputLeakageError(
                f"Sample {self.sample_id} has oracle_context=True, which is forbidden in non-oracle runs."
            )
        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "question": self.original_question,
            "context_text": self.context_text,
            "context_pack_id": self.context_pack_id,
            "task_type": self.task_type,
            "answer_choices": list(self.answer_choices),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataRecord":
        rec = cls(**raw)
        rec.validate()
        return rec


@dataclass
class PredictionRecord:
    run_id: str
    model_id: str
    group_id: str
    sample_id: str
    image_id: str
    condition: str
    task_type: str
    context_pack_id: str
    oracle_context: bool
    raw_output: str
    normalized_answer: str | None
    label_scores: dict[str, float] | None
    abstained: bool
    valid_output: bool
    error: str | None
    latency_ms: float

    def validate(self) -> None:
        if not self.run_id or not self.model_id or not self.sample_id or not self.group_id:
            raise SchemaValidationError("run_id, model_id, group_id, and sample_id must be non-empty.")
        if self.condition not in ALLOWED_CONDITIONS:
            raise SchemaValidationError(f"Invalid prediction condition: {self.condition}")
        if self.task_type not in ALLOWED_TASK_TYPES:
            raise SchemaValidationError(f"Invalid prediction task_type: {self.task_type}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PredictionRecord":
        rec = cls(**raw)
        rec.validate()
        return rec


class PredictionWriter:
    """JSONL prediction writer that prevents duplicate (run_id, sample_id) keys."""

    def __init__(self, path: Path | str, *, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_keys: set[tuple[str, str]] = set()
        if overwrite or not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        else:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["run_id"]), str(row["sample_id"]))
                if key in self._seen_keys:
                    raise DuplicatePredictionError(f"Existing file contains duplicate key {key}")
                self._seen_keys.add(key)

    @property
    def seen_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._seen_keys)

    def write(self, record: PredictionRecord) -> None:
        record.validate()
        key = (record.run_id, record.sample_id)
        if key in self._seen_keys:
            raise DuplicatePredictionError(f"Duplicate prediction key: {key}")
        self._seen_keys.add(key)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def write_many(self, records: Iterable[PredictionRecord]) -> int:
        count = 0
        for rec in records:
            self.write(rec)
            count += 1
        return count


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"Invalid JSON at {p}:{idx}: {exc}") from exc
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p
