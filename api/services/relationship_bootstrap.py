from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from models import (
    RelationshipEdgeEvidenceInput,
    RelationshipEdgeInput,
    RelationshipEntityInput,
)
from services.relationships import relationship_repository

REPO_ROOT = Path(__file__).resolve().parents[2]


class BootstrapRelationshipSeedModel(BaseModel):
    relationship_id: str = Field(min_length=1, max_length=120)
    subject_entity_id: str = Field(min_length=1, max_length=120)
    relationship_type: str
    object_entity_id: str = Field(min_length=1, max_length=120)
    relationship_scope: str = Field(min_length=1, max_length=64)
    source_type: str | None = None
    source_refs_json: list[str] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: str
    sensitivity_level: str = "medium"
    mentionability: str = "mentionable"
    allowed_persona_scopes_json: list[str] = Field(default_factory=list, max_length=16)
    blocked_persona_scopes_json: list[str] = Field(default_factory=list, max_length=16)
    valid_from: str | None = None
    valid_until: str | None = None
    supersede_existing_relationship_id: str | None = None
    superseded_by_relationship_id: str | None = None
    revoked_at: str | None = None
    evidence: list[RelationshipEdgeEvidenceInput] = Field(default_factory=list, max_length=8)


class BootstrapSeedModel(BaseModel):
    entities: list[dict] = Field(default_factory=list)
    relationships: list[BootstrapRelationshipSeedModel] = Field(default_factory=list)


@dataclass(frozen=True)
class BootstrapParseResult:
    entities: list[RelationshipEntityInput]
    relationships: list[tuple[RelationshipEdgeInput, list[RelationshipEdgeEvidenceInput]]]
    source_ref: str


def load_bootstrap_seed(seed_path: str | Path) -> BootstrapParseResult:
    path = Path(seed_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        parsed = BootstrapSeedModel.model_validate(raw)
    except ValidationError as exc:
        raise RuntimeError(f"relationship_bootstrap_invalid_seed:{exc}") from exc

    source_ref = _canonical_bootstrap_source_ref(path)
    entities = [
        _normalize_entity_input(item, default_source_ref=source_ref)
        for item in parsed.entities
    ]
    relationships = [
        _normalize_relationship_input(item, default_source_ref=source_ref)
        for item in parsed.relationships
    ]
    return BootstrapParseResult(
        entities=entities,
        relationships=relationships,
        source_ref=source_ref,
    )


def apply_bootstrap_seed(
    *,
    owner_id: str,
    seed_path: str | Path,
    dry_run: bool = False,
) -> dict[str, int]:
    parsed = load_bootstrap_seed(seed_path)
    return relationship_repository().bootstrap_apply(
        owner_id=owner_id,
        entities=parsed.entities,
        relationships=parsed.relationships,
        dry_run=dry_run,
    )


def _normalize_entity_input(raw: dict, *, default_source_ref: str) -> RelationshipEntityInput:
    payload = dict(raw)
    if not payload.get("entity_id"):
        raise RuntimeError("relationship_bootstrap_entity_id_required")
    payload.setdefault("source_type", "trusted_config")
    payload.setdefault("source_ref", default_source_ref)
    return RelationshipEntityInput.model_validate(payload)


def _normalize_relationship_input(
    raw: BootstrapRelationshipSeedModel,
    *,
    default_source_ref: str,
) -> tuple[RelationshipEdgeInput, list[RelationshipEdgeEvidenceInput]]:
    source_refs = raw.source_refs_json or [default_source_ref]
    edge = RelationshipEdgeInput(
        relationship_id=raw.relationship_id,
        subject_entity_id=raw.subject_entity_id,
        relationship_type=raw.relationship_type,
        object_entity_id=raw.object_entity_id,
        relationship_scope=raw.relationship_scope,
        source_type=raw.source_type or "trusted_config",
        source_refs_json=source_refs,
        confidence=raw.confidence,
        status=raw.status,
        sensitivity_level=raw.sensitivity_level,
        mentionability=raw.mentionability,
        allowed_persona_scopes_json=raw.allowed_persona_scopes_json,
        blocked_persona_scopes_json=raw.blocked_persona_scopes_json,
        valid_from=raw.valid_from,
        valid_until=raw.valid_until,
        supersede_existing_relationship_id=raw.supersede_existing_relationship_id,
        superseded_by_relationship_id=raw.superseded_by_relationship_id,
        revoked_at=raw.revoked_at,
    )
    return edge, list(raw.evidence)


def _canonical_bootstrap_source_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
        relative_posix = relative.as_posix()
        if relative_posix.startswith("api/config/"):
            relative_posix = relative_posix.removeprefix("api/")
        return f"bootstrap:{relative_posix}"
    except ValueError:
        return f"bootstrap:{path.as_posix()}"
