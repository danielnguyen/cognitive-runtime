from __future__ import annotations

from pathlib import Path

from services.relationship_bootstrap import apply_bootstrap_seed, load_bootstrap_seed
from services.relationships import relationship_repository
from services.social_context import social_context_repository


def _write_seed(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_packaged_seed_path_exists_and_loads() -> None:
    seed_path = Path(__file__).resolve().parents[1] / "config" / "relationships.seed.example.yaml"

    assert seed_path.exists()

    parsed = load_bootstrap_seed(seed_path)

    assert parsed.source_ref == "bootstrap:config/relationships.seed.example.yaml"
    assert any(entity.entity_id == "project:llm-memory" for entity in parsed.entities)
    assert any(
        edge.relationship_id == "rel:llm-memory:contains:cognitive-runtime"
        for edge, _ in parsed.relationships
    )


def test_bootstrap_seed_apply_creates_entities_and_relationships(tmp_path: Path):
    seed = _write_seed(
        tmp_path / "seed.yaml",
        """
entities:
  - entity_id: project:alpha
    entity_type: project
    canonical_label: Alpha
    display_label: Alpha
    domain: project_context
    sensitivity_level: medium
    status: active

  - entity_id: repo:alpha
    entity_type: repository
    canonical_label: alpha-repo
    display_label: alpha-repo
    domain: project_context
    sensitivity_level: medium
    status: active

relationships:
  - relationship_id: rel:alpha:contains:repo
    subject_entity_id: project:alpha
    relationship_type: contains
    object_entity_id: repo:alpha
    relationship_scope: project_context
    confidence: 1.0
    status: active
    sensitivity_level: medium
    mentionability: mentionable
    evidence:
      - evidence_type: config_reference
        source_ref: bootstrap:test
        summary: Trusted config bootstrap relationship.
        confidence_delta: 0.0
""",
    )

    result = apply_bootstrap_seed(owner_id="owner", seed_path=seed)

    assert result["entities_upserted"] == 2
    assert result["relationships_upserted"] == 1
    assert result["evidence_inserted"] == 1
    repo = relationship_repository()
    diagnostics = repo.diagnostics(owner_id="owner")
    assert len(diagnostics.entities) == 2
    assert len(diagnostics.relationships) == 1


def test_bootstrap_seed_is_idempotent_for_entities_relationships_and_evidence(tmp_path: Path):
    seed = _write_seed(
        tmp_path / "seed.yaml",
        """
entities:
  - entity_id: project:alpha
    entity_type: project
    canonical_label: Alpha
    display_label: Alpha
    domain: project_context
    sensitivity_level: medium
    status: active

  - entity_id: repo:alpha
    entity_type: repository
    canonical_label: alpha-repo
    display_label: alpha-repo
    domain: project_context
    sensitivity_level: medium
    status: active

relationships:
  - relationship_id: rel:alpha:contains:repo
    subject_entity_id: project:alpha
    relationship_type: contains
    object_entity_id: repo:alpha
    relationship_scope: project_context
    confidence: 1.0
    status: active
    sensitivity_level: medium
    mentionability: mentionable
    evidence:
      - evidence_type: config_reference
        source_ref: bootstrap:test
        summary: Trusted config bootstrap relationship.
        confidence_delta: 0.0
""",
    )

    first = apply_bootstrap_seed(owner_id="owner", seed_path=seed)
    second = apply_bootstrap_seed(owner_id="owner", seed_path=seed)

    assert first["evidence_inserted"] == 1
    assert second["evidence_inserted"] == 0
    assert second["evidence_skipped"] == 1
    diagnostics = relationship_repository().diagnostics(owner_id="owner")
    assert len(diagnostics.entities) == 2
    assert len(diagnostics.relationships) == 1
    assert len(diagnostics.evidence) == 1


def test_bootstrap_missing_subject_rolls_back_all_writes(tmp_path: Path):
    seed = _write_seed(
        tmp_path / "seed.yaml",
        """
entities:
  - entity_id: repo:alpha
    entity_type: repository
    canonical_label: alpha-repo
    display_label: alpha-repo
    domain: project_context
    sensitivity_level: medium
    status: active

relationships:
  - relationship_id: rel:alpha:contains:repo
    subject_entity_id: project:alpha
    relationship_type: contains
    object_entity_id: repo:alpha
    relationship_scope: project_context
    confidence: 1.0
    status: active
    sensitivity_level: medium
    mentionability: mentionable
""",
    )

    try:
        apply_bootstrap_seed(owner_id="owner", seed_path=seed)
    except RuntimeError as exc:
        assert "relationship_subject_entity_missing" in str(exc)
    else:
        raise AssertionError("expected bootstrap failure")

    diagnostics = relationship_repository().diagnostics(owner_id="owner")
    assert diagnostics.entities == []
    assert diagnostics.relationships == []


def test_bootstrap_missing_object_rolls_back_all_writes(tmp_path: Path):
    seed = _write_seed(
        tmp_path / "seed.yaml",
        """
entities:
  - entity_id: project:alpha
    entity_type: project
    canonical_label: Alpha
    display_label: Alpha
    domain: project_context
    sensitivity_level: medium
    status: active

relationships:
  - relationship_id: rel:alpha:contains:repo
    subject_entity_id: project:alpha
    relationship_type: contains
    object_entity_id: repo:alpha
    relationship_scope: project_context
    confidence: 1.0
    status: active
    sensitivity_level: medium
    mentionability: mentionable
""",
    )

    try:
        apply_bootstrap_seed(owner_id="owner", seed_path=seed)
    except RuntimeError as exc:
        assert "relationship_object_entity_missing" in str(exc)
    else:
        raise AssertionError("expected bootstrap failure")

    diagnostics = relationship_repository().diagnostics(owner_id="owner")
    assert diagnostics.entities == []
    assert diagnostics.relationships == []


def test_bootstrap_rejects_model_inference_active_relationship(tmp_path: Path):
    seed = _write_seed(
        tmp_path / "seed.yaml",
        """
entities:
  - entity_id: project:alpha
    entity_type: project
    canonical_label: Alpha
    display_label: Alpha
    domain: project_context
    sensitivity_level: medium
    status: active

  - entity_id: repo:alpha
    entity_type: repository
    canonical_label: alpha-repo
    display_label: alpha-repo
    domain: project_context
    sensitivity_level: medium
    status: active

relationships:
  - relationship_id: rel:alpha:contains:repo
    subject_entity_id: project:alpha
    relationship_type: contains
    object_entity_id: repo:alpha
    relationship_scope: project_context
    source_type: model_inference
    confidence: 1.0
    status: active
    sensitivity_level: medium
    mentionability: mentionable
""",
    )

    try:
        apply_bootstrap_seed(owner_id="owner", seed_path=seed)
    except RuntimeError as exc:
        assert "model_inference_cannot_create_active_relationship" in str(exc)
    else:
        raise AssertionError("expected bootstrap failure")


def test_bootstrap_accepts_trusted_config_active_relationship(tmp_path: Path):
    seed = _write_seed(
        tmp_path / "seed.yaml",
        """
entities:
  - entity_id: project:alpha
    entity_type: project
    canonical_label: Alpha
    display_label: Alpha
    domain: project_context
    sensitivity_level: medium
    status: active

  - entity_id: repo:alpha
    entity_type: repository
    canonical_label: alpha-repo
    display_label: alpha-repo
    domain: project_context
    sensitivity_level: medium
    status: active

relationships:
  - relationship_id: rel:alpha:contains:repo
    subject_entity_id: project:alpha
    relationship_type: contains
    object_entity_id: repo:alpha
    relationship_scope: project_context
    source_type: trusted_config
    confidence: 1.0
    status: active
    sensitivity_level: medium
    mentionability: mentionable
""",
    )

    result = apply_bootstrap_seed(owner_id="owner", seed_path=seed)
    assert result["relationships_upserted"] == 1


def test_bootstrap_dry_run_validates_without_writing(tmp_path: Path):
    seed = _write_seed(
        tmp_path / "seed.yaml",
        """
entities:
  - entity_id: project:alpha
    entity_type: project
    canonical_label: Alpha
    display_label: Alpha
    domain: project_context
    sensitivity_level: medium
    status: active
""",
    )

    result = apply_bootstrap_seed(owner_id="owner", seed_path=seed, dry_run=True)

    assert result["entities_upserted"] == 1
    diagnostics = relationship_repository().diagnostics(owner_id="owner")
    assert diagnostics.entities == []


def test_bootstrap_preserves_restricted_diagnostics_redaction(tmp_path: Path):
    seed = _write_seed(
        tmp_path / "seed.yaml",
        """
entities:
  - entity_id: project:alpha
    entity_type: project
    canonical_label: Alpha
    display_label: Alpha
    domain: project_context
    sensitivity_level: medium
    status: active

  - entity_id: repo:alpha
    entity_type: repository
    canonical_label: alpha-repo
    display_label: alpha-repo
    domain: project_context
    sensitivity_level: medium
    status: active

relationships:
  - relationship_id: rel:alpha:contains:repo
    subject_entity_id: project:alpha
    relationship_type: contains
    object_entity_id: repo:alpha
    relationship_scope: project_context
    source_type: trusted_config
    source_refs_json:
      - secret:ref
    confidence: 1.0
    status: active
    sensitivity_level: restricted
    mentionability: restricted
    evidence:
      - evidence_type: config_reference
        source_ref: secret:ref
        summary: Restricted bootstrap relationship.
        confidence_delta: 0.0
""",
    )

    apply_bootstrap_seed(owner_id="owner", seed_path=seed)
    diagnostics = relationship_repository().diagnostics(owner_id="owner")

    assert diagnostics.relationships[0].source_refs_json == []
    assert diagnostics.relationships[0].source_refs_redacted is True
    assert diagnostics.evidence[0].summary is None


def test_bootstrap_creates_no_social_context_rows(tmp_path: Path):
    seed = _write_seed(
        tmp_path / "seed.yaml",
        """
entities:
  - entity_id: project:alpha
    entity_type: project
    canonical_label: Alpha
    display_label: Alpha
    domain: project_context
    sensitivity_level: medium
    status: active
""",
    )

    apply_bootstrap_seed(owner_id="owner", seed_path=seed)
    diagnostics = social_context_repository().diagnostics(owner_id="owner")
    assert diagnostics.items == []
    assert diagnostics.usage_events == []
