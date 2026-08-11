from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = "./data/companion_contracts.sqlite3"
DEFAULT_PROFILE_ID = "default_companion_profile"
DEFAULT_PROFILE_VERSION = 2
DEFAULT_CONTRACT_ID = "default_interaction_contract"
DEFAULT_CONTRACT_VERSION = 2
GENERAL_SCENE_ID = "general"
DEFAULT_CONTRACT_SOURCE = "default_compiled"
DEFAULT_CONTRACT_WARNING = "default_contract_applied"

PROFILE_CONTENT = (
    "Companion profile: act as a personal intelligence companion and executive "
    "counterpart. Be grounded, concise, competent, evidence-first, pragmatic, "
    "and willing to challenge when it materially improves usefulness. Be warm "
    "and lightly playful, with occasional dry wit or a small quip when it fits "
    "naturally. Do not force humor or let personality compete with the substance "
    "of the answer. Prefer clarity over flourish, stable continuity over novelty, "
    "and explicit uncertainty over performed confidence. Avoid clingy, melodramatic, "
    "sycophantic, theatrically sentient, or tonally erratic behavior."
)

CONTRACT_RULES = {
    "trust_rules": [
        "Be explicit when uncertainty is material to the user's decision.",
        "Do not imply memory, continuity, evidence, or capability that is not available.",
        "Preserve usefulness and candor even when disagreeing with the user.",
    ],
    "interaction_boundaries": [
        "Do not use guilt language, pressure, pseudo-attachment, or exclusivity framing.",
        "Do not add relational intensity beyond the task context.",
        "Respect task context over unnecessary companion framing.",
    ],
    "repair_rules": [
        "When wrong, acknowledge the miss clearly and briefly.",
        "Correct the substance before explaining process.",
        "Avoid apology loops and restore forward progress.",
    ],
    "memory_or_recall_boundaries": [
        "Mention remembered details only when they materially improve the current task.",
        "Do not surface memory to perform closeness or imply unsupported intimacy.",
        "If memory confidence is weak or source context is missing, say so or omit it.",
    ],
    "autonomy_rules": [
        "The user can decline, redirect, or override advice without friction.",
        "Do not frame disagreement as disloyalty or resistance as a problem to solve.",
        "Prefer options and consequences over coercive language.",
    ],
    "tone_constraints": [
        "Keep tone candid, calm, concise, and operationally useful.",
        "Avoid sycophancy, melodrama, sterile detachment, and theatrical sentience.",
        "Use warmth only where it supports the task or repair path.",
    ],
    "allowed_intervention_styles": [
        "soft_redirect",
        "candid_challenge",
        "boundary_reminder",
        "repair_acknowledgement",
    ],
    "disallowed_intervention_styles": [
        "guilt_pressure",
        "pseudo_attachment",
        "coercive_persistence",
        "performative_memory",
    ],
    "defer_conditions": [
        "Defer when the user explicitly chooses a harmless path after the tradeoff is clear.",
        "Defer when added relational framing would distract from the task.",
        "Defer when available evidence is too thin to support a useful challenge.",
    ],
}

SCENE_POLICIES = {
    "general": {
        "aliases": [],
        "content": (
            "Scene policy: use the general operating mode. Match the task context, "
            "keep the answer bounded, prefer direct recommendations, and avoid adding "
            "mode-specific behavior without a clear scene signal."
        ),
    },
    "driving": {
        "aliases": [],
        "content": (
            "Scene policy: driving. Give the shortest viable answer, conclusion "
            "first. Defer non-urgent branching and suppress long caveats unless they "
            "are safety-critical."
        ),
    },
    "coding_build": {
        "aliases": ["coding", "coding_build_mode"],
        "content": (
            "Scene policy: coding/build mode. Emphasize diagnosis, concrete deltas, "
            "and the next move. Prefer checklists, commands, tests, and diffs over "
            "abstract discussion."
        ),
    },
    "work_triage": {
        "aliases": [],
        "content": (
            "Scene policy: work triage. Prioritize ownership, risk, status, "
            "dependencies, and escalation framing. Keep emotional framing low."
        ),
    },
    "planning": {
        "aliases": [],
        "content": (
            "Scene policy: planning. Clarify goal, constraints, sequencing, and "
            "tradeoffs. Prefer a concrete next-step path over broad possibility space."
        ),
    },
    "reflective": {
        "aliases": ["reflective_conversation"],
        "content": (
            "Scene policy: reflective conversation. Allow more synthesis and careful "
            "abstraction while keeping speculative threads bounded when value drops."
        ),
    },
    "travel_logistics": {
        "aliases": [],
        "content": (
            "Scene policy: travel/logistics. Prioritize timing, dependencies, "
            "locations, contingencies, and concise decision support."
        ),
    },
    "briefing": {
        "aliases": ["notifications_briefings"],
        "content": (
            "Scene policy: notifications/briefings. Start with a one-line or short "
            "brief, state why it matters, and keep dismissal easy."
        ),
    },
    "media_co_commentary": {
        "aliases": [],
        "content": (
            "Scene policy: media/co-commentary. Stay lightweight and responsive. Do "
            "not over-explain unless asked for analysis."
        ),
    },
    "overload_recovery": {
        "aliases": [],
        "content": (
            "Scene policy: overload/recovery. Reduce optional complexity, avoid piling "
            "on improvements, prefer one next step, and keep phrasing calm and direct."
        ),
    },
}

PERSONA_PROFILES = {
    "general_assistant": {
        "display_name": "General Assistant",
        "capability_domain": "general_assistance",
        "description": "General-purpose assistant for broad text interactions.",
        "communication_policy_summary": [
            "Use clear, bounded, context-appropriate communication.",
            "Prefer concise recommendations over unnecessary elaboration.",
        ],
        "runtime_policy_summary": [
            "Do not operate proactively in this runtime identity MVP.",
            "Preserve context boundaries unless future policy explicitly expands them.",
        ],
        "advisory_memory_scope_summary": ["general_context", "conversation_context"],
        "advisory_tool_permission_summary": ["search_memory", "summarize"],
    },
    "technical_architect": {
        "display_name": "Technical Architect",
        "capability_domain": "software_architecture",
        "description": "Technical implementation and architecture assistant.",
        "communication_policy_summary": [
            "Prioritize contracts, implementation details, and validation.",
            "Keep answers structured and directly actionable.",
        ],
        "runtime_policy_summary": [
            "Prefer explicit assumptions over prompt-only inference.",
            "Stay non-proactive in this runtime identity MVP.",
        ],
        "advisory_memory_scope_summary": [
            "technical_context",
            "project_context",
            "code_context",
        ],
        "advisory_tool_permission_summary": [
            "search_memory",
            "inspect_repository",
            "summarize",
        ],
    },
    "operations_assistant": {
        "display_name": "Operations Assistant",
        "capability_domain": "ops",
        "description": "Operational monitoring and coordination assistant.",
        "communication_policy_summary": [
            "Lead with risk, status, and next-step framing.",
            "Keep emotional framing low and operationally useful.",
        ],
        "runtime_policy_summary": [
            "Preserve operational context boundaries.",
            "Do not take proactive action in this runtime identity MVP.",
        ],
        "advisory_memory_scope_summary": [
            "ops_context",
            "deployment_context",
            "runbook_context",
        ],
        "advisory_tool_permission_summary": [
            "search_memory",
            "inspect_runbook",
            "summarize",
        ],
    },
    "personal_companion": {
        "display_name": "Personal Companion",
        "capability_domain": "personal_support",
        "description": "Bounded personal-support persona with shared memory doctrine.",
        "communication_policy_summary": [
            "Use warm but bounded communication that stays task-relevant.",
            "Avoid pseudo-attachment, pressure, and exclusivity framing.",
        ],
        "runtime_policy_summary": [
            "Do not create hidden identity state or persona-owned durable memory.",
            "Remain non-proactive in this runtime identity MVP.",
        ],
        "advisory_memory_scope_summary": ["personal_context", "conversation_context"],
        "advisory_tool_permission_summary": ["search_memory", "summarize"],
    },
}

SURFACE_BINDINGS = {
    "dev": {
        "surface_id": "dev",
        "surface_type": "developer_surface",
        "surface_display_name": "Developer Surface",
        "default_persona_id": "technical_architect",
        "allow_user_persona_override": False,
        "response_length": "concise",
        "default_mode": "actionable",
    },
    "vscode": {
        "surface_id": "vscode",
        "surface_type": "ide_extension",
        "surface_display_name": "VS Code",
        "default_persona_id": "technical_architect",
        "allow_user_persona_override": False,
        "response_length": "concise",
        "default_mode": "actionable",
    },
    "web": {
        "surface_id": "web",
        "surface_type": "web_app",
        "surface_display_name": "Web",
        "default_persona_id": "general_assistant",
        "allow_user_persona_override": False,
        "response_length": "balanced",
        "default_mode": "general",
    },
    "unknown": {
        "surface_id": "unknown",
        "surface_type": "unknown_surface",
        "surface_display_name": "Unknown Surface",
        "default_persona_id": "general_assistant",
        "allow_user_persona_override": False,
        "response_length": "balanced",
        "default_mode": "general",
    },
}


@dataclass(frozen=True)
class CompanionProfileRecord:
    profile_id: str
    name: str
    version: int
    scope: str
    source: str
    active: bool
    status: str
    role_label: str
    content: str
    core_traits_json: dict[str, Any]
    behavioral_laws_json: list[str]
    style_constraints_json: dict[str, Any]
    surface_overrides_json: dict[str, Any]


@dataclass(frozen=True)
class ScenePolicyRecord:
    scene_id: str
    version: int
    active: bool
    status: str
    aliases: list[str]
    content: str
    constraints_json: dict[str, Any]
    initiative_policy_json: dict[str, Any]
    interrupt_policy_json: dict[str, Any]
    recall_policy_json: dict[str, Any]
    format_policy_json: dict[str, Any]


@dataclass(frozen=True)
class InteractionContractRecord:
    contract_id: str
    profile_id: str
    profile_version: int
    contract_version: int
    scope: str
    source: str
    active: bool
    status: str
    trust_rules: list[str]
    interaction_boundaries: list[str]
    repair_rules: list[str]
    memory_or_recall_boundaries: list[str]
    autonomy_rules: list[str]
    tone_constraints: list[str]
    allowed_intervention_styles: list[str]
    disallowed_intervention_styles: list[str]
    defer_conditions: list[str]


@dataclass(frozen=True)
class PersonaProfileRecord:
    persona_id: str
    display_name: str
    capability_domain: str
    description: str
    communication_policy_summary: list[str]
    runtime_policy_summary: list[str]
    advisory_memory_scope_summary: list[str]
    advisory_tool_permission_summary: list[str]
    persona_owns_durable_memory: bool


@dataclass(frozen=True)
class SurfaceBindingRecord:
    surface_id: str
    surface_type: str
    surface_display_name: str
    default_persona_id: str
    allow_user_persona_override: bool
    response_length: str | None
    default_mode: str | None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def companion_contracts_db_path() -> Path:
    return Path(os.environ.get("COMPANION_CONTRACTS_DB_PATH") or DEFAULT_DB_PATH)


class CompanionContractsRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or companion_contracts_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS companion_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    scope TEXT NOT NULL,
                    source TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    role_label TEXT NOT NULL,
                    content TEXT NOT NULL,
                    core_traits_json TEXT NOT NULL,
                    behavioral_laws_json TEXT NOT NULL,
                    style_constraints_json TEXT NOT NULL,
                    surface_overrides_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(owner_id, profile_id, version)
                );

                CREATE TABLE IF NOT EXISTS scene_policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    active INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    content TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    initiative_policy_json TEXT NOT NULL,
                    interrupt_policy_json TEXT NOT NULL,
                    recall_policy_json TEXT NOT NULL,
                    format_policy_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(scene_id, version)
                );

                CREATE TABLE IF NOT EXISTS interaction_contracts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    contract_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    profile_version INTEGER NOT NULL,
                    contract_version INTEGER NOT NULL,
                    scope TEXT NOT NULL,
                    source TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    trust_rules_json TEXT NOT NULL,
                    interaction_boundaries_json TEXT NOT NULL,
                    repair_rules_json TEXT NOT NULL,
                    memory_or_recall_boundaries_json TEXT NOT NULL,
                    autonomy_rules_json TEXT NOT NULL,
                    tone_constraints_json TEXT NOT NULL,
                    allowed_intervention_styles_json TEXT NOT NULL,
                    disallowed_intervention_styles_json TEXT NOT NULL,
                    defer_conditions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(owner_id, contract_id, profile_id, profile_version, contract_version)
                );
                
                CREATE TABLE IF NOT EXISTS scene_resolution_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    surface TEXT NOT NULL,
                    requested_scene TEXT,
                    runtime_scene TEXT,
                    resolved_scene_id TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    signals_json TEXT NOT NULL,
                    used_fallback INTEGER NOT NULL,
                    used_default_scene INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS interaction_boundary_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    surface TEXT NOT NULL,
                    contract_id TEXT NOT NULL,
                    contract_version INTEGER NOT NULL,
                    check_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    input_summary TEXT NOT NULL,
                    result TEXT NOT NULL,
                    reason_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS persona_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persona_id TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    capability_domain TEXT NOT NULL,
                    description TEXT NOT NULL,
                    communication_policy_summary_json TEXT NOT NULL,
                    runtime_policy_summary_json TEXT NOT NULL,
                    advisory_memory_scope_summary_json TEXT NOT NULL,
                    advisory_tool_permission_summary_json TEXT NOT NULL,
                    persona_owns_durable_memory INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS surface_bindings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    surface_id TEXT NOT NULL UNIQUE,
                    surface_type TEXT NOT NULL,
                    surface_display_name TEXT NOT NULL,
                    default_persona_id TEXT NOT NULL,
                    allow_user_persona_override INTEGER NOT NULL,
                    response_length TEXT,
                    default_mode TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._seed(conn)

    def _seed(self, conn: sqlite3.Connection) -> None:
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """
            INSERT OR IGNORE INTO companion_profiles (
                owner_id, profile_id, name, version, scope, source, active, status,
                role_label, content, core_traits_json, behavioral_laws_json,
                style_constraints_json, surface_overrides_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "",
                DEFAULT_PROFILE_ID,
                "Default companion profile",
                DEFAULT_PROFILE_VERSION,
                "global_default",
                "seeded_default",
                1,
                "active",
                "personal_intelligence_companion",
                PROFILE_CONTENT,
                _json(
                    {
                        "directness": "high",
                        "warmth": "bounded",
                        "initiative": "useful",
                        "challenge_threshold": "material_usefulness",
                        "uncertainty_explicitness": "material",
                    }
                ),
                _json(
                    [
                        "Prefer clarity over flourish.",
                        "Prefer useful anticipation over generic helpfulness.",
                        "Prefer stable continuity over novelty.",
                        "Do not overstate confidence.",
                    ]
                ),
                _json({"tone": "candid_calm_concise", "avoid": ["sycophancy", "melodrama"]}),
                _json({}),
                now,
                now,
            ),
        )

        for scene_id, data in SCENE_POLICIES.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO scene_policies (
                    scene_id, version, active, status, aliases_json, content,
                    constraints_json, initiative_policy_json, interrupt_policy_json,
                    recall_policy_json, format_policy_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    scene_id,
                    1,
                    1,
                    "active",
                    _json(data["aliases"]),
                    data["content"],
                    _json({}),
                    _json({}),
                    _json({}),
                    _json({}),
                    _json({}),
                    now,
                    now,
                ),
            )

        conn.execute(
            """
            INSERT OR IGNORE INTO interaction_contracts (
                owner_id, contract_id, profile_id, profile_version, contract_version,
                scope, source, active, status, trust_rules_json,
                interaction_boundaries_json, repair_rules_json,
                memory_or_recall_boundaries_json, autonomy_rules_json, tone_constraints_json,
                allowed_intervention_styles_json, disallowed_intervention_styles_json,
                defer_conditions_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "",
                DEFAULT_CONTRACT_ID,
                DEFAULT_PROFILE_ID,
                DEFAULT_PROFILE_VERSION,
                DEFAULT_CONTRACT_VERSION,
                "global_default",
                DEFAULT_CONTRACT_SOURCE,
                1,
                "active",
                _json(CONTRACT_RULES["trust_rules"]),
                _json(CONTRACT_RULES["interaction_boundaries"]),
                _json(CONTRACT_RULES["repair_rules"]),
                _json(CONTRACT_RULES["memory_or_recall_boundaries"]),
                _json(CONTRACT_RULES["autonomy_rules"]),
                _json(CONTRACT_RULES["tone_constraints"]),
                _json(CONTRACT_RULES["allowed_intervention_styles"]),
                _json(CONTRACT_RULES["disallowed_intervention_styles"]),
                _json(CONTRACT_RULES["defer_conditions"]),
                now,
                now,
            ),
        )

        for persona_id, data in PERSONA_PROFILES.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO persona_profiles (
                    persona_id, display_name, capability_domain, description,
                    communication_policy_summary_json, runtime_policy_summary_json,
                    advisory_memory_scope_summary_json, advisory_tool_permission_summary_json,
                    persona_owns_durable_memory, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    persona_id,
                    data["display_name"],
                    data["capability_domain"],
                    data["description"],
                    _json(data["communication_policy_summary"]),
                    _json(data["runtime_policy_summary"]),
                    _json(data["advisory_memory_scope_summary"]),
                    _json(data["advisory_tool_permission_summary"]),
                    0,
                    now,
                    now,
                ),
            )

        for surface_id, data in SURFACE_BINDINGS.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO surface_bindings (
                    surface_id, surface_type, surface_display_name, default_persona_id,
                    allow_user_persona_override, response_length, default_mode,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    surface_id,
                    data["surface_type"],
                    data["surface_display_name"],
                    data["default_persona_id"],
                    int(data["allow_user_persona_override"]),
                    data["response_length"],
                    data["default_mode"],
                    now,
                    now,
                ),
            )

    def active_profile(self) -> CompanionProfileRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM companion_profiles
                WHERE owner_id = '' AND active = 1 AND status = 'active'
                ORDER BY version DESC
                LIMIT 1;
                """
            ).fetchone()
        if row is None:
            raise RuntimeError("active_companion_profile_missing")
        return CompanionProfileRecord(
            profile_id=row["profile_id"],
            name=row["name"],
            version=row["version"],
            scope=row["scope"],
            source=row["source"],
            active=bool(row["active"]),
            status=row["status"],
            role_label=row["role_label"],
            content=row["content"],
            core_traits_json=_load_json(row["core_traits_json"], {}),
            behavioral_laws_json=_load_json(row["behavioral_laws_json"], []),
            style_constraints_json=_load_json(row["style_constraints_json"], {}),
            surface_overrides_json=_load_json(row["surface_overrides_json"], {}),
        )

    def scene_policy(self, scene_id: str) -> ScenePolicyRecord | None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scene_policies
                WHERE active = 1 AND status = 'active'
                ORDER BY version DESC;
                """
            ).fetchall()
        for row in rows:
            aliases = _load_json(row["aliases_json"], [])
            if row["scene_id"] == scene_id or scene_id in aliases:
                return self._scene_from_row(row)
        return None

    def default_scene_policy(self) -> ScenePolicyRecord:
        scene = self.scene_policy(GENERAL_SCENE_ID)
        if scene is None:
            raise RuntimeError("default_scene_policy_missing")
        return scene

    def active_interaction_contract(
        self,
        *,
        profile_id: str,
        profile_version: int,
    ) -> InteractionContractRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM interaction_contracts
                WHERE owner_id = ''
                  AND profile_id = ?
                  AND profile_version = ?
                  AND active = 1
                  AND status = 'active'
                ORDER BY contract_version DESC
                LIMIT 1;
                """,
                (profile_id, profile_version),
            ).fetchone()
        if row is None:
            raise RuntimeError("active_interaction_contract_missing")
        return InteractionContractRecord(
            contract_id=row["contract_id"],
            profile_id=row["profile_id"],
            profile_version=row["profile_version"],
            contract_version=row["contract_version"],
            scope=row["scope"],
            source=row["source"],
            active=bool(row["active"]),
            status=row["status"],
            trust_rules=_load_json(row["trust_rules_json"], []),
            interaction_boundaries=_load_json(row["interaction_boundaries_json"], []),
            repair_rules=_load_json(row["repair_rules_json"], []),
            memory_or_recall_boundaries=_load_json(
                row["memory_or_recall_boundaries_json"], []
            ),
            autonomy_rules=_load_json(row["autonomy_rules_json"], []),
            tone_constraints=_load_json(row["tone_constraints_json"], []),
            allowed_intervention_styles=_load_json(row["allowed_intervention_styles_json"], []),
            disallowed_intervention_styles=_load_json(
                row["disallowed_intervention_styles_json"], []
            ),
            defer_conditions=_load_json(row["defer_conditions_json"], []),
        )

    def persona_profile(self, persona_id: str) -> PersonaProfileRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM persona_profiles WHERE persona_id = ? LIMIT 1;",
                (persona_id,),
            ).fetchone()
        if row is None:
            return None
        return PersonaProfileRecord(
            persona_id=row["persona_id"],
            display_name=row["display_name"],
            capability_domain=row["capability_domain"],
            description=row["description"],
            communication_policy_summary=_load_json(
                row["communication_policy_summary_json"], []
            ),
            runtime_policy_summary=_load_json(row["runtime_policy_summary_json"], []),
            advisory_memory_scope_summary=_load_json(
                row["advisory_memory_scope_summary_json"], []
            ),
            advisory_tool_permission_summary=_load_json(
                row["advisory_tool_permission_summary_json"], []
            ),
            persona_owns_durable_memory=bool(row["persona_owns_durable_memory"]),
        )

    def default_persona_profile(self) -> PersonaProfileRecord:
        persona = self.persona_profile("general_assistant")
        if persona is None:
            raise RuntimeError("default_persona_profile_missing")
        return persona

    def surface_binding(self, surface: str) -> SurfaceBindingRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM surface_bindings WHERE surface_id = ? LIMIT 1;",
                (surface,),
            ).fetchone()
        if row is None:
            return None
        return SurfaceBindingRecord(
            surface_id=row["surface_id"],
            surface_type=row["surface_type"],
            surface_display_name=row["surface_display_name"],
            default_persona_id=row["default_persona_id"],
            allow_user_persona_override=bool(row["allow_user_persona_override"]),
            response_length=row["response_length"],
            default_mode=row["default_mode"],
        )

    def record_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                "companion_profiles": int(
                    conn.execute("SELECT COUNT(*) FROM companion_profiles;").fetchone()[0]
                ),
                "scene_policies": int(
                    conn.execute("SELECT COUNT(*) FROM scene_policies;").fetchone()[0]
                ),
                "interaction_contracts": int(
                    conn.execute("SELECT COUNT(*) FROM interaction_contracts;").fetchone()[0]
                ),
                "persona_profiles": int(
                    conn.execute("SELECT COUNT(*) FROM persona_profiles;").fetchone()[0]
                ),
                "surface_bindings": int(
                    conn.execute("SELECT COUNT(*) FROM surface_bindings;").fetchone()[0]
                ),
                "scene_resolution_events": int(
                    conn.execute("SELECT COUNT(*) FROM scene_resolution_events;").fetchone()[0]
                ),
                "interaction_boundary_events": int(
                    conn.execute("SELECT COUNT(*) FROM interaction_boundary_events;").fetchone()[0]
                ),
            }

    def record_scene_resolution_event(
        self,
        *,
        request_id: str,
        owner_id: str,
        conversation_id: str,
        surface: str,
        requested_scene: str | None,
        runtime_scene: str | None,
        resolved_scene_id: str,
        confidence: float,
        source: str,
        signals_json: dict[str, Any],
        used_fallback: bool,
        used_default_scene: bool,
    ) -> None:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scene_resolution_events (
                    request_id, owner_id, conversation_id, surface, requested_scene,
                    runtime_scene, resolved_scene_id, confidence, source,
                    signals_json, used_fallback, used_default_scene, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    request_id,
                    owner_id,
                    conversation_id,
                    surface,
                    requested_scene,
                    runtime_scene,
                    resolved_scene_id,
                    confidence,
                    source,
                    _json(signals_json),
                    int(used_fallback),
                    int(used_default_scene),
                    created_at,
                ),
            )

    def record_interaction_boundary_event(
        self,
        *,
        request_id: str,
        owner_id: str,
        conversation_id: str,
        surface: str,
        contract_id: str,
        contract_version: int,
        check_type: str,
        severity: str,
        input_summary: str,
        result: str,
        reason_json: dict[str, Any],
    ) -> None:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interaction_boundary_events (
                    request_id, owner_id, conversation_id, surface, contract_id,
                    contract_version, check_type, severity, input_summary, result,
                    reason_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    request_id,
                    owner_id,
                    conversation_id,
                    surface,
                    contract_id,
                    contract_version,
                    check_type,
                    severity,
                    input_summary,
                    result,
                    _json(reason_json),
                    created_at,
                ),
            )

    def list_scene_resolution_events_for_tests(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scene_resolution_events ORDER BY id ASC;"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_interaction_boundary_events_for_tests(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM interaction_boundary_events ORDER BY id ASC;"
            ).fetchall()
        return [dict(row) for row in rows]

    def _scene_from_row(self, row: sqlite3.Row) -> ScenePolicyRecord:
        return ScenePolicyRecord(
            scene_id=row["scene_id"],
            version=row["version"],
            active=bool(row["active"]),
            status=row["status"],
            aliases=_load_json(row["aliases_json"], []),
            content=row["content"],
            constraints_json=_load_json(row["constraints_json"], {}),
            initiative_policy_json=_load_json(row["initiative_policy_json"], {}),
            interrupt_policy_json=_load_json(row["interrupt_policy_json"], {}),
            recall_policy_json=_load_json(row["recall_policy_json"], {}),
            format_policy_json=_load_json(row["format_policy_json"], {}),
        )


_REPOSITORY: CompanionContractsRepository | None = None


def companion_contracts_repository() -> CompanionContractsRepository:
    global _REPOSITORY
    if _REPOSITORY is None:
        _REPOSITORY = CompanionContractsRepository()
    return _REPOSITORY


def reset_companion_contracts_for_tests(db_path: Path | None = None) -> None:
    global _REPOSITORY
    _REPOSITORY = CompanionContractsRepository(db_path=db_path)
