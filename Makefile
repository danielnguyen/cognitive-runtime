SHELL := /usr/bin/env bash

.PHONY: dev-test dev-install dev-lint dev-check process-naming-check replay-test smoke dev-start dev-start-reload

dev-test:
	@cd api && ./.venv/bin/python -m pytest -q

dev-install:
	@cd api && ./.venv/bin/python -m pip install -r requirements.txt

dev-lint:
	@cd api && ./.venv/bin/python -m ruff check .

dev-check: dev-lint dev-test

process-naming-check:
	@./scripts/check_process_naming.py

replay-test:
	@cd api && ./.venv/bin/python -m services.runtime_replay

smoke:
	@set -euo pipefail; \
	CR_BASE="$${CR_BASE:-http://127.0.0.1:4371}"; \
	echo "==> GET $$CR_BASE/healthz"; \
	curl -sS "$$CR_BASE/healthz" | jq -e '.status == "ok" and .service == "cognitive-runtime"' >/dev/null; \
	echo "==> POST $$CR_BASE/v1/runtime/interaction-governance/evaluate (question)"; \
	curl -sS -X POST "$$CR_BASE/v1/runtime/interaction-governance/evaluate" \
	  -H "Content-Type: application/json" \
	  -d '{"request_id":"smoke-question","owner_id":"owner","conversation_id":"conv-smoke","surface":"dev","current_user_text":"What does this function do?","recent_messages":[]}' \
	  | jq -e '.result.interaction_kind == "question" and .result.action_allowed == false' >/dev/null; \
	echo "==> POST $$CR_BASE/v1/runtime/interaction-governance/evaluate (history follow-up candidate)"; \
	curl -sS -X POST "$$CR_BASE/v1/runtime/interaction-governance/evaluate" \
	  -H "Content-Type: application/json" \
	  -d '{"request_id":"smoke-history-followup","owner_id":"owner","conversation_id":"conv-smoke","surface":"dev","current_user_text":"What was that based on?","recent_messages":[],"history_followup_candidate":{"source":"deterministic","intent":"support_explanation","confidence":1.0,"target_mode":"immediate_previous","new_verification_requested":false}}' \
	  | jq -e '.result.interaction_kind == "question" and .result.history_followup_policy.status == "accepted" and .result.history_followup_policy.history_lookup_allowed == true and .result.history_followup_policy.new_verification_allowed_after_history_resolution == false' >/dev/null; \
	echo "==> POST $$CR_BASE/v1/runtime/interaction-governance/evaluate (tense debugging)"; \
	curl -sS -X POST "$$CR_BASE/v1/runtime/interaction-governance/evaluate" \
	  -H "Content-Type: application/json" \
	  -d '{"request_id":"smoke-tense","owner_id":"owner","conversation_id":"conv-smoke","surface":"dev","current_user_text":"I think I broke the server and prod is failing","recent_messages":[]}' \
	  | jq -e '.result.interaction_kind == "tense_debugging" and .result.response_posture == "tactical" and .result.humor_allowed == false and .result.commentary_allowed == false' >/dev/null; \
	echo "==> POST $$CR_BASE/v1/runtime/interaction-governance/evaluate (playful)"; \
	curl -sS -X POST "$$CR_BASE/v1/runtime/interaction-governance/evaluate" \
	  -H "Content-Type: application/json" \
	  -d '{"request_id":"smoke-playful","owner_id":"owner","conversation_id":"conv-smoke","surface":"dev","current_user_text":"lol roast my tiny todo list","recent_messages":[]}' \
	  | jq -e '.result.interaction_kind == "joke_or_playful" and .result.humor_allowed == true' >/dev/null; \
	echo "==> POST $$CR_BASE/v1/runtime/interaction-governance/evaluate (ambiguous destructive)"; \
	curl -sS -X POST "$$CR_BASE/v1/runtime/interaction-governance/evaluate" \
	  -H "Content-Type: application/json" \
	  -d '{"request_id":"smoke-ambiguous","owner_id":"owner","conversation_id":"conv-smoke","surface":"dev","current_user_text":"nuke this","recent_messages":[]}' \
	  | jq -e '.result.interaction_kind == "ambiguous" and .result.action_allowed == false and .result.requires_confirmation == true' >/dev/null; \
	echo "==> POST $$CR_BASE/v1/runtime/persona-containment/evaluate"; \
	curl -sS -X POST "$$CR_BASE/v1/runtime/persona-containment/evaluate" \
	  -H "Content-Type: application/json" \
	  -d '{"request_id":"smoke-persona","owner_id":"owner","conversation_id":"conv-smoke","surface":"web","current_user_text":"My car needs an oil change soon","recent_messages":[]}' \
	  | jq -e '.result.capability_domain == "vehicle_maintenance" and .result.cross_scope_access_allowed == false' >/dev/null; \
	echo "==> POST $$CR_BASE/v1/runtime/restraint/evaluate"; \
	curl -sS -X POST "$$CR_BASE/v1/runtime/restraint/evaluate" \
	  -H "Content-Type: application/json" \
	  -d '{"request_id":"smoke-restraint","owner_id":"owner","conversation_id":"conv-smoke","surface":"dev","current_user_text":"give me the prompt","recent_messages":[]}' \
	  | jq -e '.result.restraint_policy == "short_answer" and (.result.domains | index("output")) != null' >/dev/null; \
	echo "==> POST $$CR_BASE/v1/runtime/turns/start"; \
	TURN="$$(curl -sS -X POST "$$CR_BASE/v1/runtime/turns/start" \
	  -H "Content-Type: application/json" \
	  -d '{"request_id":"smoke-turn-start","owner_id":"owner","conversation_id":"conv-smoke","surface":"dev","input_message_id":"msg-smoke"}')"; \
	RTSESSION="$$(echo "$$TURN" | jq -r '.runtime_session.runtime_session_id // empty')"; \
	RTTURN="$$(echo "$$TURN" | jq -r '.runtime_turn.runtime_turn_id // empty')"; \
	test -n "$$RTSESSION"; \
	test -n "$$RTTURN"; \
	echo "==> POST $$CR_BASE/v1/runtime/interaction-governance/evaluate (turn integration)"; \
	curl -sS -X POST "$$CR_BASE/v1/runtime/interaction-governance/evaluate" \
	  -H "Content-Type: application/json" \
	  -d '{"request_id":"smoke-turn-governance","owner_id":"owner","conversation_id":"conv-smoke","surface":"dev","runtime_session_id":"'"$$RTSESSION"'","runtime_turn_id":"'"$$RTTURN"'","current_user_text":"rename this variable to count","recent_messages":[]}' \
	  | jq -e '.result.interaction_kind == "command"' >/dev/null; \
	echo "==> GET $$CR_BASE/v1/runtime/sessions/$$RTSESSION"; \
	curl -sS "$$CR_BASE/v1/runtime/sessions/$$RTSESSION" \
	  | jq -e '.latest_turn.intent_class == "action_command" and ([.events[] | select(.event_type == "interaction_governance_evaluated")] | length) >= 1' >/dev/null; \
	echo "Smoke passed."

dev-start:
	@cd api && ./.venv/bin/uvicorn main:app --host 0.0.0.0 --port "$${APP_PORT:-4371}"

dev-start-reload:
	@cd api && ./.venv/bin/uvicorn main:app --host 0.0.0.0 --port "$${APP_PORT:-4371}" --reload
