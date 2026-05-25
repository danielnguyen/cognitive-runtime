# cognitive-runtime

Cluster 7.5 keeps this repo as an R40 runtime-state boundary scaffold only.

The runtime-state boundary may eventually own temporary, inspectable interaction state such as active scene, interaction mode, temporary task state, reset semantics, and trace references. That state must remain separate from canonical memory in `basic-memory-store` and from prompt assembly in `chat-orchestrator`.

Out of scope for Cluster 7.5:

- service runtime or API server
- state machine implementation
- worker or persistence layer
- full conversational runtime engine
- Phase 3 R41/R42 live-state, turn negotiation, timing, pause, backchannel, or interruption behavior
