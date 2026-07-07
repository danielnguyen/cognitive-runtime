# CCP Agent Instructions

## Mandatory bootstrap

Before trusting local files, repository evidence, or a commit SHA named by a prompt:

- synchronize this repository and the sibling `../projects` repository from `origin`;
- update only the declared base branches by fast-forward;
- stop on unrelated local changes, fetch failure, unsafe divergence, or an unavailable remote baseline;
- verify every referenced SHA in the repository where it is claimed to exist;
- when a prompt says a SHA is merged into or contained by a branch, verify that it is reachable from that remote branch;
- stop before editing or implementation when any verification fails. Do not proceed from stale local state.

Then read and follow:

- `../projects/process/repository-sync-preflight.md`
- `../projects/AGENTS.md`

These canonical files define the complete procedure and CCP workflow. If either cannot be loaded after synchronization, stop and report the missing governance source.

## Minimal execution containment

`../projects/AGENTS.md` is the canonical CCP workflow source. Its Minimal CCP Execution Protocol is binding in this repository.

Local containment rules:

- Do not create new planning artifacts, execution briefs, inventory docs, cleanup plans, scripts, or Make targets unless the user explicitly asks for them or the allowed-file list names them.
- Do not introduce wave, phase, cluster, packet, pass, or requirement labels into production runtime code as architecture, symbols, trace/API fields, or provider-visible wording.
- Prefer bounded edits to existing files over adding files. If a new file or broader file scope seems necessary, stop and report why before editing.
- Do not rename historical replay, smoke, or test taxonomy unless the task explicitly asks for that cleanup.
- Treat allowed files, forbidden files, validation commands, and explicit non-goals in the user/Codex task as hard constraints.
