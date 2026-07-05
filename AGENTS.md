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
