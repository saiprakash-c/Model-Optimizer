# .fork/

Fork-only files for agents that work on this fork. They live on the `agent-main` branch and never go upstream.

| Branch | Contents |
|--------|----------|
| `main` | Exact mirror of `NVIDIA/Model-Optimizer` `main` |
| `agent-main` | `main` + our docs: this folder, module `README.md` files, `docs/architecture.md`, skills |

| File | Contents |
|------|----------|
| `known_failures.md` | Failures on Thor that are not bugs |
| `pre-commit-ours.yaml` | Our extra checks. They run after the repository's own `pre-commit` checks. |

- Task branches start from `agent-main`. Upstream PRs come from `pr/<task>`, built on `main` with code commits only.
- Sync: `main` is pulled from upstream (fast-forward). `main` is merged into `agent-main`.
- Design: `saiprakash-c/agent-docs`, `design/agent_first_workflow.md` section 2.17.
