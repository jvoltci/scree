# Contributing to scree

Thanks for considering a contribution. scree is small enough that any change matters; here's how to land one.

## Before you write code

For anything beyond a typo fix, **open a GitHub Discussion first**: <https://github.com/scree-dev/scree/discussions>. One or two paragraphs is enough. Tell us:

- What you want to change
- Why (the problem you're hitting)
- Roughly how (the API you have in mind)

The maintainers will respond with whether it fits scree's scope and what the design should look like. This is the cheapest way to avoid sunk-cost rewrites.

If your change is a clear bug fix with an obvious solution, skip the discussion and open a PR — the bar there is "is this actually a bug" and "is the fix surgical."

## Setting up

```bash
git clone https://github.com/scree-dev/scree
cd scree
python -m venv .venv
.venv/bin/pip install -e ".[dev,torch,mlx]"
.venv/bin/pytest tests/ -v
```

You should see all tests pass (31 as of v0.0.x). On macOS without MLX or CUDA without torch, some tests skip gracefully.

## The bar for code

scree is intentionally small. The package philosophy lives in [CLAUDE.md](CLAUDE.md):

- **Think before coding.** Surface tradeoffs. If unsure, ask.
- **Simplicity first.** Minimum code that solves the problem.
- **Surgical changes.** Touch only what you must.
- **Goal-driven.** Tests pass before and after.

Concretely:

- New code lives in `src/scree/`. Tests live in `tests/`. Docs live in `docs/`.
- Every public function has a docstring with parameters and return type.
- Every behavior change has a test. Bug fixes start with a test that fails before the fix.
- Lint with `ruff`, format with `black` (loose: 100 cols). If your editor argues with the existing style, match the existing style.

## What to work on

The most impactful contributions in v0.0:

1. **JAX backend** — the most-requested feature. See [docs/architecture.md](docs/architecture.md) → "Adding a new backend" for the template (MLX integration in commit `4876a21`).
2. **Triton backward pass** for `varlen_attention`. Forward is in [`src/scree/kernels/triton/varlen_attention.py`](src/scree/kernels/triton/varlen_attention.py).
3. **More reference kernels** — `varlen_rmsnorm`, `varlen_rope`, `scatter_add`/`gather`. See [docs/kernels.md](docs/kernels.md) → "Adding a new optimized kernel" for the recipe.
4. **HF Transformers PR** — wire scree as an optional backend for one model class (likely Llama or Mistral). See [docs/bridges.md](docs/bridges.md) for the integration point.
5. **vLLM / SGLang integration sketch** — convince the inference engines to accept `scree.Array` as a batch format.

If you want something smaller to start: fix any of the open issues marked `good first issue` on GitHub.

## The PR flow

1. Fork and branch from `main`.
2. Make the change. Keep it focused; one PR = one logical change.
3. Add or update tests. They must pass.
4. Add or update docs. If you changed the public API or behavior, this is mandatory.
5. Run `pytest tests/ -v` locally and confirm 31+ passing (or whatever the current count is).
6. Push and open a PR. Include in the description:
   - What changes and why (link to the Discussion if you opened one)
   - How you tested it
   - Any non-obvious tradeoffs
7. CI runs on Python 3.10/3.11/3.12 on Ubuntu + 3.11 on macOS. It must be green before merge.

## Commit message style

Follow the convention used in the repo:

```
<area>: <one-line summary, imperative mood>

<empty line>

<body explaining what changed and why, wrapped at ~72 cols>
```

`<area>` is one of: `core`, `bridges`, `kernels`, `tests`, `docs`, `benchmarks`, `triton`, `mlx`, `ci`. For multi-area changes, use the most prominent one. Examples from the history:

```
v0.0.1: core Array primitive + reference varlen kernels
bridges: torch.nested, HF padded, cross-framework via DLPack
mlx: third backend, Apple Silicon native via Metal
triton: varlen_attention kernel + Modal-hosted H100 benchmark
modal bench: first GPU validation — 1.21x of FA-2 varlen, PASS correctness
```

Never use `git commit --amend` on commits that have been pushed. If you need to fix a pushed commit, push a follow-up commit.

## Don't

- Don't add backwards-compat shims for in-flux internal APIs. Anything in v0.0 is allowed to break.
- Don't add a new optional dependency without strong justification. Each one adds install friction.
- Don't refactor unrelated code. Surgical changes.
- Don't add comments that re-state what the code does. Comment the *why* when it's non-obvious; leave the *what* to the code.
- Don't add features beyond what was asked. If your PR description says "fix X," the diff should be about X.

## Reviewing other people's PRs

If you have time and the maintainers haven't reviewed a PR yet, dive in. We don't have a "approved reviewer" gate — useful reviews are useful. Be specific. Reference line numbers. Suggest alternative code when you think something's wrong.

## Governance

For v0.0 through v0.5, scree is a BDFL project (the author of the original commits has merge authority). Starting v0.5 or v1.0 (whichever comes first), the project moves to a formal RFC process via GitHub Discussions for any behavior-changing PR.

The maintainers are listed in [`pyproject.toml`](pyproject.toml). Contact: open an issue or a discussion.

## License

scree is Apache-2.0. By contributing, you agree that your contribution will be licensed under Apache-2.0. No CLA is required.

## Thank you

Genuinely — every PR, issue, benchmark, doc fix is what makes this go from "one person's weekend code" to a primitive the field actually uses.
