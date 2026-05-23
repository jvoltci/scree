# Release process

How to cut a release of `scree`. Owner: maintainer with merge access.

## Versioning

scree follows [SemVer](https://semver.org/):

- **v0.0.x** — pre-alpha; the API may break between any two commits.
- **v0.1.0** — first beta release; the public API in `scree.*` becomes stable. Backwards-incompatible changes after this require a minor version bump.
- **v1.0.0** — first stable release. Backwards-incompatible changes require a major bump.

## v0.1.0 release checklist

This is the gate from "pre-alpha source repo" to "library people can `pip install` and depend on."

### 1. Code readiness

- [ ] All 56+ tests passing on CI (Python 3.10/3.11/3.12 × Ubuntu + 3.11 × macOS)
- [ ] No `# TODO` or `# FIXME` comments in `src/scree/`
- [ ] Triton autotune unblocked (either Triton 3.1+ image or empirical safe-list of configs)
- [ ] Triton backward pass for `varlen_attention` validated on H100 with PASS correctness vs FA-2
- [ ] Modal benchmark autotuned result documented (target: ≤1.0× of FA-2 varlen)
- [ ] No backend-specific imports at module level — all imports lazy inside functions

### 2. API surface frozen

- [ ] Final review of [`src/scree/__init__.py`](src/scree/__init__.py) — anything not exported here is private and may break in v0.2
- [ ] Final review of [`src/scree/bridges/__init__.py`](src/scree/bridges/__init__.py) — same
- [ ] Public API documented in [`docs/api.md`](docs/api.md), every function with signature + return + invariants
- [ ] Type hints present on every public function

### 3. Documentation

- [ ] [`README.md`](README.md) reflects v0.1.0 status (status table updated, "pre-alpha" removed)
- [ ] [`docs/`](docs/) tree complete: getting-started, concepts, api, bridges, kernels, architecture, benchmarks, faq
- [ ] [`CHANGELOG.md`](CHANGELOG.md) has a real v0.1.0 entry with all changes since v0.0.1
- [ ] All examples in [`examples/`](examples/) run cleanly on a fresh `pip install`
- [ ] [`CONTRIBUTING.md`](CONTRIBUTING.md) reflects the current contribution flow

### 4. Benchmarks

- [ ] [`benchmarks/bench_memory.py`](benchmarks/bench_memory.py) reproduces the README's memory-savings table
- [ ] [`benchmarks/bench_throughput.py`](benchmarks/bench_throughput.py) reproduces the README's CPU throughput table
- [ ] [`benchmarks/modal_bench.py`](benchmarks/modal_bench.py) reproduces the README's GPU parity table
- [ ] Benchmark numbers in README and `docs/benchmarks.md` are within fp tolerance of the latest run

### 5. Packaging

- [ ] [`pyproject.toml`](pyproject.toml) `version = "0.1.0"`
- [ ] Package builds cleanly: `python -m build` produces a wheel + sdist
- [ ] Install from the wheel into a fresh venv and run all examples
- [ ] Install from the sdist into a fresh venv and run the test suite
- [ ] Long-description in `pyproject.toml` matches `README.md`
- [ ] `LICENSE` is included in the sdist (verified via `tar tf`)

### 6. Pre-launch credibility

- [ ] DM list of 8–10 named individuals contacted with API critique + benchmark sanity check
  - Tri Dao (FlashAttention)
  - Horace He (PyTorch, FlexAttention)
  - Edward Yang (PyTorch internals)
  - Driss Guessous (FlexAttention)
  - Patrick von Platen / HF team
  - vLLM core (Woosuk Kwon, Simon Mo)
  - SGLang core (Lianmin Zheng)
  - Phil Wang (lucidrains)
- [ ] 2–3 public endorsements lined up (quote tweet or comment on launch day)
- [ ] Launch blog post drafted on `scree.dev`, reviewed by 2 outside readers

### 7. Repo hygiene

- [ ] `main` branch protected; no direct pushes
- [ ] Issue templates for bug reports and feature requests
- [ ] Pull-request template references CONTRIBUTING.md
- [ ] CODEOWNERS or maintainers file
- [ ] Discord / Discussions enabled, link in README

### 8. Release artifacts

- [ ] Git tag `v0.1.0` on the release commit
- [ ] GitHub release with auto-generated changelog
- [ ] PyPI upload via `twine upload dist/*` (after `python -m build`)
- [ ] Verify `pip install scree==0.1.0` works on a fresh machine

## Launch day sequence (after release artifacts land)

Pacific time:

- **05:00** — GitHub repo flips public (if not already), v0.1.0 tag pushed
- **06:00** — Blog post published on scree.dev
- **06:15** — X thread with the headline benchmark plot
- **06:30** — Show HN submission with technical depth
- **07:00** — Email + DM the pre-launch endorser list asking for RTs
- **08:00** — Reddit r/MachineLearning thread
- **Throughout** — Respond to every HN comment / X reply / GitHub issue within 1 hour

## Hotfix releases (v0.1.x)

Any patch that fixes a bug without changing the public API:

1. Branch from the v0.1.x tag (or main if main is still on v0.1.x)
2. Fix the bug with a test that fails before the fix and passes after
3. Bump the patch version in `pyproject.toml`
4. Add a CHANGELOG entry under the new version
5. Tag and release via the same PyPI flow above

## Pre-release checks (a script)

A `scripts/check_release.py` could automate steps 1, 2, 4 mechanically — listed in PLAN.md as a non-GPU follow-up. Until that exists, the checklist is manual.
