# Launch-day runbook

A minute-by-minute checklist for the v0.1.0 (or v0.0.1 first-public)
launch of scree. Tighter than `RELEASE.md` — this is the actual
sequence with the exact commands.

All times Pacific. Cluster on a Tuesday or Wednesday for max attention
(Mon = catch-up, Thu = pre-weekend dropoff).

---

## T-24h (day before launch)

- [ ] Final test run on a clean machine: `pytest tests/ -v` shows 68/68 passing
- [ ] Final benchmark runs: `python benchmarks/bench_memory.py` and `python benchmarks/bench_throughput.py` match the numbers in README
- [ ] Re-read the launch blog post (`docs/launch-blog-draft.md`) one more time
- [ ] Confirm 2+ endorsers are still good to RT/comment on launch day
- [ ] Schedule X/Twitter thread to auto-post at 06:15 PT (Tweetdeck, Typefully, etc)
- [ ] Pre-write the HN title and first comment in a text file you can paste
- [ ] Confirm Modal credit balance is healthy (anyone running benchmarks during launch day will burn some)

## T-2h (launch morning)

- [ ] Open these tabs:
  - GitHub repo settings
  - PyPI account
  - `news.ycombinator.com/submit`
  - X compose
  - r/MachineLearning new post page
  - GitHub Discussions
  - Email client with the endorser DM drafts open

- [ ] Sanity check that you're well-rested, fed, and have 4 hours blocked for responses

## T-0 (launch sequence)

### 05:00 PT — Make repo public + tag release

```bash
cd /Users/shivya/Documents/volt/v4/scree
git push -u origin master
git tag -a v0.0.1 -m "v0.0.1 — first public release"
git push origin v0.0.1
gh release create v0.0.1 --title "scree v0.0.1" --notes-file CHANGELOG.md
```

If repo isn't public yet:
```bash
gh repo edit jvoltci/scree --visibility public --accept-visibility-change-consequences
```

### 05:15 PT — PyPI upload

```bash
cd /Users/shivya/Documents/volt/v4/scree
rm -rf dist build
python -m build
twine check dist/*
twine upload dist/*   # will prompt for username (__token__) + password (the API token)
```

Verify install works from a fresh machine or fresh venv:
```bash
pip install scree==0.0.1
python -c "import scree; print(scree.__version__)"
```

### 05:30 PT — Enable docs deployment + Discussions

```bash
gh api -X POST /repos/jvoltci/scree/pages -f source[branch]=gh-pages -f source[path]=/
gh api -X PATCH /repos/jvoltci/scree -f has_discussions=true -f has_issues=true
```

Within ~2 min the CI workflow `docs.yml` deploys `docs/` to
`jvoltci.github.io/scree`. Verify the site renders before announcing.

### 06:00 PT — Blog post live

If using a separate scree.dev: deploy. If using GH Pages:
`docs/launch-blog-draft.md` should already be navigable at
`jvoltci.github.io/scree/launch-blog-draft/`.

### 06:15 PT — X thread

Use the draft in `docs/launch-blog-draft.md` as the source. The
opening tweet should be ONE sentence + the benchmark plot screenshot.

Suggested opener:
> "Just shipped scree v0.0.1 — a cross-framework ragged tensor primitive
>  for variable-length sequence data. One type, four backends, full Triton
>  forward + backward at 1.30x / 1.61x of FA-2 on H100. github.com/jvoltci/scree
>  Thread ↓"

Thread should have ~6 follow-up tweets:
1. The problem (5 incompatible solutions today)
2. The primitive (one values+offsets array)
3. Memory savings number (71-85%)
4. GPU number (1.30x forward, 1.61x train step)
5. Bridges (one line to migrate from HF)
6. Multimodal (interleaved patches in one Array)

### 06:30 PT — Show HN

Title: `Show HN: scree – a cross-framework ragged tensor primitive (1.6x of FA-2)`

First comment from you (paste prewritten):
> Author here. scree solves the "every team carries their own packed-batch
> representation" problem — torch.nested, RaggedTensor, FA cu_seqlens, vLLM
> packed batches, HF attention_mask all converge on the same packed layout
> internally; scree exposes it as one typed value.
>
> What's in v0.0.1: four backends (NumPy + PyTorch + MLX + JAX), full Triton
> forward + backward on H100, 71-85% memory savings vs HF padded, 68 tests.
> Apache-2.0. Modal benchmarks reproducible at ~$0.40/run.
>
> Happy to answer questions about the autograd path, the backward kernel
> implementation, or the multi-shape characterization.

### 07:00 PT — DM the endorser list

Use the drafts in `docs/endorser-dms.md`. Send manually (one per person
— canned DMs read as canned). Personalize the first line based on what
each person is currently working on.

### 08:00 PT — Reddit r/MachineLearning

Wait an hour after HN to avoid simultaneous-fire fatigue. Title:
`[P] scree: a cross-framework ragged tensor primitive`

Reddit prefers no clickbait, light formatting. Body should mention the
HN thread and link to it.

### 09:00 PT onward — Engage

- Respond to every HN comment within 60 min
- Respond to every X reply within 30 min
- Respond to every GitHub issue within 2 hours
- If you have to step away: post a "back in 30 min" tweet so people know

### 12:00 PT — Mid-day check

- Read HN sentiment. If something's getting pushback, address it directly
  in the thread (not in a tweet).
- If a high-profile person quoted/retweeted, send them a thank-you DM
  and engage on whatever angle they took.
- Pin the most insightful HN comment to the top.

### 18:00 PT — Wrap

- Save snapshots of the HN page, X thread engagement, GitHub stars
  count, PyPI download count for "first-day" metrics.
- Post a short "thanks everyone" message on X with key numbers.
- Triage GitHub issues into "next-up" / "v0.1" / "won't fix" labels.
- Plan tomorrow's follow-up content (a deeper-dive blog, e.g.,
  "how the FA-2 backward kernel works" or "why we shipped MLX from day 1").

## T+24h to T+1week

- Reply to anyone who DM'd / commented in the last 24h who didn't get
  a response yet
- Write the v0.1 milestone plan in GitHub Issues based on launch feedback
- Identify the biggest reasonable PR coming in and prioritize it
- Reach out to vLLM/SGLang teams with `docs/vllm-integration-sketch.md`
- Submit MLSys paper draft if the timing fits

## Failure modes & playbook

**If a critical bug ships:**
1. Write a minimal repro, post to your X thread acknowledging it
2. Fix in ~2 hours, push v0.0.2
3. Update the HN top comment with the fix info
4. Don't panic — bugs that get fixed within hours actually generate goodwill

**If nobody notices:**
1. The endorser DM list IS the safety net. If 2-3 people RT, you'll get
   the first wave of attention
2. After 24h with no traction, write a follow-up blog post about a
   specific technical detail (the backward kernel, the multimodal use
   case, the autotune saga)
3. Submit to other places: Reddit r/programming, Lobste.rs, lobby Tri
   Dao or others for a quote tweet

**If someone gets hostile:**
1. Don't reply same-day to angry people. Sleep on it
2. Always respond with technical substance, never tone
3. If they're right, say so publicly and fix it
4. If they're wrong, post the data
