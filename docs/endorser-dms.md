# Pre-launch endorser DM drafts

Eight individuals from the RELEASE.md pre-launch credibility list. Send
manually on launch morning (07:00 PT per runbook). DO NOT canned-blast
— each message has a personalized opener tied to what they're publicly
working on. The middle paragraph (the scree summary) can be reused, but
the open and the ask should be specific to the recipient.

Send via DM on X/Twitter first; if no profile or DMs closed, email is
the fallback (most of these are findable via lab pages or GitHub).

---

## Tri Dao (FlashAttention)

**Channel**: X (@tri_dao) or email via Princeton/Together.

> Hi Tri — huge fan of FA-2 / FA-3. Quick context: I just shipped scree
> v0.0.1, a cross-framework ragged tensor primitive that uses your
> cu_seqlens convention as its canonical offsets format. The Triton
> forward+backward kernel set is structurally a FA-2 port (we cite you);
> first-attempt H100 numbers are 1.30× of FA-2 forward, 1.61× full
> training step.
>
> No ask except: any thoughts on the type design would be huge (what
> would `scree.Array` need to be a useful pass-through for FA-3 or future
> kernels?). And if the project is worth a RT on launch, I'd be in your
> debt.
>
> Code: github.com/jvoltci/scree
> Bench: 1.30× / 1.61× of FA-2 varlen on H100 (modal_bench.py reproducible)

---

## Horace He (PyTorch core, FlexAttention)

**Channel**: X (@cHHillee) or email via Meta.

> Hi Horace — your FlexAttention work + 2026 PyTorch internals posts have
> been my bedside reading while building scree v0.0.1 (a cross-framework
> ragged tensor primitive). One angle I'd love your read on: scree
> deliberately stays at the type layer and doesn't ship a custom
> attention API — the bet is FlexAttention + scree compose cleanly once
> FlexAttention can consume a `cu_seqlens`-flavored batch.
>
> Would a quick look at the type design (docs/api.md) be too much to ask?
> If FlexAttention's roadmap and scree are pointing the same way, the
> joint story would be compelling.
>
> github.com/jvoltci/scree

---

## Edward Yang (PyTorch internals, ezyang)

**Channel**: X (@ezyang) or his blog comments.

> Hi ezyang — your 2026 DTensor / sharding-type-system posts crystallized
> a thing I've been chewing on for months: that the field has
> reinvented packed-batch / ragged-array types incompatibly five
> different ways. scree v0.0.1 (just shipped) tries to be the neutral
> primitive between PyTorch (torch.nested), JAX (no ragged story), HF
> (padded+mask), and FlashAttention (cu_seqlens convention).
>
> I'd value your read on the type's invariants in particular. Bridges
> are bit-exact round-trip with HF padded, zero-copy with cu_seqlens.
>
> github.com/jvoltci/scree (docs/concepts.md has the design rationale)

---

## Driss Guessous (FlexAttention, PyTorch)

**Channel**: X (@driss_guessous) or GitHub.

> Hi Driss — scree v0.0.1 just shipped, a cross-framework ragged tensor
> primitive. The integration story I'd love your read on: scree could
> be the typed batch that FlexAttention consumes (its `cu_seqlens` is
> literally scree's `offsets`). One converter per FlexAttention kernel,
> ~10 LOC.
>
> Sketch in docs/vllm-integration-sketch.md; happy to flesh out a
> proper RFC if there's interest.
>
> github.com/jvoltci/scree

---

## Patrick von Platen (HuggingFace)

**Channel**: X (@PatrickPlaten) or HF GitHub.

> Hi Patrick — quick heads-up on scree v0.0.1 which just shipped. It's a
> cross-framework ragged tensor primitive that bit-exact round-trips
> with HF's (hidden_states, attention_mask) convention via a single
> bridges.from_hf_padded / to_hf_padded function call.
>
> The pitch for HF Transformers: scree gives users a 70-85% memory
> savings by dropping the padding tokens that the default attention
> path still allocates. Wiring scree as an optional backend for one
> model class (Llama or Mistral) would be ~50 LOC; sketched in
> examples/04_hf_compat.py.
>
> Any chance you'd entertain a draft PR? Either way, would love your
> thoughts on the type design.
>
> github.com/jvoltci/scree

---

## Woosuk Kwon (vLLM)

**Channel**: X (@woosuk_k) or vLLM Slack.

> Hi Woosuk — congrats on the vLLM v1 work this year. scree v0.0.1 just
> shipped, a cross-framework ragged tensor primitive that uses the same
> packed `(values, cu_seqlens)` layout vLLM already does internally.
>
> The proposal in docs/vllm-integration-sketch.md is small: two
> converter functions in vllm/worker/scree_bridge.py (~10 LOC each,
> both zero-copy). Doesn't replace anything in vLLM — adds a typed name
> for the batch tensors so cross-engine code (e.g., a researcher who
> wants the same batch in scree+JAX-prototype before vLLM-prod) doesn't
> have to translate.
>
> Worth a discussion thread on the vLLM repo? Would love your team's
> read.
>
> github.com/jvoltci/scree

---

## Simon Mo (vLLM)

**Channel**: X (@simon_mo_) or vLLM Slack.

> Hi Simon — same context as above (scree v0.0.1, cross-framework ragged
> tensor primitive that aligns with vLLM's packed-batch convention).
> If a vLLM v0.x.y release could optionally accept a `scree.Array` as
> input, the typed-batch story across the OSS inference ecosystem gets
> a lot tighter. Sketch: docs/vllm-integration-sketch.md.
>
> Happy to draft the PR; would love your steer on whether the type
> shape is right.
>
> github.com/jvoltci/scree

---

## Lianmin Zheng (SGLang)

**Channel**: X (@lm_zheng) or SGLang GitHub.

> Hi Lianmin — SGLang's RadixAttention work has been on my mind while
> building scree v0.0.1 (a cross-framework ragged tensor primitive). The
> integration angle: scree.Array could be the typed batch SGLang exposes
> to users, with your radix-tree prefix cache treating scree views as
> first-class. The bridges in docs/vllm-integration-sketch.md are the
> minimal version (single converter), and there's a bigger v0.2
> conversation about whether SGLang's prefix cache could be reformulated
> on scree handles.
>
> Curious if any of this is worth a deeper conversation. Either way,
> launch-day RT would be appreciated if the direction resonates.
>
> github.com/jvoltci/scree

---

## Phil Wang / lucidrains

**Channel**: X (@lucidrains) or GitHub.

> Hi Phil — scree v0.0.1 just shipped, a cross-framework ragged tensor
> primitive (NumPy + PyTorch + MLX + JAX). Your model-implementation
> repos have been a north star for "clean, readable, single-file
> demos"; scree's examples/02_no_pad_transformer.py is shamelessly in
> that spirit (full transformer block in <100 lines on scree primitives,
> no attention_mask threading).
>
> If any of the repos in lucidrains/* would benefit from a varlen-native
> batch type, the bridges are zero-copy. Or just RTing the launch would
> mean a lot — your audience is exactly the right one for this.
>
> github.com/jvoltci/scree

---

## After they reply

If they respond positively, follow up with:

1. **A specific technical question** based on their answer — converts
   "thanks for the heads-up" into a real conversation
2. **An invite to a private GitHub Discussion** before public launch,
   so they can shape the launch direction
3. **A draft PR ready to submit** if they signal interest in
   integration — never make them write the integration glue themselves

If they respond negatively or skeptically:

1. **Don't argue** — say "fair point, here's the data" with a link to
   the specific benchmark or test
2. **Ask what would change their mind** — turn the skepticism into a
   roadmap item
3. **Don't push for endorsement** if they decline — a forced endorsement
   is worse than no endorsement
