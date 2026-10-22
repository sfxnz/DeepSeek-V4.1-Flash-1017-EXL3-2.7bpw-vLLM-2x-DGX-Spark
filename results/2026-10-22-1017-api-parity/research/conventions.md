# API-parity campaign, track `conventions`: how to land changes, and how quality is measured

2026-10-22. Worktree `/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-parity` (branch `api-parity/explore`
= origin/main 7851999, tree left clean). Unless stated otherwise, file:line refs are into that worktree. Harness
scripts and logs: `/home/sfxnz/projects/data/dsv41-1017/api-parity/harness/conventions/`.

This track has no request-level divergences of its own. Section 4 is a table of gaps in the quality harness
rather than a divergence table. Sections 1-2 are the how-to for Phase 3.

## 0. Key points

- **The established way to change API-side behaviour** is a `docker/patch/*.py` source rewrite of the image's vLLM
  file. `sitecustomize` applies it in every process at interpreter start. The precedent is `chat_encoder_1017.py`
  (required, no env). It reaches both nodes with no image rebuild, because run.sh bind-mounts `docker/patch` and
  scp's it to the worker.
- **The conventions favour the pure-Python route over a new image with the deepseek-recipe wheel.** Use
  deepseek-recipe as a reference oracle in tests and in an offline reference-data tool, not at serve time.
  The repo builds an image layer only for compiled kernels. It deliberately keeps a pip extra (`b12x`) out of
  the image, and a test checks that it stays out. A new tag touches 17 files, and spark2 cannot reach the
  internet. Section 2 has a pinned, hash-checked `Dockerfile.e17` sketch for the case where the runtime itself
  must call deepseek-recipe.
- **CONTEXT.md has two errors.**
  - `image-src/` is not pristine. `tokenizers/deepseek_v41{,_encoding}.py` already carry the
    `dsv41-chat-encoder-1017` rewrite (6 and 3 markers). The images bake it in, because build-time `RUN python3`
    steps execute the copied sitecustomize. Other build-time rewrites are also baked in. The env-gated boot-time
    rewrites are not.
  - CONTEXT's in-image docker command fails to import the wheel:
    `ImportError: libopencv_core-917defc8.so.412`. The fix is to also mount `deepseek_recipe.libs`.
- **The harness's own chat prompts are already at render parity with deepseek-recipe: 46/46 identical token-id
  lists.** That covers smoke, c2, the 12 selfcons goldens, the 30 tools30 items and GSM8K thinking on and off.
  Parity fixes that leave these request shapes alone do not invalidate the 1017 baseline or its goldens.
- **The quality harness covers numerics and simple single-turn task scores.** It covers none of these:
  streaming, reasoning-field parsing, multi-turn tool round trips, structured outputs (only an out-of-tree perf2
  script), live rendering parity, API-style request fields, default sampling, more than one image, and the
  Responses and Anthropic endpoints.
- **Baseline:** `python3 -m unittest discover -s tests -q` ran 1267 tests in 43.6 s, OK (skipped=49), the count
  PR 10's test plan states. `python3 kit/render.py --check` gives rc 0. The opt-in chat-encoder image test on
  e16-rows is OK, 26/26 checks.

## 1. How this recipe lands a change

### 1.1 Patch mechanics (`docker/patch/sitecustomize.py`)

- **Mounting.** run.sh mounts `docker/patch` at `/opt/dsv41-patch:ro` and `docker/patch/sitecustomize.py` at
  `/usr/lib/python3.12/sitecustomize.py:ro` on both ranks (`run.sh:581-587`). The head scp's the patch dir to the
  worker's `~/.cache/dsv41-patch` on every boot (`run.sh:861-862`). sitecustomize runs in every Python process of
  both containers (the API server, EngineCore, the workers) and also in image build `RUN python3` steps.
- **`_patch(name, fn, required=False)`** (`:33-53`) prints exactly one stderr line per patch:
  - `dsv41-patch ok NAME[: detail]` when fn returns.
  - `dsv41-patch skip NAME: why` when fn raises `_PatchSkip` ("env off", "file absent").
  - `dsv41-patch FAIL NAME: repr` for any other exception.
  - `_PatchAbsent` (the target file is missing) is a skip for an optional patch and a FAIL for a required one.
  - A required patch that FAILs calls `os._exit(1)` when `DSV41_PATCH_STRICT=1`, so the boot fails. run.sh
    forwards STRICT=1 to both ranks (`run.sh:112`; `test_patch_strict_is_forwarded_to_both_ranks`). Unset, as in
    docker build, a FAIL only prints.
  - A non-`Exception` (SystemExit) from an optional patch is re-raised. Anchor misses should raise `ValueError`.
- **`_rewrite(path, fn)`** (`:56-74`): returns `"already applied"` when `fn(src)` changes nothing. Otherwise it
  writes tmp + `os.replace` (atomic, mode kept) and returns `"rewritten"`.
- **Audit markers.** sitecustomize's own `LOG_DISARMED = ("dsv41-patch FAIL", " skipped: ")` (`:17`) makes every
  FAIL line and every legacy `dsv41: X skipped: ...` line a disarm. The `_patch` skip lines are not disarms.
- **Import state at that point.** After sitecustomize, 755 vllm modules are already imported, including
  `vllm.tokenizers.{registry,hf,protocol}` and `vllm.reasoning.abs_reasoning_parsers`. These are not imported yet:
  `vllm.tokenizers.deepseek_v41*`, the deepseek_v41 reasoning and tool parser modules, and
  `vllm.entrypoints.openai`. Rewrites of those files take effect in the same process. Still copy
  `kv_packed_groups.install`'s guard: raise if the module is in `sys.modules` after `"rewritten"`
  (`kv_packed_groups.py:333-345`).

### 1.2 Engagement audit (`tools/engagement_audit.py`, `AUDIT=` in run.sh)

- **When it runs.** After `/v1/models` is up, run.sh saves the docker logs of the head and of the worker (over ssh)
  to `.run-state/audit-*.log`. It runs the audit with every FORWARD_ENVS value and `ENFORCE_EAGER` in the env,
  plus `--pack-config <snapshot>/config.json` (`run.sh:784-805`).
- **Modes.** `AUDIT=warn` is the default and exits 0. `strict` exits 1 ("The serve is up; ./stop.sh tears it
  down"). `off` skips. Campaign boots run strict.
- **`PATCHES`** (`:39-82`) maps a patch file to the env values that must all hold for its `LOG_ENGAGED` to be
  expected:
  - `{}` means always expected. `chat_encoder_1017.py` is `{}`.
  - `None` means only the patch's disarm markers are collected.
  - A lever is gated on its recipe default only, for example `{"DSV41_SCHED_PREFILL_CAP": "2048"}`.
- **How markers are found.** The audit reads module-level `LOG_ENGAGED` / `LOG_DISARMED` constants with `ast`.
  Nothing is imported.
- **Every expected marker must appear in every rank's log** (`audit()`, `:190-204`). That is why API-side
  patches are sitecustomize source rewrites that print in every process: an API-server-only marker would fail
  strict on the worker.
- **Tests** (`tests/test_engagement_audit.py:39-58`): every marker appears at least twice in its file (the
  constant and the print); every gated patch defines `LOG_ENGAGED`; every gate env name appears in run.sh.
- **Runtime disarms.** `tools/disarm_scan.sh` (ARMS step 6) catches levers that disarm after the audit.

### 1.3 Worked example 1: a required source rewrite (`docker/patch/chat_encoder_1017.py`)

- **Anatomy:**
  - Docstring with what, why and how it is tested.
  - Constants `MARK`, `LOG_ENGAGED`, `LOG_DISARMED`, `ENCODING_REL`, `TOKENIZER_REL`.
  - Verbatim `*_OLD` / `*_NEW` anchor pairs. Each `_NEW` embeds the MARK comment.
  - `_sub` requires exactly one match and raises `ValueError` otherwise.
  - The `patch_*_source` functions are idempotent (they return src when `MARK in src`).
  - `install(vllm_root, rewrite)` rewrites both files and prints the engaged or FAILED line on stderr.
  - Wiring: `_patch("chat_encoder_1017", lambda: __import__("chat_encoder_1017").install(_VLLM, _rewrite),
    required=True)` (`sitecustomize.py:669-672`).
- **Host tests** (`tests/test_chat_encoder_1017.py`, stdlib):
  - `RewriteTests` on the pristine e14 pins (`tests/fixtures/dsv41_tokenizer{,_encoding}_e14.pin.py`): apply
    once, idempotent, compiles; an anchor miss raises; ordering; install logs engaged, then FAILED on a drifted
    file; the sitecustomize line is present verbatim and required.
  - `VectorTests` (skipped without the HF 1017 snapshot): the 5 `encoding/tests` golden vectors plus the 1017
    behaviours, and a negative control in which the unpatched encoder must fail.
  - Both share their checks with `tests/chat_encoder_1017_image_check.py`.
- **Image test** (opt-in: `DSV41_IMAGE_TESTS=1 DSV41_IMAGE=...`). It runs `docker run --network none` with the
  patch dir and sitecustomize mounted as run.sh mounts them, and runs the check script, which prints
  `PASS/FAIL <check>` and `N/M passed`. The test asserts rc 0, `LOG_ENGAGED`, `PASS serving vector 1` and no FAIL.
  - Today on e16-rows: 26/26 (`logs/chat-encoder-image-check-e16.log`).
  - The test's default image is still `e15-kbits` (`:35`). AGENTS.md's Verify section passes the e16-rows tag.
- **Docs.** A flags.md row (`flags.md:99`, with "26/26; unpatched control 12/26"), README.md:11 (efforts
  low 50 / high 75 / max 100, `xhigh` rejected) and AGENTS.md. There is no env, no recipe.yaml entry and no
  FORWARD_ENVS entry.

### 1.4 Worked example 2: an env lever (`docker/patch/sched_prefill_cap.py` + `tests/test_sched_prefill_cap.py`)

- **Module.**
  - Top-level imports are stdlib only (a test asserts the set).
  - `ENV_CAP` / `ENV_STEPS` constants name the env vars.
  - `knobs()` raises `ValueError` on typos.
  - `CONTRACT` holds verbatim lines of the image's `Scheduler.schedule`.
  - `install()` returns `off`, `armed`, `already installed` or `refused`; on `refused` it prints `LOG_DISARMED`
    and leaves the code stock. sitecustomize maps `off` to a skip and anything unexpected to FAIL
    (`sitecustomize.py:1378-1393`).
- **Wiring, as listed in commit 03cc231:** sitecustomize (after decoder_replay), FORWARD_ENVS `NAME=` (empty), the
  recipe.yaml serve.env default (which renders the run.sh line and a README row), an export in the promoted boot,
  the audit gate at the recipe value, an AGENTS.md line and a flags.md row.
- **Tests.**
  - The image source is pinned verbatim (`tests/fixtures/sched_e15.pin.py`) and driven with fakes like the
    EngineCore.
  - Behaviour tests, contract-drift tests and knob tests.
  - `test_markers_and_wiring`: each marker appears exactly twice; the sitecustomize line and its order; the audit
    gate literal; run.sh's `  NAME=\n` and `NAME="${NAME:-default}"`.
  - `ImageTests` (skipped unless the image's vLLM imports): the pin equals `inspect.getsource`, and install is
    idempotent.

### 1.5 Checklist for a new API-parity patch

1. **Patch file.** Add `docker/patch/<name>.py` as in 1.3.
   - Env-gated rewrites also need `enabled()`, and either `refuse_stale` (`kv_packed_groups.py:347-353`) or
     `restore`/`unpatch_source` (`model_loop_release.py:96-140`). Images bake build-time rewrites, so a container
     can already carry the rewritten text when the env is 0.
   - Top-level imports stay stdlib only.
2. **Order.** Add a `_patch(...)` line in sitecustomize. Tokenizer-file rewrites go after `chat_encoder_1017`, and
   their anchors must match the post-1017 text. Build the fixture by chaining the e14 pin, the 1017 rewrite and the
   new patch; never pin an e16 file, which is already rewritten.
3. **Audit.** Add the `PATCHES` entry: `{}` for required, or `{ENV: recipe value}`.
4. **Env.** Every env the file reads goes into FORWARD_ENVS (`run.sh:110-239`). Read env through literal names or
   module-level `ENV* = "X"` constants: the checker regex is `tests/test_recipe_ops.py:166-172`, and the tests
   `test_every_patch_env_read_is_forwarded` and `test_forwarded_names_are_read_by_something` sit at `:195-207`.
5. **Default-on lever.** Add the `recipe.yaml serve.env` entry (order must match the generated block) and a
   `readme.defaults` row, then run `kit/render.py`. Export it in the promoted boot:
   `test_defaults_match_promoted_boot` (`:795-847`) requires every `DSV41_*` default there. A new campaign may add
   a new promoted boot and move `PROMOTED_BOOT` (`:75`).
6. **vLLM CLI flag.** It goes in `start_local`'s shared argument list. Head and worker args must be identical from
   `--max-model-len` on (`:220-234`). A configurable value gets a recipe knob plus `worker_config`
   (`run.sh:835-842`). `EXTRA_ARGS` carries one-off flags (the test's own example is
   `--override-generation-config`).
7. **Tests.** Host stdlib tests on pins. Torch/vLLM tests run CPU-only in the image from `tests/` with plain
   module names. Docker tests are opt-in with `DSV41_IMAGE_TESTS=1`.
8. **Docs.** A flags.md row (What / Why / Effect + evidence, verdict in bold), an AGENTS.md working-rule line,
   README prose if users see it, and the evidence files under `results/` before any doc cites them.
9. **Verify.** `python3 -m unittest discover -s tests -q`, `python3 kit/render.py --check`, and
   `python3 tests/run_sh_harness.py [NAME=VALUE]` (prints both ranks' `docker run`). Then a TP=2 boot with
   `AUDIT=strict`, the smokes, and `quality_eval --quick --baseline .../quick.json`.

### 1.6 recipe.yaml + kit/render.py

- `recipe.yaml` holds `model`, `image` (`base`, `digest`, `local_tag`, `build`), `serve.env` (exactly the run.sh
  generated defaults, in order; `kit/render.py:58-77` exits otherwise), `readme.defaults` (rows with `{NAME}`
  placeholders) and `measured.decode.rows` (with evidence paths).
- `render.py` rewrites only the marked blocks in run.sh and README. `--check` prints a diff and exits 1;
  `--strict` also requires every evidence file. The kit is vendored from sfxnz/forge @3d294ff.
- **Free text is synced by hand.** Nothing tests AGENTS.md or model-card.md content except a few literals.
  - README.md:3 still names `e15-kbits`, which is drift.
  - model-card.md is the HF pack card (`tools/publish_pack.py:25`); it changes in the last PR of a stack.
  - `tools/corpus.py:42` builds bench prompts from README / AGENTS / model-card / flags / run.sh / sitecustomize
    text, so doc edits change the bench documents.

### 1.7 flags.md and results/

- **flags.md** has tables per area with What / Why / Effect, bold verdicts followed by numbers and evidence paths,
  and a round log at the end. ARMS.md holds the protocol: one lever per boot, ABAB for perf, quick quality before
  any KEEP, `--full` for levers that change numerics.
- **`results/`** holds one dated campaign directory per campaign, each with a README index: `2026-10-18-1017/`
  (release) and `2026-10-19-1017-perf2/` (one folder per boot).
  - Each boot folder has `command.txt` / `boot-*.sh`, `boot.txt`, `boot-grep-*.txt`, `numbers.{txt,json}`, cell
    files and `quick`/`full.json`.
  - `.log` is gitignored, so evidence is renamed `.txt`. No traces or container logs are committed.
  - The promoted boot `results/2026-10-19-1017-c4/boot.sh` exports the defaults; its header lists what each PR
    changed.

### 1.8 Commit and PR style

- **Commits.** One squashed commit per PR.
  - Subject: `<campaign>: <imperative>[, on|off by default]`.
  - Body: a problem paragraph with "Squash of <branch> @sha", file-level bullets, then "Wiring:",
    "Knobs (forwarded):", "Exactness:", "Evidence: results/...", and the Co-Authored-By trailer.
- **PRs.**
  - Title: `<campaign> (N/M): <subject>`.
  - The body opens with `<!-- draft: branch pr/NN-<slug> @sha, base pr/(N-1)-<slug>; squash of ...; sources:
    ... -->`.
  - Sections: What, Why, Knobs and defaults (table: Knob | Default | Off / notes), Exactness, Evidence (then
    caveats), Test plan (unittest with the branch's count, render `--check`, image CPU tests, then
    `SKIP_DOWNLOAD=1 AUDIT=strict ./run.sh`, a `docker logs` grep of the markers, the smokes and quality_eval),
    and Depends on.
  - Footer: "Generated with [Claude Code](https://claude.com/claude-code)" with the robot emoji.
- **Stack (perf2).** `pr/01-evidence` .. `pr/10-c4-sched` form a linear chain off main @20ee8ba, each PR opened
  against the previous PR's branch.
  - The evidence PR comes first, so flags.md citations resolve.
  - The last code PR also moves the stack-level docs: README intro and Measured via recipe.yaml, the model card,
    AGENTS.md.
  - Merged 2026-10-20 10:37-10:38 in order. Each PR was retargeted to main seconds before merging (PR #2:
    `base_ref_changed` 10:37:31, `merged` 10:37:35), using merge commits. Head branches were deleted afterwards;
    the remote now has only main.
- **Repo rules.** Ruleset "Protect main" (since 2026-10-20): PR required, 0 approvals, review threads resolved,
  no force-push or deletion. There is no CI.
- **AGENTS.md rule:** work in a worktree, never in the checkout a running serve was started from.

### 1.9 Gotchas found while checking (evidence)

- **The images bake build-time rewrites.** `docker run --entrypoint grep <img> -c dsv41-chat-encoder-1017` finds
  6 / 3 matches in e16-rows, e15-kbits and image-src alike. `grep -rl dsv41` in image-src also finds the indexer,
  engram, sm120 and o_proj rewrites. At boot the image check logs "chat encoder 1017 parity engaged (encoding
  already applied, tokenizer already applied)".
- **API-server-only behaviour** must print its marker from sitecustomize, because the audit checks every rank.
- **Any `... skipped: ...` line** counts as a disarm.
- **The promoted-boot test** forces every new `DSV41_*` default into `boot.sh`.

## 2. Getting a new pip dependency into the image

### 2.1 How images are built, pinned and selected today

- **Base.** `docker/Dockerfile:3` starts `FROM vllm/vllm-openai:deepseekv41-flash-0909@sha256:d84a1232...`. The
  digest is also in `recipe.yaml image.digest` and is asserted by `test_dockerfile_from_is_digest_pinned`
  (`test_recipe_ops.py:619-664`).
- **Upstream code is pinned as git refs in the RUN lines** (`ARG EXLLAMAV3_REF`, `ARG VLLM_EXL3_REF`). It is
  cloned at build and installed with `pip install --no-build-isolation --no-deps`, and each stage ends with a
  `python3 -c "...asserts...; print('... ok')"` build assert.
- **No pip package has been added at serve time.** The `b12x` pip extra is deliberately left out: with it, vLLM
  routes MXFP8 through the package and prose decoding was 5-9% slower (`docker/Dockerfile:72-76`). A test asserts
  it stays out (`test_recipe_ops.py:645-649`).
- **Two chains, kept in sync by tests.**
  - The layered chain: `Dockerfile.e14` on `canonical-e13`, `e15` on `canonical-e14` (tag `e15-kbits`), `e16` on
    `e15-kbits` (tag `e16-rows`).
  - The from-scratch chain: `docker/Dockerfile` builds the same e16-rows.
  - `tests/test_widen_p2b_dataflow.py:263-290` and `test_widen_p2b_rows.py:259-282` require the same stages, the
    same asserts and the same `LABEL dsv41.recipe.patches` in both, plus the provenance labels (`DSV41_GIT_REV`,
    `DSV41_TREE_DIFF` build-args).
- **Tag selection.**
  - `recipe.yaml image.local_tag` feeds `serve.env.IMAGE`, which renders run.sh's `IMAGE` default. The worker gets
    it through `worker_config`.
  - `ensure_image` (`run.sh:433-443`) runs on both nodes in preflight. It requires the tag locally (it never
    pulls) and only warns when the label is missing. Nothing compares the image IDs of the two nodes.
  - The tag literal appears in 17 tracked files, excluding the perf2 evidence folder:
    - `recipe.yaml`, `run.sh`, `docker/Dockerfile`, `docker/Dockerfile.e16`;
    - `README.md`, `AGENTS.md`, `ARMS.md`, `flags.md`, `model-card.md`;
    - `results/2026-10-19-1017-c4/boot.sh`;
    - `tests/test_recipe_ops.py` (`IMAGE_TAG`), `tests/test_moe_prefill_tc.py`;
    - `tools/nccl_{decode_sweep,dualrail,twin_check}.sh`;
    - `kernel_study/p2b_rows/{README.txt,bench_rows.py}`.
- **Build.** `docker build -f docker/Dockerfile -t ...:e16-rows docker`, or the layered build with
  `--memory 24g --memory-swap 24g` and the provenance build-args. The build context is `docker/`.
- **How the worker really gets the image.** spark2 has no route to github.com: its chain-A build failed with
  "Could not resolve host". It received e16-rows by `docker save | ssh spark2 docker load`, which streams the
  whole 23.7 GB image and aborts below 40 GB free (`results/2026-10-19-1017-perf2/image-builds/
  transfer-e16-spark2.{sh,txt}`; spark2 was 98% full with 78 GB free). PR #3 says: "the recipe's path is still to
  build on each node".

### 2.2 Facts about the deepseek-recipe wheel (checked)

- **Wheel tag and dependencies.** `deepseek_recipe-0.2.0-cp310-abi3-manylinux_2_28_aarch64.whl` (maturin, MIT).
  The only `Requires-Dist` is `pytest` under the `test` extra, so there are no runtime dependencies.
- **Size.** 20 MB `_native.abi3.so` plus 11 MB of vendored OpenCV 4.12 libraries with hashed sonames. The library
  path is `/opt/opencv-min/lib:$ORIGIN/../deepseek_recipe.libs`.
- **It works in e16-rows** (Python 3.12.3, glibc 2.39). It imports next to the image's cv2 5.0.0 and renders and
  encodes with the pack's `tokenizer.json` (`wheel_in_image.py`, `logs/wheel-in-image.log`).
- **The served model name must be aliased.** With default options it rejects
  `deepseek-ai/DeepSeek-V4.1-Flash-1017` ("supported ... DeepSeek-V4-Pro, DeepSeek-V4.1-Flash"). It needs
  `ConversionOptions().with_model_alias(name, ModelVersion.DeepseekV41Flash)`.
- **No vLLM file references deepseek_recipe.** Installing it changes nothing until a patch imports it.
- **The archive sha256 is not available offline.** The uv cache holds the unpacked tree, not the `.whl`.

### 2.3 Minimal `docker/Dockerfile.e17` (sketch, if the runtime must import deepseek-recipe)

```dockerfile
# e17-recipe (API parity): e16-rows + deepseek-recipe 0.2.0 (MIT; Rust + vendored OpenCV 4.12), installed
# offline from the build context, version- and sha256-pinned. The wheel is not committed: fetch once on spark1
# (PyPI), copy docker/wheels/ to spark2 (no internet), then build on each node:
#   python3 -m pip download deepseek-recipe==0.2.0 --no-deps --only-binary=:all: --platform manylinux_2_28_aarch64 \
#     --python-version 3.12 --implementation cp -d docker/wheels
#   sha256sum docker/wheels/*.whl     # must equal the --hash in docker/wheels/requirements-e17.txt (tracked)
#   docker build --memory 24g --memory-swap 24g -f docker/Dockerfile.e17 -t dsv41-flash-exl3-sm121:e17-recipe \
#     --build-arg DSV41_GIT_REV=$(git rev-parse HEAD) \
#     --build-arg DSV41_TREE_DIFF=$(git diff HEAD --binary -- docker | sha256sum | cut -c1-64) docker
FROM dsv41-flash-exl3-sm121:e16-rows

COPY wheels/requirements-e17.txt wheels/deepseek_recipe-0.2.0-cp310-abi3-manylinux_2_28_aarch64.whl /tmp/wheels/
RUN python3 -m pip install --no-index --no-deps --require-hashes --find-links /tmp/wheels -r /tmp/wheels/requirements-e17.txt \
 && python3 -c "import deepseek_recipe as d; assert d.__version__ == '0.2.0', d.__version__; d.DeepseekV41Encoding(); print('deepseek_recipe ok', d.__version__)" \
 && rm -rf /tmp/wheels

COPY patch /opt/dsv41-patch
RUN cp /opt/dsv41-patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py \
 && python3 -m py_compile /usr/lib/python3.12/sitecustomize.py
LABEL dsv41.recipe.patches="shapes,mrow,cfg1,codebook,fshift,srcsort,coop,dataflow,kbits,rows,b12x_smalls,fix_o_proj_woa_fp8,woa_prepack,moe_prefill_tc,deepseek_recipe"
ARG DSV41_GIT_REV=unknown
ARG DSV41_TREE_DIFF=unknown
LABEL dsv41.recipe.git_rev="$DSV41_GIT_REV" dsv41.recipe.tree_diff_sha256="$DSV41_TREE_DIFF"
```

`docker/wheels/requirements-e17.txt` (tracked) contains one line:
`deepseek-recipe==0.2.0 --hash=sha256:<sha of the .whl>`. Add `docker/wheels/*.whl` to `.gitignore`.

The same change also needs:

- the same stage appended to `docker/Dockerfile` (from scratch);
- the chain-sync tests moved from e16 to e17, plus a test that asserts the pin, `--require-hashes`, `--no-index`
  and the build assert;
- `recipe.yaml local_tag`, then `render.py`, plus the other 16 tag sites (the promoted boot included);
- a runtime guard in the patch that imports the package: print a `LOG_DISARMED` line ("... DISABLED -> stock:
  deepseek_recipe not importable; rebuild docker/Dockerfile.e17") so `AUDIT=strict` fails on an older image.

Because the wheel sits in the build context, spark2 can build the layer offline on its own e16-rows. That avoids
another 23.7 GB save/load.

### 2.4 Alternative with no rebuild: pure Python under `docker/patch`

Port the behaviours that need fixing into source rewrites of the image's own V4.1 code:

- `tokenizers/deepseek_v41{,_encoding}.py`;
- `reasoning/deepseek_v41_engine_reasoning_parser.py`, `tool_parsers/deepseekv41_engine_tool_parser.py`,
  `parser/deepseek_v41.py`;
- `entrypoints/openai/chat_completion/*` where needed.

Follow the chat_encoder_1017 pattern. Rewrites take effect on restart (ARMS.md:247-248) and reach the worker
through run.sh's scp.

Use deepseek-recipe only off the serve path:

- **A reference-data tool** (for example `tools/api_ref.py`, run in a venv or in the image with the wheel mounted)
  writes golden fixtures to `tests/fixtures/` / `tests/quality/`. Record the generator name and version in the
  fixture.
- **Oracle tests** that import deepseek_recipe use `skipUnless(find_spec("deepseek_recipe"))`, the same way the
  1017 vector tests skip without the HF cache.

### 2.5 Which the conventions favour

The pure-Python route (2.4):

- Every serve-path behaviour change so far is a mounted patch.
- Images change only for compiled kernels. The Dockerfile.e16 header says the JIT levers "compile from the mounted
  docker/patch and need no build stage".
- A pip extra was kept out on purpose, with a test.
- V4.1 chat encoding already has a required rewrite with golden-vector tests.
- A tag bump costs 17 files, both chains, and an offline spark2.

Reserve e17 for wholesale adoption of deepseek-recipe at runtime, for example its Responses or Anthropic
conversion or its OpenCV image preprocessing. Even then, keep the vLLM glue as a mounted patch with markers.

## 3. The quality / eval harness today

### 3.1 Inventory

- **`smoke_chat.py`** (`.../v1/chat/completions`): "What is 17*19? Return only the integer.", `max_tokens` 32,
  temperature 0, `chat_template_kwargs {thinking:false, reasoning_effort:low}`. It needs a 2xx response and
  non-empty content matching `\b323\b` (`--expect ''` accepts any content).
- **`smoke_vision.py`**: a stdlib-built 64x64 red PNG as an `image_url` data URL, `max_tokens` 64. It fails on
  HTTP 400 "is not a multimodal model", and the content must contain "red" as a word.
- **`tests/correctness.{sh,py}`**: 8 checks, plus a 64k needle with `--full`:
  - math 17*19 and 21*12;
  - strict JSON in prose;
  - a `get_weather` tool call (`tool_choice` auto);
  - a code trace;
  - an 8k needle;
  - prose anti-collapse (type-token ratio >= 0.30, 8-gram diversity >= 0.80);
  - the red PNG.

  Temperature 0, seed 20260918, thinking off / low. The run exits 1 on any failure. It is a fixed check, not
  baselined.
- **`tools/served_kl.py`**: sends `/v1/completions` with token ids (`prompt: rec["tokens"]`) and `prompt_logprobs`
  20, on 72 panel rows encoded offline by `tools/calib/data/encode.py` with HF `encoding.py` and `tokenizer.json`.
  It skips image rows, uses the eval boot `MAX_NUM_BATCHED_TOKENS=512 LANGUAGE_MODEL_ONLY=1`, and always returns
  rc 0 (a measurement: 0.0253 nats/token at FINAL-EVAL). It never touches chat rendering or tokenization.
- **Benches** (`benches/e2e.py`, `bench_decode.py`, `benches/micro.py`, `tools/measure_lail_prose.py`): streamed,
  thinking off. They measure performance; e2e also grades a streamed tool call and a needle.
- **Out of tree:** `/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools/grammar_check.py` sends
  `response_format json_schema` (strict), 3 requests at c=1 and 4 at c=4, thinking off. It scored 7/7 on boot D4
  (`results/2026-10-19-1017-perf2/D4/grammar.{json,txt}`).

### 3.2 `tests/quality_eval.py` (stdlib, HTTP; `--url http://127.0.0.1:8000`; model = first `/v1/models` id)

- **Components.** `--quick` runs `nll decode tools needle selfcons c2 vision`; `--full` adds
  `gsm8k gsm8k_think mmlu`; `--only` takes a list.
  - **nll** (`:433-463`): 40 public-domain passages of about 512 tokens. It sends `BOS + text` to
    `/v1/completions` with `prompt_logprobs 0`, skips the first 16 tokens, and in full mode runs twice and records
    `repeat_abs_delta`.
  - **decode** (`:466-495`): 8 token-id prefixes; greedy 128 tokens with logprobs, then the same tokens re-scored
    through `prompt_logprobs`. Reports the median / p99 |dlogprob| and `gen_nll_prefill`.
  - **tools** (`:498-506`, scoring `:196-227`): `tools30.json` has 22 calls and 8 no-call negatives with 16
    schemas. Each is one user turn, `tool_choice` auto, `max_tokens` 256, thinking off, non-streamed. Only the
    first call is scored: `json_valid`, `exact_args` (name and keys equal, strings casefolded) and `no_call`.
  - **needle** (`:509-540`): seeded filler, depths 0.1 / 0.5 / 0.9, lengths 8k / 32k (+128k in full), a nonce
    header to bust the prefix cache, `max_tokens` 32.
  - **selfcons** (`:543-563`): 12 prompts x 2 runs, greedy 128 tokens, `return_token_ids`. Reports the A/A hazard;
    run A becomes the goldens.
  - **c2**: two concurrent short arithmetic chats. **vision**: the red PNG.
  - **gsm8k**: 100 items, thinking off, `max_tokens` 1024. **gsm8k_think**: 40 items, `{thinking:true,
    reasoning_effort:high}`, `max_tokens` 4096. Both score `message.content` only: the number after the last
    "Answer:", else the last number (`:149-167`, `:593-607`).
  - **mmlu**: 4 x 57 subjects, `max_tokens` 8, first letter A-D.
- **Request settings.** Every chat request uses temperature 0 and `{thinking:false, reasoning_effort:low}`
  (`:69`), except gsm8k_think (`:70`). No seed, no streaming, no `response_format`. Concurrency is at most 2, and
  the eval must not run next to a bench.
- **Gates** (`:97-108`, `:323-377`):
  - Absolute: `vision` and `c2` must pass, and no component may raise.
  - With `--baseline`:
    - NLL <= b + max(0.01, 3 x repeat delta).
    - decode median <= b + 0.05; gen-NLL <= b + 0.15.
    - Rates (tools json / exact / no_call, gsm8k, gsm8k_think, mmlu) pass when the Wilson 95% upper bound of
      (k+1)/n >= the baseline rate.
    - needle found >= b on the cells both runs share.
    - Golden hazard (this run's selfcons vs the baseline's run A) <= 2 x max(b A/A, 0.005). The run's own A/A
      hazard has the same limit, and the floor comes from the baseline only.
  - A gate is skipped when the baseline lacks that component.
- **Goldens and baselines.**
  - Recording: run with no `--baseline` on the first boot of a pack and store the JSON. `--result saved.json
    --baseline other.json` re-gates offline.
  - The golden hazard is valid only on the same weights. A pack change follows ARMS step 6 (paired NLL, McNemar,
    A/A band); the 1017 retarget rests on absolute results (ARMS.md:145-157).
  - Provenance: model, root and `max_model_len`, plus the `docker inspect dsv41-flash-exl3` image and a hash of
    the `DSV41_`/`VLLM_`/`NCCL_` env and cmd.
- **1017 baseline** (`results/2026-10-18-1017/quality-baseline/`, `baseline_summary.py`):

  | | `full.json` | `quick.json` |
  |---|---|---|
  | Recorded | boot D1 2026-10-18, image e15-kbits, max_model_len 1M | boot r1 |
  | Runtime | 1,429 s | 368 s |
  | nll | 0.144473 (repeat 0.0018) | 0.143046 |
  | decode median | 0.00069 | 0.00096 |
  | tools | 22/22, 8/8 no-call | 22/22, 8/8 no-call |
  | needle | 9/9 | 6/6 |
  | selfcons | 5/12 identical, A/A hazard 0.01426 | A/A hazard 0.01064 |
  | Task scores | gsm8k 97/100 (mean 188 tokens), gsm8k_think 39/40 (mean 217 tokens), mmlu 208/228 | — |

  FINAL (e16-rows) passed both (full: golden hazard 0.01956 against a limit of 0.02852; gsm8k_think 38/40 at a mean
  of 243 tokens).
- **Where the gate sits.** ARMS.md step 6 (`:94-111`) requires quick to pass before any KEEP; `--full` is required
  for levers that change numerics. flags.md:224-231 freezes the settings ("reasoning_effort and thinking stay
  locked in every harness").

### 3.3 Finding: the harness's prompts are already at API render parity

`harness_prompts_render.py` runs in e16-rows (CPU, no network, wheel mounted). It compares the image's
`DeepseekV41Tokenizer.apply_chat_template(..., tokenize=True)` against deepseek-recipe 0.2.0
`ChatCompletionRequest(thinking:{type}, reasoning_effort)` converted with the alias and encoded with the pack's
`tokenizer.json`. The result is **46/46 identical token-id lists**: smoke, c2, the 12 selfcons prompts, all 30
tools30 items with their schemas, and GSM8K item 0 with thinking off and on (effort high)
(`logs/harness-prompts-render.log`).

Caveat: this is the direct call. vLLM's renderer normalizes the request messages first, which is a no-op for these
string user turns. Parity fixes elsewhere keep the goldens valid unless they change these shapes, so this is worth
a standing test.

## 4. Gap table: what API-parity work needs that the harness does not cover

| id | Behaviour | API (deepseek-recipe) | Coverage today | Impact | Evidence | Minimal addition | Risk |
|---|---|---|---|---|---|---|---|
| G1 | Live render parity | Renders the Conversation; only explicit special ranges encode as special | CPU only: 5 golden vectors, 3 via `apply_chat_template`; no live `/tokenize` check | high: every rendering fix needs a regression guard | `chat_encoder_1017_image_check.py`; 3.3 | `render` component: `/tokenize` {messages, tools, chat_template_kwargs} vs ids recorded offline (5.1) | low: deterministic; `/tokenize` merges `--default-chat-template-kwargs` (`image-src entrypoints/serve/tokenize/serving.py:54,89`; `launchers/api_server/app_state.py:113-119`) |
| G2 | Tool round trips | finish_reason `tool_calls`, ids, multiple calls, results, tool_choice required/none/named, thinking kept within a tool loop | First call of one turn only, auto, thinking off; correctness.py 1 call; e2e 1 streamed call | high | `quality_eval.py:498-506,196-215` | `toolflow` | med: model-dependent, so a rate gate |
| G3 | Reasoning parsing | `reasoning_content` field; `reasoning_tokens` | gsm8k_think reads `content` only; reasoning is not stored | high: leakage into content or a missing field passes | `quality_eval.py:599`; served field is `reasoning` (`image-src .../chat_completion/protocol.py:71`) vs API `reasoning_content` (`upstream .../chat_completion/response/schema.rs:61,83`); thinking vs off means 217/243 vs 188/192 tokens | `reasoning` | low |
| G4 | Structured outputs | `response_format` text / json_object / json_schema / regex | Out-of-tree grammar_check only (D4 7/7, thinking off) | med | `results/2026-10-19-1017-perf2/D4/grammar.txt` | `structured` with thinking-on variants | low |
| G5 | Multi-turn with reasoning history | drop_thinking; current turn kept across tool results | CPU encoder cases only; input `reasoning_content` and `reasoning` not exercised live | med | `chat_encoder_1017_image_check.py:62-70,147-150`; `protocol.py:530-546` | `render` cases | low |
| G6 | Images | OpenCV preprocessing, grid, token span; multiple images; images in tool results | One red PNG; tool-result images CPU only | med | `smoke_vision.py` | `render` count cases plus `vision2` | med: image token counts are the vision track's facts |
| G7 | Streaming | Chunk generators for content, reasoning and tool deltas; usage | Benches only (e2e assembles tool deltas, `benches/e2e.py:94-162`) | high | — | `stream` | med: greedy A/A diverges, so compare structure |
| G8 | API request fields | top-level `thinking:{type}`, `reasoning_effort` none..max | Only `chat_template_kwargs` | med | `upstream .../chat_completion/request/schema.rs:53-57` | the same cases sent API-style | low |
| G9 | Default sampling and limits | thinking forces temperature 1.0; default max_tokens 8192 / 65536 / 131072; bound 393216; 400 texts | Never exercised | med | `upstream deepseek-recipe/src/protocol/validation.rs:112-118` | host tests of the resolution patch; live 400 texts | **high for the gates**: gsm8k_think would become sampled |
| G10 | Chat logprobs | content / reasoning_content split | none | low | — | shape check | low |
| G11 | Responses / Anthropic Messages | first-class in deepseek-recipe | none | med (scope decision) | — | `protocols` smoke | low |
| G12 | Error and validation texts | effort, placeholders, max_tokens, tool names | placeholder and effort checks on CPU only | low | — | 400-text asserts | low |

## 5. Minimal harness additions (sketch, in quality_eval's style)

### 5.1 `render`

Deterministic, so it belongs in `--quick`.

- **Data.** `tests/quality/api_render.json` holds cases `{id, messages, tools?, kwargs | api_fields, ref_ids,
  known?}` and a header `{generated_by: tools/api_ref.py, deepseek_recipe: 0.2.0, encoding: HF encoding.py
  @1bae0e1}`.
- **Cases:** the 46 harness prompts (a golden guard); multi-turn with `reasoning_content` and with `reasoning`; tool
  round trip and parallel results; mid-conversation system and developer roles; literal special-token text; an
  image (count only); effort low / high / max / none and numeric, as kwargs and as API fields.
- **Code:**

```python
def run_render(c: Client) -> dict:
    spec = json.loads((DATA / "api_render.json").read_text())
    rows = []
    for case in spec["cases"]:
        body = {"messages": case["messages"], "chat_template_kwargs": case.get("kwargs", CHAT_KWARGS)}
        if case.get("tools"):
            body["tools"] = case["tools"]
        got = c.post("/tokenize", body)["tokens"]
        d = first_divergence(got, case["ref_ids"])
        rows.append({"id": case["id"], "same": d is None, "first_diff": d, "known": case.get("known")})
    bad = [r["id"] for r in rows if not r["same"] and not r["known"]]
    return {"pass": not bad, "same": sum(r["same"] for r in rows), "total": len(rows), "bad": bad, "items": rows}
```

- **Gate.** An absolute row like vision and c2. A `known` divergence id is reported but not failed until its fix
  lands.

### 5.2 Live components behind a new `--api` mode, folded into `--quick` once baselined

About 30-40 short requests at c <= 2, roughly 1-2 min.

- **`toolflow`.** Items in `api_toolflow.json`: `{prompt, tools, tool_choice, think?, results{name: text}, n_calls,
  expect_in_final}`.
  - Turn 1 asserts `finish_reason == "tool_calls"`, ids, JSON-object arguments and the call count.
  - Append the assistant message (with `reasoning_content` when thinking is on) and the `role: tool` results; turn
    2 asserts the final content contains the expected values.
  - Returns `{"acc": rate(...)}` for the Wilson gate.
- **`reasoning`.** Thinking on at effort low / high / max with a fixed product (23*47 = 1081). Pass when reasoning
  is non-empty, content has no `<think>`, `</think>` or `<｜`, the answer is right and `finish_reason == "stop"`.
  Record `reasoning_tokens`. Thinking off must give empty reasoning.
- **`structured`.** grammar_check moved in-tree, plus `json_object` and thinking-on variants.
- **`stream`.** e2e.py's SSE assembler. Send the same request streamed and non-streamed, and compare tool name and
  JSON-equal args, reasoning and content separation, finish_reason and usage. Require equal text only for answers
  of at most 16 tokens.
- **`vision2`.** Two `solid_png()` colours asked "in order", plus an image in a tool result.

### 5.3 Unit tests

- Extend `tests/test_quality_eval.py`'s `_Fake` handler (`:245-279`) with `/tokenize`, tool-call replies and an SSE
  body, and add a `MainWiringTests` case that runs `--only render,toolflow,stream`.
- Extend `VendoredDataTests` with the new data files.
- Document them in `tests/quality/README.md`.

### 5.4 How the new gates treat a parity change

The new gates are absolute or skipped while the baseline lacks the component, so existing baselines stay valid.
Record a new quick baseline on the first boot of the parity stack. If any of the 46 harness prompts changes how it
renders, the golden hazard fails by construction; treat that like a pack change (owner decision, then re-record).

## 6. Baseline run (part 4)

| Check | Result | Log |
|---|---|---|
| `unittest discover -s tests -q` | 1267 tests, 43.6 s, OK (skipped=49); rc 0; equals PR #10's count | `logs/unittest-host.log` |
| The same run with `-v`, skip reasons | torch missing (27), `DSV41_VLLM_SRC` (7), torch + safetensors (5), vllm / image only (4), vLLM tree (2), `DSV41_IMAGE_TESTS` (1); none are failures (42 of 49 lines parsed) | `logs/unittest-host-verbose.log` |
| `kit/render.py --check` | rc 0, no output | `logs/render-check.log` |
| chat-encoder unittest with `DSV41_IMAGE_TESTS=1` on e16-rows | 8/8 OK | `logs/chat-encoder-image-test.log` |
| `chat_encoder_1017_image_check.py` in e16-rows | 26/26 | `logs/chat-encoder-image-check-e16.log` |

## 7. Already in place or already matching (no need to re-check)

- The 1017 encoder rewrite is engaged and exact on e16-rows.
- The 46 harness prompts render at parity with deepseek-recipe.
- `/tokenize` applies the server's default chat kwargs.
- FORWARD_ENVS, audit, render and promoted-boot consistency was green at 7851999.
- The wheel is usable in e16-rows: Python 3.12, glibc 2.39, next to cv2 5.0, no dependencies.

## 8. Open questions and cross-track notes

- **image-src** carries the baked rewrites, so it is the served text for the tokenizer files. Pristine text:
  `tests/fixtures/*_e14.pin.py`.
- **CONTEXT's in-image docker command** must also mount
  `.../site-packages/deepseek_recipe.libs:/opt/dsr/deepseek_recipe.libs:ro`. With only the package dir mounted:
  `ImportError: libopencv_core-917defc8.so.412` (`logs/context-mount-check.log`).
- **Response field name:** the served message and delta use `reasoning`; deepseek-recipe returns
  `reasoning_content` (and `logprobs.reasoning_content`). On input, vLLM accepts both.
- **Request shape:** the API takes top-level `thinking:{type}` and `reasoning_effort`
  (none / minimal / low / medium / high / xhigh / ultra / max). The served tokenizer reads `chat_template_kwargs`.
  deepseek-recipe's default with no thinking field rendered thinking on with "Reasoning Effort: 75". The recipe's
  default stays off.
- **Thinking-mode temperature:** the API forces 1.0. Adopting that would make `gsm8k_think` sampled.
- **Possible thin reasoning:** GSM8K with thinking on averages 217-243 completion tokens, against 188-192 with
  thinking off. The harness cannot tell whether reasoning was produced and parsed, or leaked into content; G3
  closes this.
- **Wheel sha256** must be fetched on spark1. No live behaviour was verified (no GPU, no serve).

## 9. Harness and logs (`/home/sfxnz/projects/data/dsv41-1017/api-parity/harness/conventions/`)

- `baseline_summary.py`: summarizes `full.json` / `quick.json`.
- `skip_reasons.py`: groups unittest skip reasons from a `-v` log.
- `wheel_in_image.py`: imports, renders and encodes the wheel inside e16-rows.
- `harness_prompts_render.py`: the 46 harness prompts, our renderer vs deepseek-recipe.
- `logs/`: `unittest-host.log`, `unittest-host-verbose.log`, `render-check.log`, `chat-encoder-image-test.log`,
  `chat-encoder-image-check-e16.log`, `wheel-in-image.log`, `harness-prompts-render.log`,
  `context-mount-check.log`.

Image command used for the last two scripts:

```bash
docker run --rm --network none --memory 8g --cpus 4 -e HF_HUB_OFFLINE=1 \
  -v <venv>/lib/python3.12/site-packages/deepseek_recipe:/opt/dsr/deepseek_recipe:ro \
  -v <venv>/lib/python3.12/site-packages/deepseek_recipe.libs:/opt/dsr/deepseek_recipe.libs:ro \
  -v <worktree>:/w:ro -v /home/sfxnz/projects/data/dsv41-1017/api-parity:/ap:ro -v ~/.cache/huggingface/hub:/hub:ro \
  -e PYTHONPATH=/opt/dsr --entrypoint python3 dsv41-flash-exl3-sm121:e16-rows /ap/harness/conventions/<script>.py
```
