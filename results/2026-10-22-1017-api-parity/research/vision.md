# VISION track: how images get from the request to the model input (DeepSeek API via deepseek-recipe 0.2.0, compared with our serve)

## SUMMARY (short form)

On ordinary photos and screenshots our serve matches the API on the placeholder layout, the token grid and the token count:
- 37 of the 42 corpus images both sides accept get the same size and token count.
- All 26 common camera, screen and phone sizes match, and so do 21 long-screenshot sizes.
- The prompt strings are identical in 9 of 9 placement cases.


| ID | Divergence | API | Ours | Impact | Fix |
|---|---|---|---|---|---|
| V1 | 16-bit grayscale PNG | 16U×1/257 → correct | Pillow I;16→RGB clips → almost white (PSNR 4.9 dB against API) | **H** when hit (content lost), rare | patch `convert_image_mode`: I;16*→L, rint(v/257). Prototype gives 51.5 dB |
| V2 | Tool results with images, out of call order | images collected after the tool-result sort | images kept in request order while the 1017 encoder sorts → **image bound to the wrong tool result** | **H** when hit, rare | renderer patch: sentinel render → permute mm_data/uuids. Prototype passes |
| V3 | Anthropic tool_result image | image inside `<tool_result>…</tool_result>`, "\n\n" joins | vLLM moves the image to a separate user message after `</tool_result>`, joins with "\n" | **M** (agent screenshots over /v1/messages) | patch `anthropic/serving.py:_convert_user_tool_result` (coordinate with the protocols track) |
| V4 | EXIF orientation | ignored (phone photos arrive sideways) | `exif_transpose` (upright) | ours is better; tokens 322 vs 317 | **keep ours**, document it |
| V5 | Image sizing | budget 1024, fixed point, WebP 16383-px clamp | budget 1023 (pad reservation), one pass, no clamp | L-M: tokens differ for 3.0% of 31,684 sizes; e.g. 1024x1895 gets 945 vs 1010 tokens; extreme aspect ratios get more | rewrite `mm_preprocess` sizing to the upstream rule (port has 0 mismatches against Rust) |
| V6 | Pixel pipeline | OpenCV INTER_CUBIC (no antialias) + pad 127 + WebP q90 (4:2:0) | PIL bicubic with antialias + pad 127, no codec (= HF reference) | unknown sign. PSNR: photos 37-46, screenshots 35-37, small coloured text 23.9 dB | no default change. Opt-in `api` lever (cv2 5.0 in the image reproduces the API bit-exactly, 43/43) plus a vision A/B |
| V7 | `detail: "low"` | long side ≤ 512 px (INTER_AREA) → about 190 tokens | ignored (1920x1080 → 968 tokens) | L quality / M cost | optional patch |
| V8 | Limits and formats | jpeg/png/gif/webp only; ≤ 8192 px (≤ 4096 with ≥ 15 images); 32/64 MiB; 600 images | any PIL format (BMP, TIFF, AVIF); up to 179 MP (above that HTTP 500); no data-URL byte cap; 256 MiB for URLs; 999 images | L (client-visible) | optional: `--limit-mm-per-prompt '{"image":600}'`, `VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB=32` |
| V9 | Data-URL base64 | forgiving: line-wrapped, spaced and unpadded all accepted | strict pybase64: 400 "Non-base64 digit found"; `;BASE64` → 501 | L-M (clients that MIME-wrap base64 fail) | patch `ImageMediaIO.load_base64`: strip whitespace, re-pad |
| V10 | Image in an assistant message | 400 | accepted and rendered inside the assistant turn | L | reject in `_normalize_messages` |
| V11 | `file` part with image `file_data` | accepted (flattened form) | 400 "Unsupported chat content part type" | L | optional |
| V12 | Literal `<｜deepseek_image｜>` in text | plain text | 400 (same as HF encoding.py) | L | keep ours |
| V13 | Error codes | 400 | file:// → 500, percent or `;BASE64` data URL → 501, > 179 MP → 500 | L | optional |
| V14 | URL-fetch security | DestinationFilter checks host, every resolved IP and every redirect; ≤ 5 redirects; 32 MiB; no proxy | no filter (loopback and LAN reachable), ≤ 9 redirects, `--allowed-media-domains` unset and not re-checked on redirect, env proxies honoured, 256 MiB; serve on 0.0.0.0, no API key | **M security** if port 8000 is reachable by untrusted clients | `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` + allowed domains, or a resolver-filter patch, or a data-URL-only policy |
| V15 | Compressor-alignment pad | not counted | +0 or 1 pad token per image (stock vLLM) | unknown | keep; open question |
| V16 | (model side) Bidirectional attention within image spans | n/a; public reference model.py is causal | stock vLLM ports `get_image_visible` (bidirectional SWA inside images); the recipe's SM120 rewrite sets it to 0 | **potentially H, unverified**; the 64x64 red-PNG gate cannot see it | quality track: build a vision eval, A/B if a kernel path exists |

Top recommendations:
1. Ship V1, V2, V9 and V10 together as one small "vision input fixes" patch set. Normal images are unaffected, and V1 and V2 have working prototypes.
2. V5: sizing-parity source rewrite of `mm_preprocess.py`. It is pure math and the harness checks it (`spec_port.upstream_target`).
3. V3: with the protocols track. This is the main prompt-placement divergence (Claude Code style screenshot tool results).
4. Before touching V6 or V16, add a real vision eval: OCR on rendered text, charts, colour, orientation, counting, multi-image. The current gate is a solid red 64x64 PNG.
5. V14: decide the exposure policy. Minimum: `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` and `VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB=32`.
6. Keep V4, V12 and V15 as they are, and document them.

Files (all under `/home/sfxnz/projects/data/dsv41-1017/api-parity/harness/vision/`):
- `run_all.sh` re-runs everything in about 80 s, CPU only, no network.
- `run_in_image.sh` is the image runner.
- Scripts: `gen_images.py` (48-image corpus in `images/`, plus `manifest.json`), `spec_port.py`, `pixel_helpers.py`, `upstream_run.py` (host venv), `vllm_run.py`, `vllm_chat.py`, `api_emulation_check.py`, `sweep_causes.py`, `fix_probe_16bit.py`, `fix_probe_tool_order.py`, `dataurl_leniency.py`.
- Evidence: `out/*.json` and `out/*.log`; upstream WebP outputs in `out/upstream/`.
- `model_stub/` is the pack config with `quantization_config` removed, plus the tokenizer files (used to build the processor without weights).

---

## FULL REPORT

### 0. Method and caveats
- **Upstream (API):** the deepseek-recipe 0.2.0 wheel in the host venv. `OpenCvImagePreprocessor` with `PreprocessOptions(detail, 8192 or 4096, 512)`, then `ImageResolver`, Chat Completions / Messages `convert`, then `DeepseekV41Encoding.render_conversation`. Request model name is `DeepSeek-V4.1-Flash`.
- **Ours:** the e16-rows image, CPU only, with the baked sitecustomize active. Steps run:
  - `ImageMediaIO().load_bytes`, the exact serve decode path.
  - `DeepseekV4VLProcessor` from the pack config.
  - `chat_utils.parse_chat_messages(content_format="string")`.
  - The deepseek_v41 tokenizer's `apply_chat_template`.
  - `MULTIMODAL_REGISTRY.create_processor(...)` for the expanded span. No weights are loaded.
- **Checks on the harness itself:**
  - Python port of the upstream sizing against Rust `calc_token_count`: 0 mismatches over 31,684 sizes.
  - Port of our sizing against the real `mm_preprocess.safe_resize`: 0 mismatches.
  - Unpatchified processor output equals the uint8 replica: all images.
  - Port target equals the API's emitted size: all images.
- **CONTEXT correction:** image-src is not pristine.
  - Baked rewrites: the chat-encoder-1017 rewrite in `tokenizers/deepseek_v41{,_encoding}.py`, and the sm120 image-width clamp at `models/deepseek_v4_1/attention.py:244` and `v1/attention/backends/mla/sparse_swa.py:449`.
  - Stock (no MARK strings): `mm_preprocess.py`, `vl_model.py`, `multimodal/media/*`, `multimodal/image.py`, `entrypoints/chat_utils.py`, `renderers/deepseek_v4.py`, `entrypoints/anthropic/serving.py`.

### 1. What upstream does (deepseek-recipe @ 935817c)
1. **Request → ImageSource**
   - Chat `image_url`: an `http`-prefixed URL (≤ 8192 bytes) becomes a Url source, anything else a DataUrl source (`protocol/image.rs:73-93`).
   - Data URLs must contain the literal `base64,`, parse, declare png/jpeg/jpg/gif/webp, and decode with forgiving base64 (`image.rs:100-124`).
   - A Chat `file` part with flattened `file_data` becomes an image (`chat_completion/request/schema.rs:133-139`, `image.rs:130-152`). `file_id` → 400.
   - Images in assistant messages → 400 "Image in assistant message is not supported" (`chat_completion/request/convert.rs:461-467`).
   - Text and image blocks are joined with "\n\n" and each image is an explicit special token (`convert.rs:291-357`). It is spelled `<｜image｜>` (`deepseek-recipe-core/src/special_tokens.rs:69`) and has id 129264 (`multimodal/mod.rs:18`).
   - Anthropic image blocks stay inside tool_result content, joined with "\n\n" (`anthropic/messages/request/convert.rs:266-290, 329-357`).
   - Responses: `input_image`. V4.1 also moves an `<image_resize_notice>` developer message right after the tool output (`responses/request/convert.rs:372-406`; protocols track).
2. **Placement in the prompt (V4.1)**
   - System, user and tool messages can carry images (`dsv41.rs:167-193`).
   - Consecutive user messages are merged with "\n\n" (`dsv41.rs:321-344`).
   - Tool results are wrapped in `<tool_result>…</tool_result>` and sorted by tool-call order (`dsv41.rs:190-193, 353-419`).
   - Images are collected from the normalized, sorted messages, so they follow placeholder order (`v4/mod.rs:109-136`).
3. **Resolve**
   - Defaults (`limits.rs:40-53`): ≤ 600 images per request (counted across calls with `ImageQuota`), ≤ 32 MiB per image (a data URL is estimated before decoding), ≤ 64 MiB total, 8 concurrent.
   - Fetch (`fetcher.rs:16, 124-128, 232-324`): reqwest, ≤ 5 redirects, 10 s connect / 60 s total, Content-Length checked first, streamed body checked against the budget.
   - Retries at 0.3 / 0.6 / 1.0 s on transport errors and 5xx (`retry.rs:27-35`, `error.rs:417-423`).
   - `DestinationFilter` checks the URL host, every resolved address and every redirect target, and uses no proxy. It is installed by the caller; there is no default filter.
4. **Decode**
   - Media type comes from magic bytes; only jpeg, png, gif and webp are accepted.
   - Header dimensions must be ≤ 8192, or ≤ 4096 when the request has ≥ 15 images; they are re-checked after decoding.
   - OpenCV `imdecode(IMREAD_UNCHANGED)` (`opencv.rs:136`), which ignores EXIF orientation.
   - GIF goes through the `image` crate: first frame, RGBA (`opencv.rs:49-147, 433-465`).
5. **Colour** (`opencv.rs:19, 150-236`)
   - 16-bit → 8-bit by ×1/257.
   - Grayscale is replicated to BGR.
   - Gray+alpha and BGRA are blended onto **#FDFDFD**: (c·a + 253·(255−a) + 127)/255.
   - CMYK JPEG uses OpenCV's own conversion.
6. **detail=low:** long side ≤ 512 with INTER_AREA before fitting; fitting may upscale again. High, original and auto are treated the same (`opencv.rs:56-58, 240-262`; `multimodal/mod.rs:83-98`).
7. **Fit** (`token_spec.rs:69-235`, `opencv.rs:315-359`)
   - `ImageTokenSpec::v41`: patch 14, downsample 3, ≤ 1024 tokens, minimum area 544².
   - Upscale to the minimum area (truncating), round up to the 14-px grid, solve the largest aspect-preserving size if the count exceeds 1024.
   - Iterate to a fixed point (≤ 10 times), then clamp both sides to ≤ 16383 (the WebP limit).
8. **Resize:** aspect-preserving **INTER_CUBIC**, centred pad with value **127** (round-half-even offsets); no resize when the size already fits (`opencv.rs:17, 284-289, 363-415`).
9. **Encode:** **WebP q90** (lossy, 4:2:0) into `ImageInfo{data, w, h}` (`opencv.rs:25, 291-304`). Token count is n_h·(n_w+1)+2 (`token_spec.rs:197-199`). The prompt carries one placeholder per image and the backend expands it (`multimodal/mod.rs:197-204`).
10. **Errors:** per-image `"{path}.image[i]: You have uploaded an unsupported image…"`; size, total and count limits produce their own texts; status 400. `Client` and `TokenBudget` errors are 500 (`error.rs:450-517`).
11. **HF reference** (`inference/image_processor.py`, identical in Sept and 1017):
    - PIL `convert("RGB")`: no EXIF transpose, and alpha is dropped without compositing.
    - Budget 1024, one pass (`101-112`).
    - `ImageOps.pad` with BICUBIC and colour 127 (`125`); x/255, then (x−0.5)/0.5, then bf16, 14×14 patches (`126-128`).
    - `encoding.py:39, 323-396`: placeholder `<｜deepseek_image｜>`; parts joined "\n\n"; a literal placeholder in text → ValueError.
    - Tech report: "constrain the input resolution between 544 × 544 and 1344 × 1344".

### 2. What our serve does
1. **Parse**
   - `renderers/deepseek_v4.py:35-87` → `chat_utils.parse_chat_messages(content_format="string")` (`chat_utils.py:2120-2150`). Images are collected **in request order**, from every role including assistant.
   - Accepted part types: image_url, input_image, image_pil; image_embeds needs `--enable-mm-embeds`. A `file` part → 400 (`chat_utils.py:1937`).
   - `detail` is dropped. The warning at `chat_utils.py:1682-1688` reads `part["detail"]` rather than `part["image_url"]["detail"]`, so it never fires.
2. **Prompt** (1017 rewrite, baked): `tokenizers/deepseek_v41.py:21-47`.
   - Each image part becomes `<｜deepseek_image｜>`. In the pack tokenizer this is id 129264, a non-special added token.
   - Parts are joined with "\n\n"; a literal placeholder in text → 400.
   - The encoder sorts tool results by call order. The full prompt string is then tokenized (`deepseek_v41.py:90-102`).
3. **Fetch:** `multimodal/media/connector.py:313-465`, `connections.py:63-151, 278-347`.
   - Data URLs: base64 only, `pybase64.b64decode(validate=True)` (`media/image.py:104-105`); the declared media type is ignored.
   - http(s): aiohttp with `trust_env=True`. `--allowed-media-domains` checks the first host only. Redirects are on by default (`VLLM_MEDIA_URL_ALLOW_REDIRECTS=1`; aiohttp follows ≤ 9).
   - Timeout 5 s ×4 per retry, 3 attempts; 256 MiB cap (`envs.py:71-86`).
   - `file://` needs `--allowed-local-media-path`; without it, RuntimeError → 500.
   - No destination filter.
4. **Decode:** `media/image.py:79-102`.
   - PIL accepts any of its formats, including BMP, TIFF, ICO and AVIF (Pillow 12.3 has AVIF).
   - Over 178,956,970 px, PIL raises `DecompressionBombError` → HTTP 500.
   - `ImageOps.exif_transpose` is applied (`multimodal/image.py:21-25`).
5. **Colour:** `multimodal/image.py:28-60`, `media/image.py:64-77`.
   - RGBA is pasted onto **white (255)**; P, L or RGB with tRNS, plus LA and PA, go through RGBA onto white.
   - Everything else uses PIL `convert("RGB")`; for **I;16 this clips to 255**.
   - GIF: first frame.
6. **Fit:** `mm_preprocess.load_image` (`mm_preprocess.py:151-191`), the HF one-pass plan with budget **1024 − (COMPRESS_PAD_TO − 1) = 1023** (`:138`). No low-detail path, no WebP clamp.
7. **Resize:** `ImageOps.pad` with PIL BICUBIC (antialiased) and colour 127 (`:183`).
8. **Normalise:** x/255, (x−0.5)/0.5, bf16, 14×14 patches (`:184-190`).
9. **Span:** [START] + ([IMAGE]×n_w + [NEWLINE])×n_h + [END], all id 129264, plus a leading **0 or 1 compressor-alignment pad** (id 129265) so the span starts at an odd position (`:401-463`).
   - The pad is embedded as 129264 (`vl_model.py:280-285`).
   - Maximum per item is 1025 (`:291-302`).
10. **Limits:** 999 images per prompt by default (`config/multimodal.py:501`). `media_io_kwargs` per request is client-settable: `image_mode`, `rgba_background_color`.
11. **Recipe**
    - `run.sh:33-34, 558, 603-608`: `--mm-encoder-tp-mode data`, `--mm-processor-cache-gb 1`, `--host 0.0.0.0`.
    - Not set: `--limit-mm-per-prompt`, `--allowed-media-domains`, `--api-key`.
    - No recipe patch touches preprocessing. The only vision-related patches are the 1017 chat encoder (placeholder check, `docker/patch/chat_encoder_1017.py`) and the SM120 image-width clamp (`sm120_page.py:520-530, 544-592`, `sitecustomize.py:642-656`).
    - Vision gate: `smoke_vision.py` and `tests/quality_eval.py:580-590` (64x64 solid red PNG).

### 3. Divergences (full form)
**V1: 16-bit grayscale → white.**
- API: `opencv.rs:151-158`. Ours: `multimodal/image.py:60` / `media/image.py:75-77`; the processor's `convert` at `mm_preprocess.py:166` has the same behaviour, and so does the HF reference.
- Evidence (`out/vllm_run.log`, gray16):
  - ours vs API: PSNR 4.86 dB, mean |Δ| 125.5.
  - Probe at x = 25% / 75%: ours 255 / 255, API 63.8 / 190.8.
  - `out/fix_probe_16bit.log`: fixed vs API 51.5 dB; RGB16 and L unchanged.
- Impact: H for scientific, medical, depth and scan images; rare overall.
- Fix: new `docker/patch/image_16bit_gray.py` that wraps `vllm.multimodal.media.image.convert_image_mode` (the name is bound there): `I;16*` → L via `rint(v/257)`, then the stock conversion.
- Risk: low.

**V2: tool-result images misassigned.**
- API: `v4/mod.rs:109-121` + `dsv41.rs:353-419`; HF `encoding.py:826-833`. Ours: `chat_utils.py:2120-2150` + the 1017 sort, with the processor binding images in order (`mm_preprocess.py:418`).
- Evidence (`out/vllm_chat.log`, tools_out_of_order):
  - mm_order is [[2,2],[1,1]], but the prompt has `<tool_result>first…` before `second…`.
  - The prompt string equals the API's, yet the images are swapped.
  - `out/fix_probe_tool_order.log`: stock_ok false, fixed_ok true; the permutation is a no-op on the other cases.
- Impact: H when it triggers (silent wrong image); needs parallel tool calls with image outputs returned out of order.
- Fix: `renderers/deepseek_v4.py` rewrite.
  - Render once with a unique sentinel per image part.
  - Read the sentinel order.
  - Permute `mm_data["image"]` and `mm_uuids["image"]` to it.
- Risk: low-medium (one extra string-only render per image request).

**V3: Anthropic tool_result images.**
- Ours: `anthropic/serving.py:406-450`. API: `anthropic convert.rs:329-357`.
- Evidence (`vllm_chat.log`):
  - Ours: `<tool_result>png:</tool_result>\n\n<｜deepseek_image｜>\n\nwhat do you see?`
  - API: `<tool_result>png:\n\n<｜image｜></tool_result>\n\nwhat do you see?`
- Fix: keep the image parts in the tool message's list content, in block order. `_normalize_messages` already joins list content with "\n\n".
- Risk: low; V2 also applies to this path.

**V4: EXIF orientation.**
- API and HF reference ignore it; ours applies `exif_transpose`.
- Evidence:
  - exif_rot6 / rot8: ours 602x812 with 322 tokens, API 812x602 with 317.
  - exif_rot3: PSNR 8.33 dB (the API image is upside down).
- Recommendation: keep ours. Most phone JPEGs carry orientation tags, so ours is better for quality. If strict parity is ever wanted, add a lever only for A/B.

**V5: sizing.**
- Evidence (`out/vllm_run.log` sweep, `out/sweep_causes.json`):
  - Tokens differ for 962 of 31,684 sizes (3.04%); pixel size differs for 1,154 (3.64%).
  - By cause: budget 294, budget+clamp 143, iteration 126, iteration+clamp 58, clamp 341 (clamp cases are aspect ≥ 50 only).
  - By aspect ratio: 194 cases are below 2:1, typically −63 tokens.
- Examples:

  | Size | API | Ours | Δ |
  |---|---|---|---|
  | 1024x1895 | 952x1764 / 1010 | 924x1708 / 945 | −65 |
  | 532x3026 | 1024 tokens | 504x2856 / 886 | −138 |
  | tok1024_505x3027.png | 518x3038 / 1024 | 504x3010 / 938 | −86 |
  | 8192x1 | 16380x28 / 393 | 42840x42 / 1023 | +630 |

- Unaffected: the 26 common sizes and 21 long-screenshot sizes have identical tokens.
- Fix: rewrite `load_image` / `safe_resize` to budget = `vision_max_n_token` (1024 + pad = 1025 still equals `get_mm_max_tokens_per_item`). Then port `calc_resize`'s fixed point and the 16383 clamp (`spec_port.upstream_target`).
- Risk: low. Re-check if in-image bidirectional SWA (V16) is ever re-enabled: the stock code bounds the span by `vision_max_n_token`.

**V6: interpolation and codec.** Evidence: `out/vllm_run.log` and `out/api_emulation.log`.

| Content | Interpolation only (ours vs API without WebP) | WebP only (API without vs with) |
|---|---|---|
| Text screenshots | 35.1-37.6 dB | 43-45 dB |
| Phone-size synthetic scenes | 39.1-39.3 dB | 39.1-39.8 dB |
| Real HF photos | 53.9-57.3 dB | 44.0-46.3 dB |
| Small coloured text | 43.3 dB | **23.6 dB** (chroma subsampling) |
| Zone plate | 32.9 dB | not used |

- Ours equals the HF reference (PSNR inf) apart from alpha and EXIF cases.
- **cv2 5.0 in the image reproduces the API's WebP bit for bit on all 43 images, with identical byte counts.**
- Fix: optional lever `DSV41_IMAGE_PREPROCESS=api` in `ImageMediaIO.load_bytes`, which has the original bytes. Steps: cv2 `IMREAD_UNCHANGED`, blend onto #FDFDFD, 16-bit ×1/257, INTER_CUBIC fit + pad 127 at the upstream target size, imencode WebP q90, decode.
  - Requires V5, so the processor does not resample a second time.
  - Default stays as it is until a vision A/B decides.
- Risk: medium.

**V7: `detail: "low"`.**
- API examples (high vs low tokens): 1920x1080 is 968 vs 192; 4032x3024 is 1001 vs 194.
- Ours: always high.
- Fix: optional; carry `detail` through chat_utils into the media load (long side 512, INTER_AREA).
- Risk: medium (touches the parse path).

**V8: limits and formats.** Evidence: `upstream_run.log` limits, `vllm_run.log`.
- API messages:
  - "Too many images: max 600 images per request, got 601".
  - 15 images including one 5000-px image → ImageDimensionsTooLarge.
  - "image file size exceeds limit 32 MB".
  - "Total image size exceeds the limit: max 64 MB per request, got 75 MB".
- Ours:
  - Accepts 9000x6000 (977 tokens), 10000x158, BMP and TIFF (and AVIF).
  - 199 MP → `DecompressionBombError` → 500.
  - A 40 MiB URL body is accepted.
- Fix: optional `--limit-mm-per-prompt '{"image":600}'`, `VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB=32`, and a validation patch (8192 / 4096 px, format allowlist) if strict parity is wanted.

**V9: data-URL leniency** (`out/dataurl_leniency_{api,ours}.log`).
- API accepts line-wrapped (MIME 76-column), space-separated and unpadded base64, and a charset parameter. It rejects the URL-safe alphabet, `;BASE64` and a missing media type.
- Ours rejects wrapped, spaced and unpadded payloads (400 "Non-base64 digit found"); `;BASE64` → 501; a missing or BMP media type is accepted.
- Fix: in `load_base64`, strip ASCII whitespace and re-pad before the strict decode.
- Risk: low.

**V10: assistant images.** Ours renders `…</think>look\n\n<｜deepseek_image｜><｜end▁of▁sentence｜>` and runs the image; the API returns 400. Fix: ValueError in `_normalize_messages` for assistant messages with image parts (fits the chat_encoder_1017 rewrite).

**V11: `file` parts.** API accepts the flattened `{"type":"file","file_data":…}` (the OpenAI nested form gets a 400 there too); ours rejects both. Optional mapping in chat_utils.

**V12: literal `<｜deepseek_image｜>` in text.** Ours returns 400 (HF behaviour); the API treats it as plain text. Keep ours: without the check the literal tokenizes to 129264 and breaks the placeholder/image count.

**V13: status codes.** file:// → 500 (`connector.py:340-342`), `;BASE64` or percent-encoded data URL → 501, > 179 MP → 500; the API returns 400 for all. Optional: convert these to ValueError (400).

**V14: URL fetch (security).** Evidence (`vllm_chat.log` media):
- `loopback_no_filter`: ok.
- `redirects_6`: ok; `redirects_10`: 422.
- `allowed_domain_direct_ip`: 400, but `allowed_domain_redirect_to_ip`: **ok** (redirects bypass `--allowed-media-domains`).
- 40 MiB body: ok.

Context: `--network host`, `--host 0.0.0.0`, no `--api-key`, so any LAN client can make spark1 fetch internal URLs, including its own services. Fix:
- In run.sh env: `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` and `VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB=32`.
- Then either set `--allowed-media-domains`, or add a resolver-level filter patch (reject loopback, private, link-local, multicast and reserved addresses; aiohttp supports a custom resolver), or document a data-URL-only policy.

No quality impact.

**V15: compressor-alignment pad.** Spans at even offsets get pad=1 (`vllm_chat.log` spans). This is stock DeepSeek day-0 vLLM behaviour, absent from the HF reference and from deepseek-recipe's count. Keep.

**V16 (model side): in-image bidirectional SWA.**
- The stock image says "image spans are visible bidirectionally": `sparse_swa.py:905-945` is a "port of `get_image_visible`"; see also `attention.py:237-247` and `configs/deepseek_v41.py:63-76`.
- The recipe's `swa_image_tokens_for_dispatch` returns 0 (`sm120_page.py:520-530`), imported from the Sept recipe for FlashInfer SM120 dispatch reasons. flags.md has no quality evidence for it.
- The public reference model.py (both Sept and 1017, `410-427`) and the tech report ("40 causal Transformer layers") are causal, so the recipe matches the public reference but not the stock port of DeepSeek's function.
- Re-enabling it would also break decoder replay's W ≥ 2414 exactness bound, which assumes a 128-row window per layer.
- Hand this to the quality track.

### 4. Already matching (no need to re-check)
- **Placeholder and placement**
  - One placeholder per image with id 129264; upstream's `<｜image｜>` and our `<｜deepseek_image｜>` both encode to it.
  - Joins with "\n\n", including empty text parts. Images in system and tool messages.
  - Merged consecutive user messages; the `<tool_result>` wrapper.
  - The prompt string equals the API's in 9 of 9 cases: text_then_image, image_then_text, two_images, empty_text_part, system_image, consecutive_users, tool_image, tools_out_of_order (text only), detail_low.
- **Grid and constants**
  - n_h·(n_w+1)+2 tokens, START / IMAGE / NEWLINE / END layout.
  - Patch 14, downsample 3, maximum 1024 tokens, minimum area 544²; the pack's `vision_config` agrees.
  - Pad colour 127 with round-half-even centring; bf16 normalisation (HF reference).
- **Sizes and token counts:** all 26 common sizes, 21 long-screenshot sizes, and 37 of 42 corpus images. The 5 that differ are 2 extreme aspect ratios, 2 EXIF rotations and 1 budget case.
- **Colour handling within interpolation and codec noise**
  - Alpha composites to near-white on both sides: 255 vs 253, PSNR 45.7 dB before WebP.
  - LA, palette with and without tRNS, 8-bit grayscale, 16-bit RGB, progressive JPEG, lossy and alpha WebP.
  - CMYK: PIL is closer to the RGB source than OpenCV (45.6 vs 42.9 dB).
  - Animated GIF: frame 0 on both sides.
  - GIF transparency: white vs #FDFDFD.
- **ICC profiles:** neither side converts colour profiles (by code inspection, not tested).
- **Prefix caching:** vLLM's mm hashes key the image (original bytes + io_config). The recipe's prefix patches only change retention.

### 5. Open questions
1. Does DeepSeek's production front end use `OpenCvImagePreprocessor` unchanged (WebP q90), and was the model trained on that path or on the PIL reference path? This decides V6.
2. Does DeepSeek's engine pad image spans for compressor alignment (V15)?
3. Is in-image bidirectional SWA part of 1017 training (V16)? This needs a vision eval on GPUs, which I could not run.
4. Which destination filter does the real API install? The library has no default.
5. Responses `<image_resize_notice>` handling and `input_image` `detail` / `file_id`: protocols track.
6. Not tested: HEIC (not registered in the image's Pillow), APNG and animated-WebP frame choice, and vLLM `uuid`-keyed image cache reuse (a multi-tenant risk only).
