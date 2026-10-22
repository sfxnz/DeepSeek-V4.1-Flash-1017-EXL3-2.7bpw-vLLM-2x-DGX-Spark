#!/usr/bin/env python3
"""DeepSeek API front end for the dsv41 serve: deepseek-recipe over vLLM's token-in/token-out endpoint.

DeepSeek publishes the library its API front end is built on (deepseek-recipe 0.2.0, MIT). This server
runs it on the head node in front of vLLM:

  POST /v1/chat/completions, /v1/messages, /v1/responses
    1. <Protocol>Request(body).convert(): the API's validation, thinking switch (`thinking:{type}`,
       effort aliases minimal..max), sampling policy, stop sequences and output constraint. The serve's
       default stays thinking off (ConversionOptions(default_thinking_mode=False)).
    2. DeepseekV41Encoding renders the V4.1 prompt and encodes it with the pack's tokenizer.json; only
       the template's special-token ranges encode as special tokens, so a spelling like <｜User｜> in
       request text stays plain text (docs/tokenizer.md). The pack spells id 129264 <｜deepseek_image｜>,
       hence SpecialTokens(image=...).
    3. Images keep vLLM's pixel pipeline: the original bytes go to vLLM as data URLs, in the order of the
       prompt's placeholders (vLLM binds images to placeholders in order). URL images are fetched here
       within the API's count and byte limits (DSV41_GATEWAY_BLOCK_PRIVATE_URLS=1 refuses loopback,
       private, link-local, multicast and reserved addresses, redirects included, and refuses any request
       bound for vLLM that names an http(s) media URL). The API's image token count, plus one token per
       image for vLLM's alignment pad, resolves max_tokens (8192 / 65536 / 131072 at max effort, capped).
    4. The token IDs stream from vLLM's /inference/v1/generate (VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1) with
       the API's temperature, top_p, max_tokens, seed and logprobs, the structured outputs of
       constraints.py, and the vLLM sampling extras of PASS_THROUGH. Each token-ID delta feeds the API's
       StreamProcessor (one InferenceChunk.tokens per vLLM delta; the EOS id of a stop is not pushed,
       inference.rs) and the protocol's chunk generator: the response body or SSE stream is the API's
       (header x-dsv41-gateway: api). Prefix-cache hits arrive in vLLM's final usage chunk and are
       written into the final usage.
  Stop strings are not sent to vLLM, which would also match them in reasoning and tool markup: the
  processor matches them in answer content only, and the gateway closes the upstream stream when the
  processor finishes, or when the client goes away, so vLLM aborts the request. vLLM answers 200 and
  reports an engine-side rejection (grammar, logprobs or context limit) as its first stream event, so a
  response starts with vLLM's first event: such a rejection is an HTTP error in the API's shape, and a
  stream's headers arrive with the first token (vLLM's own endpoints send headers at once).

Requests the API rejects, requests with vLLM-only fields that change the prompt, the grammar or the
output (FALLBACK_FIELDS), and grammars constraints.py cannot gate, go to vLLM unchanged (header
x-dsv41-gateway: fallback, one log line with the reason). On Chat, vLLM's chat_template_kwargs thinking
switches and max_completion_tokens map onto the DeepSeek fields with today's meaning (shim_chat).
DSV41_GATEWAY_STRICT=1 returns the API's error instead of falling back. GET /health is 200 only while vLLM's /health is; POST
/dsv41/render/{chat,messages,responses} returns what would be sent, without generating. Every other
route is proxied to vLLM as is (raw path, query, headers, body), except vLLM's scale-out routes, which
serve the gateway only (404).

The protocol core needs deepseek_recipe only; the HTTP layer (make_app: starlette, httpx, uvicorn of the
serve image) is imported when the server starts. Run it with the image's sitecustomize masked (that
imports vLLM and torch at interpreter start):

    python3 dsv41_gateway.py --host 0.0.0.0 --port 8000 --tokenizer <pack>/tokenizer.json \\
        --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import ipaddress
import json
import os
import re
import sys
import time
import urllib.parse
import uuid

import deepseek_recipe as d

from constraints import FALLBACK, structured_outputs_for

ENV_UPSTREAM = "DSV41_GATEWAY_UPSTREAM"
ENV_STRICT = "DSV41_GATEWAY_STRICT"
ENV_BLOCK_PRIVATE_URLS = "DSV41_GATEWAY_BLOCK_PRIVATE_URLS"
ENV_MAX_MODEL_LEN = "DSV41_GATEWAY_MAX_MODEL_LEN"
DEFAULT_UPSTREAM = "http://127.0.0.1:8001"
DEFAULT_MAX_MODEL_LEN = 500_000  # run.sh MAX_MODEL_LEN

# Startup marker (engagement audit with API_GATEWAY=1): LOG_ENGAGED <version> upstream=<url> model=<name>
LOG_ENGAGED = "dsv41-gateway: engaged deepseek-recipe"

PACK_IMAGE_TOKEN = "<｜deepseek_image｜>"
MODEL_NAMES = ("DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash-1017")  # plus the served name
GENERATE = "/inference/v1/generate"
ROUTES = {"/v1/chat/completions": "chat", "/v1/messages": "messages", "/v1/responses": "responses"}
PROTOCOLS = {
    "chat": (d.ChatCompletionRequest, d.ChatCompletionResponse),
    "messages": (d.MessagesRequest, d.MessagesResponse),
    "responses": (d.ResponsesRequest, d.ResponsesResponse),
}
ID_PREFIX = {"chat": "chatcmpl-", "messages": "msg_", "responses": "resp_"}
FINISH = {"stop": d.InferenceFinishReason.Stop, "length": d.InferenceFinishReason.Length}

# Request fields vLLM honours and the API path would drop (they change the prompt, the grammar, the
# sampling, the output or the conversation state), each with the value that changes nothing (absent or null
# is the same). Any other value goes to vLLM unchanged.
FALLBACK_FIELDS = {
    "previous_response_id": None,  # Responses state (vLLM answers 404 without its store)
    "conversation": None,
    "logit_bias": None,
    "thinking_token_budget": None,
    "include_reasoning": True,
    "add_special_tokens": False,
    "repetition_detection": None,
    "logprob_token_ids": None,
    "return_token_offsets": False,
    "return_prompt_text": False,
    "return_assistant_tokens_mask": False,
    "use_beam_search": False,
    "length_penalty": 1.0,
    "cache_salt": None,
    "continue_final_message": False,
    "add_generation_prompt": True,
    "chat_template": None,
    "documents": None,
    "structured_outputs": None,
    "guided_json": None,
    "guided_regex": None,
    "guided_choice": None,
    "guided_grammar": None,
    "guided_decoding_backend": None,
    "prompt_logprobs": None,
    "echo": False,
    "vllm_xargs": None,
    "mm_processor_kwargs": None,
    "media_io_kwargs": None,
    "return_token_ids": False,
    "return_tokens_as_token_ids": False,
    "logits_processors": None,
    "bad_words": [],
    "allowed_token_ids": None,
    "truncate_prompt_tokens": None,
    "stop_token_ids": [],
    "skip_special_tokens": True,
    "include_stop_str_in_output": False,
    "spaces_between_special_tokens": True,
}
# vLLM sampling extras the API does not take: passed through as today (they do not touch the prompt).
PASS_THROUGH = ("top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "min_tokens",
                "ignore_eos", "seed")
CTK_SWITCHES = ("thinking", "enable_thinking")
# The API's image limits (deepseek-recipe-image ImageLimits), checked before images are fetched or decoded.
MAX_IMAGES, MAX_IMAGE_BYTES_TOTAL = 600, 64 << 20
# Image work (fetch, decode, preprocess) runs one request at a time on its own thread: the API's resolver
# decodes a request's images 8 at a time at full size (up to ~1.7 GiB), and requests waiting for it wait in
# asyncio, not in the default executor that converts and encodes every request.
IMAGE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsv41-images")
# Content parts that name media vLLM fetches, and the keys of their URLs.
MEDIA_PARTS = ("image_url", "input_image", "image", "video_url", "audio_url", "input_file", "file", "document")
URL_KEYS = ("url", "image_url", "file_url", "video_url", "audio_url")


class Fallback(Exception):
    """Send the original request to vLLM; args[0] is the reason."""


class ApiError(Exception):
    """Answer with the API's error: status and body."""

    def __init__(self, status: int, body: dict):
        super().__init__(status, body)
        self.status, self.body = status, body


def error_body(message: str, kind: str = "invalid_request_error") -> dict:
    """The API's error shape, the same for every protocol (error_response.rs)."""
    return {"error": {"message": message, "type": kind, "param": None, "code": kind}}


def upstream_error(status: int, payload) -> ApiError:
    """vLLM's error (an HTTP error body, or an error event of its token stream with the HTTP status in
    error.code) in the API's shape: a 4xx is a 400 invalid_request_error, anything else keeps vLLM's
    status as an internal_error."""
    err = payload.get("error") if isinstance(payload, dict) else None
    message = err.get("message") if isinstance(err, dict) and err.get("message") else str(payload)[:2000]
    if 400 <= status < 500:
        return ApiError(400, error_body(message))
    return ApiError(max(status, 500), error_body(message, "internal_error"))


def log(msg: str) -> None:
    """One stderr line; URLs lose their userinfo, query and fragment (credentials, presigned signatures)."""
    msg = re.sub(r"(?i)\b(https?://)(?:[^\s/?#@]*@)?([^\s?#]*)(?:[?#]\S*)?", r"\1\2", msg)
    print("dsv41-gateway: " + msg, file=sys.stderr, flush=True)


def error_kind(message: str) -> str:
    """An error message's leading field path and class, without the request values it may quote."""
    m = re.match(r"([.\w\[\]]+:\s*)?([^:,(\"'`]*)", message)
    return ((m.group(1) or "") + m.group(2)).strip()


def fallback_reason(body: dict) -> str | None:
    for key, noop in FALLBACK_FIELDS.items():
        if body.get(key) not in (None, "", [], {}, noop):  # empty values count as absent
            return f"vLLM field {key}"
    return None


def shim_chat(body: dict) -> str | None:
    """Chat fields vLLM honours today, mapped onto the DeepSeek ones so such requests keep working.
    Returns a fallback reason for chat_template_kwargs the map does not cover.

    max_completion_tokens is the max_tokens of a request without one. chat_template_kwargs carrying
    only thinking, enable_thinking and a string reasoning_effort are dropped after mapping. Without a
    top-level thinking or reasoning_effort, thinking is on only for a true switch (vLLM ORs the two;
    the serve's default kwargs keep it off) and off for effort "none", as in vLLM; the kwargs' effort
    fills a missing reasoning_effort. Top-level thinking and reasoning_effort keep the API's semantics."""
    if body.get("max_tokens") is None and body.get("max_completion_tokens") is not None:
        body["max_tokens"] = body["max_completion_tokens"]
    ctk = body.pop("chat_template_kwargs", None)
    if not ctk:
        return None
    effort = ctk.get("reasoning_effort") if isinstance(ctk, dict) else None
    if (not isinstance(ctk, dict) or set(ctk) - {*CTK_SWITCHES, "reasoning_effort"}
            or any(not isinstance(ctk[k], bool) for k in CTK_SWITCHES if k in ctk)
            or not isinstance(effort, (str, type(None)))):
        return f"chat_template_kwargs {sorted(ctk) if isinstance(ctk, dict) else type(ctk).__name__}"
    if body.get("thinking") is None and body.get("reasoning_effort") is None:
        on = any(ctk.get(k) for k in CTK_SWITCHES) and effort != "none"
        body["thinking"] = {"type": "enabled" if on else "disabled"}
    if effort is not None and body.get("reasoning_effort") is None:
        body["reasoning_effort"] = effort
    return None


def remote_media(node, part: bool = False) -> bool:
    """Whether a request names an http(s) URL in a media content part (an image, video, audio or file part
    of its messages or input): vLLM fetches those itself when it serves the request. Metadata, schemas
    and tool declarations hold no media parts."""
    if isinstance(node, dict):
        part = part or node.get("type") in MEDIA_PARTS
        return any(part and k in URL_KEYS and isinstance(v, str) and v[:8].lower().startswith(("http://", "https://"))
                   or remote_media(v, part) for k, v in node.items())
    return isinstance(node, list) and any(remote_media(v, part) for v in node)


def internal_route(path: str) -> bool:
    """vLLM's scale-out routes (token-in generation, render, derender): for the gateway, not the public."""
    return path.startswith("/inference/") or path.rstrip("/").endswith(("/render", "/derender"))


def public_address(host: str, address: str) -> bool:
    """Destination filter for image fetches (DSV41_GATEWAY_BLOCK_PRIVATE_URLS=1)."""
    ip = ipaddress.ip_address(address)
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_global and not (ip.is_multicast or ip.is_reserved)


def data_url_bytes(url: str) -> bytes:
    """The bytes of a data URL: base64 (whitespace and missing padding tolerated, as the API does) or
    percent-encoded. vLLM decodes strict base64 only, so the gateway re-encodes."""
    head, _, data = url.partition(",")
    if head.lower().endswith(";base64"):
        data = "".join(data.split())
        return base64.b64decode(data + "=" * (-len(data) % 4))
    return urllib.parse.unquote_to_bytes(data)


def api_logprobs(content: list, top_n: int) -> list:
    """vLLM's token-ID logprobs -> one (logprob, [(id, logprob)] best first) per token.

    vLLM lists the sampled token first among the alternatives and keeps max(k, 1) entries
    (v1/engine/logprobs.py, token_in_token_out/serving.py _create_tokens_logprobs), so the gateway asks
    for top_n + 1 (top_logprobs 20 needs vLLM --max-logprobs 21) and keeps the best top_n: the API's
    top_logprobs. top_n 0 keeps none."""
    out = []
    for entry in content:
        alts = sorted(((int(t["token"].split(":", 1)[1]), t["logprob"]) for t in entry["top_logprobs"]),
                      key=lambda alt: -alt[1])
        out.append((entry["logprob"], alts[:top_n]))
    return out


def bad_upstream(what: str) -> ApiError:
    return ApiError(502, error_body(f"the model server {what}", "internal_error"))


async def sse_events(lines):
    """Parsed `data:` objects of an SSE text stream, until [DONE]."""
    async for line in lines:
        if line.startswith("data: "):
            if line == "data: [DONE]":
                return
            try:
                ev = json.loads(line[6:])
            except ValueError:
                ev = None
            if not isinstance(ev, dict):
                raise bad_upstream(f"sent a malformed event: {line[:200]!r}")
            yield ev


async def events_after(first):
    """The events of a Gateway.first_event task still pending when a stream's headers went out: a
    rejection then raises inside the stream, which ends with the API-shaped error frame. The shield keeps
    the stream's (repeated) cancellations out of the task, which reads the upstream response: the
    response's cleanup cancels it once, then closes the response, so vLLM sees the disconnect."""
    async for ev in await asyncio.shield(first):
        yield ev


class Prepared:
    """One converted request: the /inference/v1/generate body and what the response needs."""

    def __init__(self, protocol, converted, model, prompt_tokens, greq, prompt, n_images):
        self.protocol, self.converted, self.model = protocol, converted, model
        self.id = ID_PREFIX[protocol] + uuid.uuid4().hex
        self.greq = {"request_id": self.id, **greq}
        self.prompt_tokens, self.prompt, self.n_images = prompt_tokens, prompt, n_images
        io = converted.inference_options
        self.top_logprobs = (io.top_logprobs or 0) if io.logprobs else None
        self.cached = 0  # vLLM's prefix-cache hits, from its final usage chunk
        self.context_error = None  # the API's context-limit 400, raised before any upstream call


class Gateway:
    def __init__(self, tokenizer: str, served_name: str, *, upstream: str = DEFAULT_UPSTREAM,
                 max_model_len: int = DEFAULT_MAX_MODEL_LEN, strict: bool = False,
                 block_private_urls: bool = False):
        self.served_name, self.upstream = served_name, upstream.rstrip("/")
        self.max_model_len, self.strict, self.block_private_urls = max_model_len, strict, block_private_urls
        special = d.SpecialTokens(image=PACK_IMAGE_TOKEN)
        self.options = d.ConversionOptions(default_thinking_mode=False).with_special_tokens(
            special).with_model_versions_by_name(
            dict.fromkeys((served_name, *MODEL_NAMES), d.ModelVersion.DeepseekV41Flash))
        self.tokenizer = d.Tokenizer.from_file(tokenizer)
        self.encoding = d.DeepseekV41Encoding().with_special_tokens(special).with_tokenizer(self.tokenizer)
        (self.eos,) = self.tokenizer.encode(d.SpecialToken.Eos.default_str())
        self.fetcher = d.ReqwestImageFetcher()
        if block_private_urls:
            self.fetcher = self.fetcher.with_destination_filter(public_address)
        self.resolver = d.ImageResolver(self.fetcher, d.OpenCvImagePreprocessor())

    def reject(self, status: int, body: dict, reason: str | None = None):
        """A request the API rejects: vLLM gets it, or the API's error under DSV41_GATEWAY_STRICT=1. The
        logged reason names the error's class and field path, never the request values it quotes."""
        if self.strict:
            raise ApiError(status, body)
        raise Fallback(reason or f"API {status}: {error_kind(body['error']['message'])}")

    async def prepare(self, protocol: str, body: dict) -> Prepared:
        """One request, ready for vLLM. Raises Fallback or ApiError. Conversion and encoding run in the
        default executor, image work on IMAGE_EXECUTOR."""
        cr, model, rendered, ids = await asyncio.to_thread(self.convert, protocol, body)
        n_images = len(rendered.image_sources)
        adjust, parts = 0, None
        if n_images:
            adjust, parts = await asyncio.get_running_loop().run_in_executor(
                IMAGE_EXECUTOR, self.images, rendered.image_sources)
        context_error = None
        try:
            max_tokens = cr.resolve_max_tokens(len(ids) + adjust + n_images, self.max_model_len)
        except d.ConversionError as e:  # the API's context-limit 400 (vLLM would reject it too)
            max_tokens, context_error = None, ApiError(e.status_code, json.loads(e.body))
        constraint = structured_outputs_for(cr, thinking=cr.conversation.thinking_mode)
        if constraint is FALLBACK:
            raise Fallback("structured output needs vLLM's grammar gating")
        io = cr.inference_options
        sp = {k: body[k] for k in PASS_THROUGH if body.get(k) is not None}
        sp.update(max_tokens=max_tokens, temperature=io.temperature, top_p=io.top_p, detokenize=False)
        if io.seed is not None:
            sp["seed"] = io.seed
        if io.logprobs:  # one more than the API's top_logprobs (api_logprobs): vLLM --max-logprobs 21
            sp["logprobs"] = io.top_logprobs + 1 if io.top_logprobs else 0
        if constraint:
            sp["structured_outputs"] = constraint
        greq = {"model": self.served_name, "token_ids": ids, "sampling_params": sp, "stream": True,
                "stream_options": {"include_usage": True}}
        if parts:
            greq["content_parts"] = parts
        prep = Prepared(protocol, cr, model, len(ids) + adjust, greq, rendered.prompt, n_images)
        prep.context_error = context_error  # /dsv41/render still answers (max_tokens null)
        return prep

    def convert(self, protocol: str, body: dict):
        """The fallback rules, the API's conversion, render and encode (a worker thread): (converted
        request, model, rendered prompt, token IDs). Raises Fallback or ApiError."""
        reason = fallback_reason(body)
        if protocol == "chat":
            reason = reason or shim_chat(body)
        elif body.get("chat_template_kwargs"):
            reason = reason or "vLLM field chat_template_kwargs"
        if reason:
            raise Fallback(reason)
        req_cls, _ = PROTOCOLS[protocol]
        try:
            req = req_cls(body)
            model = req.model
            cr = req.convert(self.options)
        except d.ConversionError as e:
            self.reject(e.status_code, json.loads(e.body))
        except ValueError as e:  # the body does not deserialize
            self.reject(400, error_body(str(e)))
        rendered = self.encoding.render_conversation(cr.conversation)
        try:
            ids = self.encoding.encode(cr.conversation)
        except RuntimeError as e:
            self.reject(500, error_body(str(e), "internal_error"))
        return cr, model, rendered, ids

    def images(self, sources) -> tuple[int, list]:
        """The API's image token adjustment, and vLLM content parts carrying each original image as a
        data URL in placeholder order (IMAGE_EXECUTOR). URL images are fetched once, here, within the API's
        count and total-size limits; the API's resolver then checks and preprocesses them in one call."""
        if len(sources) > MAX_IMAGES:
            self.reject(400, error_body(f"{sources[MAX_IMAGES].path}: Too many images: max {MAX_IMAGES} images "
                                        f"per request, got {len(sources)}"), "image count")
        blobs, api_sources, fetched = [], [], 0
        try:
            for s in sources:
                if s.kind == "url":
                    blob = self.fetcher.fetch(s.url)
                    fetched += len(blob)
                    if fetched > MAX_IMAGE_BYTES_TOTAL:  # the API rejects the request: stop fetching
                        self.reject(400, error_body(f"{s.path}: Total image size exceeds the limit: max "
                                                    f"{MAX_IMAGE_BYTES_TOTAL >> 20} MB per request"), "image bytes")
                    api_sources.append(d.BytesImageSource(blob, s.detail, path=s.path))
                else:
                    blob = s.data if s.kind == "bytes" else None
                    api_sources.append(s)
                blobs.append(blob)
            adjust = self.resolver.resolve(api_sources, d.ImageQuota()).image_token_adjustment()
            blobs = [b if b is not None else data_url_bytes(s.data_url) for b, s in zip(blobs, sources)]
        except d.ImageError as e:
            if e.kind == "DestinationRejected":  # never hand the URL to vLLM, which would fetch it
                raise ApiError(400, error_body(str(e))) from None
            if e.kind in ("Client", "TokenBudget"):  # server-py's split of the image errors
                self.reject(500, error_body(str(e), "internal_error"), f"image {e.kind}")
            self.reject(400, error_body(str(e)), f"image {e.kind}")
        except d.CalcResizeError as e:
            self.reject(500, error_body(str(e), "internal_error"), "image token budget")
        except ValueError as e:  # a data URL the API took that Python cannot decode
            self.reject(400, error_body(f"image data URL: {e}"), "image data URL")
        return adjust, [{"type": "image_url", "url": "data:application/octet-stream;base64,"
                         + base64.b64encode(b).decode()} for b in blobs]

    @staticmethod
    async def first_event(events):
        """Wait for vLLM's first stream event and return the stream from it. vLLM answers 200 before it
        hands the request to the engine, which reports a rejection (a grammar it cannot compile, a
        logprobs or context limit) as the first event: that one raises ApiError, an HTTP error while the
        gateway's response has not started (non-stream, or a stream within FIRST_EVENT_GRACE_S)."""
        first = await anext(events, None)
        if first is None:
            raise bad_upstream("ended the stream without output")
        if "error" in first:
            code = (first["error"] if isinstance(first["error"], dict) else {}).get("code")
            raise upstream_error(code if isinstance(code, int) else 500, first)

        async def stream():
            yield first
            async for ev in events:
                yield ev

        return stream()

    async def run(self, prep: Prepared, events):
        """The API's protocol chunks for vLLM's token stream `events` (parsed SSE objects). Returns
        early when the processor finishes before vLLM does (a stop sequence): the caller then closes
        the upstream stream, which aborts the request."""
        req_cls, _ = PROTOCOLS[prep.protocol]
        proc = d.StreamProcessor(req_cls.chunk_generator(prep.converted, prep.id, prep.model),
                                 prep.converted.parsing_options, self.tokenizer)
        try:
            for ch in proc.push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=prep.prompt_tokens))):
                yield ch
            finish = None
            async for ev in events:
                if "error" in ev:  # after output started: the stream ends with the API-shaped error
                    code = (ev["error"] if isinstance(ev["error"], dict) else {}).get("code")
                    raise upstream_error(code if isinstance(code, int) else 500, ev)
                for c in ev.get("choices") or ():
                    ids, lps = c.get("token_ids") or [], None
                    if prep.top_logprobs is not None:
                        lps = api_logprobs(c["logprobs"]["content"], prep.top_logprobs)
                    finish = c.get("finish_reason") or finish
                    if c.get("finish_reason") == "stop" and ids and ids[-1] == self.eos:
                        ids, lps = ids[:-1], lps and lps[:-1]  # the API never pushes the stop token
                    if ids:
                        for ch in proc.push(d.InferenceChunk.tokens(ids, lps)):
                            yield ch
                    if proc.finished:
                        return
                if ev.get("usage"):
                    prep.cached = (ev["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            if finish not in FINISH:  # an engine abort or shutdown (vLLM drops a token-less abort output)
                raise bad_upstream(f"ended the stream without a stop or length finish ({finish})")
            for ch in proc.push(d.InferenceChunk.finish(FINISH[finish])) + proc.finish():
                yield ch
        finally:
            proc.close()

    def with_cache_hits(self, prep: Prepared, text: str) -> str:
        """Write vLLM's prefix-cache hits into the usage the processor computed with none (vLLM reports
        them only in its final usage chunk, after the processor's start event)."""
        if not prep.cached:
            return text
        p, c = prep.prompt_tokens, min(prep.cached, prep.prompt_tokens)
        cached = '{"cached_tokens":%d}'
        edits = {
            "chat": (('"prompt_tokens_details":' + cached % 0, '"prompt_tokens_details":' + cached % c),
                     ('"prompt_cache_hit_tokens":0', f'"prompt_cache_hit_tokens":{c}'),
                     (f'"prompt_cache_miss_tokens":{p}', f'"prompt_cache_miss_tokens":{p - c}')),
            "messages": ((f'"input_tokens":{p},"cache_creation_input_tokens":0,"cache_read_input_tokens":0',
                          f'"input_tokens":{p - c},"cache_creation_input_tokens":0,"cache_read_input_tokens":{c}'),),
            "responses": (('"input_tokens_details":' + cached % 0, '"input_tokens_details":' + cached % c),),
        }[prep.protocol]
        for old, new in edits:
            text = text.replace(old, new)
        return text

    async def sse_frames(self, prep: Prepared, events):
        """The API's SSE stream for one request."""
        _, resp_cls = PROTOCOLS[prep.protocol]
        try:
            async with contextlib.aclosing(self.run(prep, events)) as chunks:
                async for ch in chunks:
                    name = resp_cls.sse_event_name(ch)
                    yield (f"event: {name}\n" if name else "") + f"data: {self.with_cache_hits(prep, ch.to_json())}\n\n"
        except Exception as e:  # vLLM's error event, a broken upstream stream, or a processor failure
            body = e.body if isinstance(e, ApiError) else error_body(str(e) or repr(e), "internal_error")
            log(f"{prep.id} failed: {error_kind(body['error']['message'])}")
            yield ("" if prep.protocol == "chat" else "event: error\n") + f"data: {json.dumps(body)}\n\n"
            return
        done = resp_cls.sse_done_data()
        if done:
            yield f"data: {done}\n\n"

    async def full_body(self, prep: Prepared, events) -> str:
        """The API's response body (JSON text) for one request."""
        _, resp_cls = PROTOCOLS[prep.protocol]
        response = resp_cls(prep.id, prep.model, int(time.time()), 0, 0)
        async with contextlib.aclosing(self.run(prep, events)) as chunks:
            async for ch in chunks:
                response.accumulate(ch)
        return self.with_cache_hits(prep, response.to_json())

    def render_info(self, prep: Prepared) -> dict:
        """/dsv41/render: what the API path sends upstream (max_tokens null past the context limit)."""
        conv, sp = prep.converted.conversation, prep.greq["sampling_params"]
        return {"prompt": prep.prompt, "prompt_token_ids": prep.greq["token_ids"], "image_count": prep.n_images,
                "thinking": conv.thinking_mode, "effort": conv.reasoning_effort, "sampling": sp,
                "max_tokens": sp["max_tokens"], "fallback_reason": None}


# Hop-by-hop headers, and the one the proxy hop recomputes (the client's Host goes through: vLLM's
# redirects point at it).
HOP_HEADERS = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"proxy-connection",
               b"te", b"trailer", b"transfer-encoding", b"upgrade"}
REQUEST_SKIP = HOP_HEADERS | {b"content-length"}
RESPONSE_SKIP = HOP_HEADERS | {b"date"}  # uvicorn writes its own date header
API_HEADER = {"x-dsv41-gateway": "api"}
RENDER_FIELDS = ("prompt", "prompt_token_ids", "image_count", "thinking", "effort", "sampling", "max_tokens")
# A stream's headers wait this long for vLLM's first event (engine rejections come at once), not the prefill.
FIRST_EVENT_GRACE_S = 2.0


async def until_disconnect(request, coro):
    """Await coro unless the client goes away first; then cancel it and return None."""
    async def disconnected():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    work, watch = asyncio.ensure_future(coro), asyncio.ensure_future(disconnected())
    try:
        await asyncio.wait((work, watch), return_when=asyncio.FIRST_COMPLETED)
    finally:
        watch.cancel()
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
    return None if work.cancelled() else work.result()


def make_app(gw: Gateway):
    """The HTTP layer: the three API routes, /health, /dsv41/render/{protocol}, a proxy for the rest."""
    import anyio
    import httpx
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route

    client = None

    class Stream(StreamingResponse):
        """When the response ends or the client goes away: close the body iterator, then the upstream
        response (and cancel its pending reads), which aborts vLLM's request."""

        def __init__(self, content, resp, *tasks, **kwargs):
            super().__init__(content, **kwargs)
            self.upstream = (resp, *tasks)

        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                with anyio.CancelScope(shield=True):
                    await self.body_iterator.aclose()
                    await close(*self.upstream)

    async def close(resp, *tasks):
        for task in tasks:
            task.cancel()
        with anyio.CancelScope(shield=True):
            await resp.aclose()  # unread: the connection closes and vLLM aborts the request

    async def lines(resp):
        try:
            async for line in resp.aiter_lines():
                yield line
        except httpx.HTTPError as e:  # a dropped or broken upstream connection
            raise bad_upstream(f"connection failed: {e!r}") from None

    async def proxy(request, raw, extra=()):
        """Send a request to vLLM as it came (method, raw path and query, headers, body) and stream the
        answer back. Until vLLM's headers arrive (a non-stream answer), a disconnect cancels the call."""
        if gw.block_private_urls:
            try:
                remote = remote_media(json.loads(raw))
            except (ValueError, RecursionError):
                remote = False
            if remote:  # vLLM would fetch it without the gateway's destination filter
                log(f"refused {request.method} {request.url.path}: an http(s) media URL with "
                    "DSV41_GATEWAY_BLOCK_PRIVATE_URLS=1")
                return JSONResponse(error_body("Image URLs are only accepted where the API serves the request "
                                               "itself (DSV41_GATEWAY_BLOCK_PRIVATE_URLS=1)"), 400)
        query = request.scope.get("query_string") or b""
        url = gw.upstream + request.scope["raw_path"].decode("latin-1") + ("?" + query.decode("latin-1") if query else "")
        headers = [(k, v) for k, v in request.headers.raw if k.lower() not in REQUEST_SKIP]
        try:
            resp = await until_disconnect(request, client.send(
                client.build_request(request.method, url, headers=headers, content=raw), stream=True))
        except httpx.HTTPError as e:
            return JSONResponse(error_body(f"upstream {gw.upstream}: {e!r}", "internal_error"), 502)
        if resp is None:
            return Response(status_code=499)  # the client went away; nothing is sent
        out = Stream(resp.aiter_raw(), resp, status_code=resp.status_code)
        out.raw_headers = [(k, v) for k, v in resp.headers.raw if k.lower() not in RESPONSE_SKIP] + list(extra)
        return out

    async def passthrough(request):
        if internal_route(request.url.path):  # vLLM's scale-out routes serve the gateway, not the public
            return JSONResponse({"detail": "Not Found"}, 404)
        return await proxy(request, await request.body())

    async def fallback(request, raw, protocol, reason):
        log(f"fallback {protocol}: {reason:.300}")
        return await proxy(request, raw, [(b"x-dsv41-gateway", b"fallback")])

    async def prepare(protocol, raw):
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            gw.reject(400, error_body("The request body is not a JSON object"))
        try:
            return await gw.prepare(protocol, body)
        except (Fallback, ApiError):
            raise
        except Exception as e:  # a gateway bug must not break what vLLM serves today
            gw.reject(500, error_body(f"gateway: {e!r}", "internal_error"), f"gateway {type(e).__name__}")

    async def api(request):
        protocol = ROUTES[request.url.path]
        raw = await request.body()
        try:
            prep = await prepare(protocol, raw)
            if prep.context_error:
                raise prep.context_error
        except Fallback as f:
            return await fallback(request, raw, protocol, str(f))
        except ApiError as e:
            return JSONResponse(e.body, e.status, headers=API_HEADER)
        try:
            resp = await until_disconnect(request, client.send(client.build_request(
                "POST", gw.upstream + GENERATE, content=json.dumps(prep.greq).encode(),
                headers={"content-type": "application/json"}), stream=True))
        except httpx.HTTPError as e:
            return JSONResponse(error_body(f"upstream {gw.upstream}: {e!r}", "internal_error"), 502,
                                headers=API_HEADER)
        if resp is None:
            return Response(status_code=499)
        first, streaming = None, False
        try:
            if resp.status_code != 200:
                content = await resp.aread()
                try:
                    payload = json.loads(content)
                except ValueError:
                    payload = content.decode(errors="replace")
                raise upstream_error(resp.status_code, payload)
            first = asyncio.ensure_future(gw.first_event(sse_events(lines(resp))))
            if prep.converted.stream:
                if await until_disconnect(request, asyncio.wait({first}, timeout=FIRST_EVENT_GRACE_S)) is None:
                    return Response(status_code=499)  # the client went away; nothing is sent
                events = first.result() if first.done() else events_after(first)
                streaming = True  # the response closes the upstream stream from here on
                return Stream(gw.sse_frames(prep, events), resp, first, media_type="text/event-stream",
                              headers=API_HEADER)
            events = await until_disconnect(request, first)  # a non-stream response waits for it
            text = None if events is None else await until_disconnect(request, gw.full_body(prep, events))
            if text is None:
                return Response(status_code=499)
            return Response(text, media_type="application/json", headers=API_HEADER)
        except Exception as e:  # vLLM's error, a broken upstream stream, or a processor failure
            err = e if isinstance(e, ApiError) else ApiError(500, error_body(str(e) or repr(e), "internal_error"))
            log(f"{prep.id} failed: {err.status} {error_kind(err.body['error']['message'])}")
            return JSONResponse(err.body, err.status, headers=API_HEADER)
        finally:
            if not streaming:
                await close(resp, *([first] if first else []))

    async def health(request):
        try:
            r = await client.get(gw.upstream + "/health", timeout=10)
            upstream = "ok" if r.status_code == 200 else f"status {r.status_code}"
        except httpx.HTTPError as e:
            upstream = f"unreachable: {e!r}"
        return JSONResponse({"gateway": "ok", "upstream": upstream}, 200 if upstream == "ok" else 503)

    async def render(request):
        protocol = request.path_params["protocol"]
        if protocol not in PROTOCOLS:
            return JSONResponse(error_body(f"unknown protocol {protocol!r}"), 404)
        try:
            prep = await prepare(protocol, await request.body())
        except Fallback as f:
            return JSONResponse({**dict.fromkeys(RENDER_FIELDS), "fallback_reason": str(f)})
        except ApiError as e:
            return JSONResponse(e.body, e.status)
        return JSONResponse(gw.render_info(prep))

    @contextlib.asynccontextmanager
    async def lifespan(app):
        nonlocal client
        # No pool cap (open streams never starve /health or new requests); idle connections expire before
        # vLLM's uvicorn drops them (keep-alive 5 s), so a reused one is never closed under a request.
        limits = httpx.Limits(max_connections=None, keepalive_expiry=2.0)
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0), limits=limits) as client:
            client.headers.clear()  # a proxied request carries the client's headers only
            yield

    routes = [Route(path, api, methods=["POST"]) for path in ROUTES]
    routes += [Route("/health", health, methods=["GET"]),
               Route("/dsv41/render/{protocol}", render, methods=["POST"]),
               Route("/{path:path}", passthrough, methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])]
    return Starlette(routes=routes, lifespan=lifespan)


def _flag(env, name: str) -> bool:
    value = env.get(name) or "0"
    if value not in ("0", "1"):
        raise SystemExit(f"{name}={value!r}: want 0 or 1")
    return value == "1"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dsv41 API gateway: deepseek-recipe in front of vLLM")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tokenizer", required=True, help="the pack's tokenizer.json")
    ap.add_argument("--served-model-name", required=True, help="vLLM's --served-model-name")
    args = ap.parse_args(argv)
    env = os.environ
    gw = Gateway(args.tokenizer, args.served_model_name, upstream=env.get(ENV_UPSTREAM) or DEFAULT_UPSTREAM,
                 max_model_len=int(env.get(ENV_MAX_MODEL_LEN) or DEFAULT_MAX_MODEL_LEN),
                 strict=_flag(env, ENV_STRICT), block_private_urls=_flag(env, ENV_BLOCK_PRIVATE_URLS))
    print(f"{LOG_ENGAGED} {d.__version__} upstream={gw.upstream} model={gw.served_name}", file=sys.stderr, flush=True)
    import uvicorn

    uvicorn.run(make_app(gw), host=args.host, port=args.port, server_header=False, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
