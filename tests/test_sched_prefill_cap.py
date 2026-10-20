"""docker/patch/sched_prefill_cap.py: prefill cap with co-tenants + decode-only steps.

The scheduler under test is e15's own code: tests/fixtures/sched_e15.pin.py holds the
image's Scheduler.schedule and the helpers it calls, verbatim; build_pinned() assembles
them into Scheduler / AsyncScheduler classes (methods under one class header) next to
fakes for what schedule() only passes through (KV cache manager, encoder cache, output
containers). Engine drives them like EngineCore: depth 1 = synchronous scheduling, depth
2 = async scheduling (step N+1 is scheduled before step N's output is processed, as
step_with_batch_queue does). Every draft is accepted, so token arithmetic stays exact.
"""

from __future__ import annotations

import abc
import ast
import collections
import contextlib
import enum
import io
import itertools
import re
import sys
import time
import types
import unittest
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
FIX = ROOT / "tests" / "fixtures" / "sched_e15.pin.py"
sys.path.insert(0, str(PATCH))

import sched_prefill_cap as spc  # noqa: E402

B = 8192
K = 3  # DSpark num_speculative_tokens
DRAFT_SLOTS = K - 1  # DSpark max_num_new_slots_for_drafting
HEADER = re.compile(r"^# ---- (\S+):(\d+) (\S+)$")
PIN: dict = {}


def segments() -> dict[str, str]:
    """{name: verbatim source} of tests/fixtures/sched_e15.pin.py, by '# ---- file:line name'."""
    out, name, buf = {}, None, []
    for line in FIX.read_text().splitlines():
        m = HEADER.match(line)
        if m:
            if name is not None:
                out[name] = "\n".join(buf).rstrip() + "\n"
            name, buf = m.group(3), []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out[name] = "\n".join(buf).rstrip() + "\n"
    return out


class _Stub:
    """Placeholder for names the pinned code only annotates with."""


class FakeBlocks:
    def get_block_ids(self, allow_none: bool = False):
        return ([],)


class FakeKVM:
    """The KVCacheManager calls schedule() makes; every allocation succeeds."""

    empty_kv_cache_blocks = FakeBlocks()

    def __init__(self) -> None:
        self.fail_next = False

    def new_step_starts(self) -> None:
        pass

    def allocate_slots(self, request, num_new_tokens, **kwargs):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("allocate_slots failed")
        return FakeBlocks()

    def get_computed_blocks(self, request):
        return FakeBlocks(), 0, 0

    def record_prefix_cache_stats(self, request, num_hits) -> None:
        pass

    def get_blocks(self, request_id):
        return FakeBlocks()

    def get_num_common_prefix_blocks(self, request_id):
        return [0]

    def take_boundary_state_offloads(self):
        return {}

    def take_kv_cache_block_copies(self):
        return [], []

    def take_new_block_ids(self):
        return []


class FakeEncoderCache:
    def get_freed_mm_hashes(self):
        return []

    def get_manager_metadata(self):
        return None


class SchedulerOutput:
    """e15 SchedulerOutput as a plain attribute bag (constructed with keywords at :1344)."""

    def __init__(self, **kw) -> None:
        self.has_structured_output_requests = False
        self.pending_structured_output_tokens = False
        self.__dict__.update(kw)


class NewRequestData:
    @staticmethod
    def from_request(req, block_ids, *args, **kwargs):
        return types.SimpleNamespace(req_id=req.request_id, block_ids=block_ids)


class CachedRequestData(types.SimpleNamespace):
    pass


class FakeRequest:
    """The e15 Request fields and properties the pinned scheduler reads (v1/request.py:59-330)."""

    def __init__(self, rid: str, prompt_len: int, max_tokens: int) -> None:
        self.request_id = rid
        self.prompt_token_ids = [1] * prompt_len
        self._all_token_ids = list(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self.all_token_ids = self._all_token_ids
        self.output_token_ids = self._output_token_ids
        self.num_prompt_tokens = prompt_len
        self.max_tokens = max_tokens
        self.status = PIN["RequestStatus"].WAITING
        self.num_computed_tokens = 0
        self.num_output_placeholders = 0
        self.num_stale_output_tokens = 0
        self.drop_stale_output = False
        self.num_in_flight_tokens = 0
        self.next_decode_eligible_step = 0
        self.last_sched_seq = 0
        self.spec_token_ids: list[int] = []
        self.is_prefill_chunk = False
        self.shared_prefix_boundary = 0
        self.num_preemptions = 0
        self.prefill_stats = None
        self.spec_decode_metrics = None
        self.lora_request = None
        self.mm_features: list = []
        self.priority = 0
        self.arrival_time = time.time()
        self.use_structured_output = False
        self.resumable = False
        self.streaming_queue = None
        self.has_encoder_inputs = False

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def append_output_token_ids(self, token_ids) -> None:
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)


def build_pinned() -> dict:
    """Namespace with the pinned e15 classes; Scheduler / AsyncScheduler run the image's code."""
    if PIN:
        return PIN
    seg = segments()
    ns = {
        "__name__": "e15_sched_pin", "enum": enum, "Enum": enum.Enum, "ABC": abc.ABC,
        "abstractmethod": abc.abstractmethod, "Iterable": Iterable, "Iterator": Iterator,
        "deque": collections.deque, "time": time, "itertools": itertools, "Any": object,
        "Request": FakeRequest, "FinishReason": _Stub, "KVCacheBlocks": FakeBlocks,
        "SchedulerOutput": SchedulerOutput, "NewRequestData": NewRequestData,
        "CachedRequestData": CachedRequestData, "KVConnectorBlockState": _Stub,
        "EngineCoreEventType": _Stub, "StreamingUpdate": _Stub, "RequestSpecDecodeMetrics": _Stub,
        "record_function_or_nullcontext": lambda name: contextlib.nullcontext(),
    }
    for name in ("RequestStatus", "PauseState", "SchedulingPolicy", "RequestQueue", "FCFSRequestQueue",
                 "create_request_queue"):
        exec(compile(seg[name], f"<e15 pin {name}>", "exec"), ns)
    methods = [src for name, src in seg.items() if name.startswith("Scheduler.")]
    exec(compile("class Scheduler:\n" + "\n".join(methods), "<e15 pin Scheduler>", "exec"), ns)
    exec(compile(seg["AsyncScheduler"], "<e15 pin AsyncScheduler>", "exec"), ns)
    PIN.update(ns)
    return PIN


def make_scheduler(async_: bool = True, lpt: int = 0, max_seqs: int = 4, budget: int = B):
    """A pinned Scheduler / AsyncScheduler with the recipe's knobs (B 8192, 4 seqs, DSpark k=3, MRV2)."""
    ns = build_pinned()
    cls = ns["AsyncScheduler"] if async_ else ns["Scheduler"]
    s = object.__new__(cls)
    s.scheduler_config = types.SimpleNamespace(
        max_num_batched_tokens=budget, long_prefill_token_threshold=lpt, enable_chunked_prefill=True,
        async_scheduling=async_)
    s.vllm_config = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(max_num_new_slots_for_drafting=DRAFT_SLOTS))
    s.max_num_scheduled_tokens = budget
    s.max_num_running_reqs = max_seqs
    s.max_model_len = 500_000
    s.num_spec_tokens = K
    s.num_lookahead_tokens = K
    s.num_prefill_lookahead = 1  # DSpark is eagle-family, not multi-module MTP (scheduler.py:276-282)
    s.num_sampled_tokens_per_step = 1
    s.dynamic_sd_lookup = None
    s._pause_state = ns["PauseState"].UNPAUSED
    s.max_num_encoder_input_tokens = 0
    s.kv_cache_manager = FakeKVM()
    s.encoder_cache_manager = FakeEncoderCache()
    s.policy = ns["SchedulingPolicy"].FCFS
    s.waiting = ns["create_request_queue"](s.policy)
    s.skipped_waiting = ns["create_request_queue"](s.policy)
    s.running = []
    s.requests = {}
    s.current_step = 0
    s.prefill_capacity_bound = False
    s.need_mamba_block_aligned_split = False
    s.requires_kv_delivery = False
    s.lora_config = None
    s.connector = None
    s.ec_connector = None
    s.connector_prefix_cache_stats = None
    s.log_stats = False
    s.num_waiting_for_streaming_input = 0
    s.scheduler_reserve_full_isl = True
    s.kv_cache_config = types.SimpleNamespace(kv_cache_groups=[None])
    s.is_encoder_decoder = False
    s.use_v2_model_runner = True
    s.model_uses_mrope = False
    s.model_uses_xdrope = False
    s.prev_step_scheduled_req_ids = set()
    s.use_pp = False
    s.reset_preempted_req_ids = set()
    s.finished_req_ids = set()
    s.defer_block_free = False
    s.sched_step_seq = 0
    s.enable_return_routed_experts = False
    s._inflight_prefills = set()
    s.needs_kv_cache_zeroing = False
    s._skip_zero_block_ids = set()
    s.spec_decode_metrics_level = "none"
    s.pp_size = 1  # AsyncScheduler.__init__
    s._spec_token_placeholders = [-1] * K
    return s


def arm(s, cap: int = 0, steps: int = 0):
    """Bind the lever's wrapper on this instance only (the pinned class stays stock)."""
    s.schedule = types.MethodType(spc.wrap(type(s).schedule, cap, steps), s)
    return s


V41 = types.SimpleNamespace(kv_source_layer_ids=[2, 8, 14, 20], num_hidden_layers=40, sliding_window=128)


def with_replay(s, rows: int = 2606):
    """What decoder_replay.install() leaves on a scheduler: its guard on the KV cache manager."""
    s.kv_cache_manager._dsv41_replay = types.SimpleNamespace(rows=rows)
    s.vllm_config.model_config = types.SimpleNamespace(hf_text_config=V41)
    return s


class Engine:
    """EngineCore's loop over the pinned scheduler: schedule, then process outputs `depth` steps late."""

    def __init__(self, s) -> None:
        self.s = s
        self.depth = 2 if s.scheduler_config.async_scheduling else 1
        self.inflight: collections.deque = collections.deque()
        self.steps: list[dict[str, int]] = []
        self.base = s.scheduler_config.long_prefill_token_threshold

    def add(self, rid: str, prompt: int, max_tokens: int = 10_000) -> None:
        self.s.add_request(FakeRequest(rid, prompt, max_tokens))

    def step(self) -> dict[str, int]:
        out = self.s.schedule()
        assert self.s.scheduler_config.long_prefill_token_threshold == self.base, "threshold not restored"
        sampled = {}
        for rid in out.num_scheduled_tokens:
            r = self.s.requests[rid]
            sampled[rid] = 0 if r.is_prefill_chunk else 1 + len(out.scheduled_spec_decode_tokens.get(rid, ()))
        self.inflight.append(sampled)
        while len(self.inflight) >= self.depth:
            self._process(self.inflight.popleft())
        self.steps.append(dict(out.num_scheduled_tokens))
        return self.steps[-1]

    def _process(self, sampled: dict[str, int]) -> None:
        running = PIN["RequestStatus"].RUNNING
        for rid, n in sampled.items():
            r = self.s.requests[rid]
            if n == 0 or r.status != running:
                continue
            r.append_output_token_ids([7] * n)
            if self.depth == 2:
                r.num_output_placeholders -= n
            else:
                r.spec_token_ids = [0] * K  # update_draft_token_ids (EngineCore.post_step)
            if r.num_output_tokens >= r.max_tokens:
                r.status = PIN["RequestStatus"].FINISHED_LENGTH_CAPPED
                self.s.running.remove(r)
                self.s.finished_req_ids.add(rid)

    def run(self, n: int) -> list[dict[str, int]]:
        return [self.step() for _ in range(n)]

    def until_prefilled(self, rid: str, limit: int = 2000) -> list[dict[str, int]]:
        """Steps up to and including the one that schedules the last prompt token of rid."""
        out, r = [], self.s.requests[rid]
        for _ in range(limit):
            out.append(self.step())
            if r.status == PIN["RequestStatus"].RUNNING and not r.is_prefill_chunk and rid in out[-1]:
                return out
        raise AssertionError(f"{rid} did not finish its prefill in {limit} steps")


def decoders(eng: Engine, n: int = 3, prompt: int = 100) -> list[str]:
    """n short sessions admitted and decoding (each schedules 1 + k tokens per step)."""
    ids = [f"d{i}" for i in range(n)]
    for rid in ids:
        eng.add(rid, prompt)
    eng.run(3)
    assert all(eng.steps[-1].get(rid) == 1 + K for rid in ids), eng.steps[-1]
    return ids


class PinTests(unittest.TestCase):
    def test_pin_has_every_segment_once(self) -> None:
        seg = segments()
        text = FIX.read_text()
        for name in ("Scheduler.schedule", "Scheduler._update_after_schedule", "Scheduler.get_request_counts",
                     "AsyncScheduler", "FCFSRequestQueue", "RequestStatus", "PauseState"):
            self.assertIn(name, seg)
            self.assertEqual(text.count(f" {name}\n"), 1, name)

    def test_contract_holds_on_the_pinned_scheduler(self) -> None:
        seg = segments()
        src = seg["Scheduler.schedule"]
        params = tuple(next(n for n in ast.parse(src.strip()).body).args.args)
        names = tuple(a.arg for a in params)
        async_defs = {n.name for n in ast.parse(seg["AsyncScheduler"]).body[0].body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(names, ("self", "throttle_prefills"))
        self.assertNotIn("schedule", async_defs)
        self.assertIsNone(spc.contract_problem(src, names, "schedule" in async_defs, True, steps=2))
        for line in spc.CONTRACT + spc.CONTRACT_STEPS:
            self.assertIn(line, {x.strip() for x in src.splitlines()}, line)

    def test_contract_refuses_drift(self) -> None:
        src = segments()["Scheduler.schedule"]
        params = ("self", "throttle_prefills")
        for line in spc.CONTRACT:
            self.assertIn("lacks", spc.contract_problem(src.replace(line, "pass"), params, False, True, 0))
        steps_line = spc.CONTRACT_STEPS[1]
        self.assertIsNone(spc.contract_problem(src.replace(steps_line, "pass"), params, False, True, 0))
        self.assertIn("lacks", spc.contract_problem(src.replace(steps_line, "pass"), params, False, True, 1))
        self.assertIn("throttle", spc.contract_problem(src, ("self",), False, True, 0))
        self.assertIn("overrides", spc.contract_problem(src, params, True, True, 0))
        self.assertIn("get_request_counts", spc.contract_problem(src, params, False, False, 0))


class StockSchedulerTests(unittest.TestCase):
    """What the pinned e15 scheduler does today (the facts the lever addresses)."""

    def test_a_long_prefill_after_decoders_takes_the_rest_of_the_budget(self) -> None:
        for async_ in (True, False):
            eng = Engine(make_scheduler(async_))
            ids = decoders(eng)
            eng.add("p", 40_000)
            steps = eng.until_prefilled("p")
            rest = B - 3 * (1 + K + DRAFT_SLOTS) - DRAFT_SLOTS  # input_budget - draft slots: 8172
            self.assertEqual([st["p"] for st in steps], [rest] * 4 + [40_000 - 4 * rest], async_)
            for st in steps:
                self.assertEqual([st[r] for r in ids], [1 + K] * 3)  # decoders ride every chunk

    def test_a_new_turn_waits_for_the_prefills_last_chunk(self) -> None:
        eng = Engine(make_scheduler())
        decoders(eng, 2)
        eng.add("p", 40_000)
        eng.step()
        eng.add("e", 500)
        steps = eng.until_prefilled("p")
        first_e = next(i for i, st in enumerate(steps) if "e" in st)
        self.assertEqual(first_e, len(steps) - 1)  # admitted only next to the prefill's last chunk

    def test_decoders_admitted_behind_a_running_prefill_wait_too(self) -> None:
        eng = Engine(make_scheduler())
        eng.add("p", 30_000)
        eng.step()
        for rid in ("d0", "d1", "d2"):
            eng.add(rid, 100)
        steps = eng.until_prefilled("p")
        self.assertTrue(all(len(st) == 1 for st in steps[:-1]), steps)  # the prefill alone, every chunk

    def test_a_second_prefill_starves_until_the_first_ends(self) -> None:
        eng = Engine(make_scheduler())
        eng.add("p1", 30_000)
        eng.add("p2", 20_000)
        steps = eng.until_prefilled("p1")
        self.assertEqual(sum("p2" in st for st in steps), 1)

    def test_long_prefill_threshold_caps_a_lone_prefill_too(self) -> None:
        eng = Engine(make_scheduler(lpt=2048))
        eng.add("p", 10_000)
        steps = eng.until_prefilled("p")
        self.assertEqual([st["p"] for st in steps], [2048] * 4 + [10_000 - 4 * 2048])


class CapTests(unittest.TestCase):
    def test_a_lone_request_is_scheduled_exactly_as_stock(self) -> None:
        for async_ in (True, False):
            runs = []
            for cap, steps in ((0, 0), (2048, 0), (2048, 2), (512, 1)):
                s = make_scheduler(async_)
                if cap or steps:
                    arm(s, cap, steps)
                eng = Engine(s)
                eng.add("p", 30_000, max_tokens=40)
                runs.append(eng.run(30))
            self.assertTrue(all(r == runs[0] for r in runs[1:]), async_)
            self.assertEqual(runs[0][0], {"p": B - DRAFT_SLOTS})

    def test_co_tenants_cap_the_prefill_and_decoders_keep_every_step(self) -> None:
        for async_ in (True, False):
            eng = Engine(arm(make_scheduler(async_), cap=2048))
            ids = decoders(eng)
            eng.add("p", 20_000)
            steps = eng.until_prefilled("p")
            self.assertEqual([st["p"] for st in steps], [2048] * 9 + [20_000 - 9 * 2048], async_)
            for st in steps:
                self.assertEqual([st[r] for r in ids], [1 + K] * 3)

    def test_the_cap_lifts_when_the_prefill_is_alone_again(self) -> None:
        eng = Engine(arm(make_scheduler(), cap=2048))
        for rid in ("d0", "d1"):
            eng.add(rid, 100, max_tokens=12)
        eng.run(2)
        eng.add("p", 40_000)
        steps = eng.until_prefilled("p")
        self.assertEqual(steps[0]["p"], 2048)
        self.assertIn(B - DRAFT_SLOTS, [st["p"] for st in steps])  # decoders done: full chunks
        self.assertFalse(any(rid in st for st in steps[-2:] for rid in ("d0", "d1")))

    def test_a_new_turn_is_admitted_at_the_next_step(self) -> None:
        eng = Engine(arm(make_scheduler(), cap=2048))
        decoders(eng, 2)
        eng.add("p", 40_000)
        eng.step()
        eng.add("e", 500)
        st = eng.step()
        self.assertEqual((st["p"], st["e"]), (2048, 500))

    def test_two_prefills_share_the_step(self) -> None:
        eng = Engine(arm(make_scheduler(), cap=2048))
        eng.add("p1", 30_000)
        eng.add("p2", 20_000)
        st = eng.step()
        self.assertEqual(st, {"p1": 2048, "p2": 2048})

    def test_a_smaller_configured_threshold_wins(self) -> None:
        eng = Engine(arm(make_scheduler(lpt=1024), cap=2048))
        decoders(eng)
        eng.add("p", 5000)
        steps = eng.until_prefilled("p")
        self.assertEqual([st["p"] for st in steps], [1024] * 4 + [5000 - 4 * 1024])

    def test_the_decoder_replay_tail_scales_the_cap(self) -> None:
        eng = Engine(arm(with_replay(make_scheduler()), cap=2048))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            decoders(eng)
            eng.add("p", 20_000)
            steps = eng.until_prefilled("p")
        self.assertEqual(out.getvalue().count("decoder replay tail of 2606 rows, cap x 21/40"), 1)
        tail_cap = 2048 * 21 // 40  # layers 0..20 of 40
        # 2048-token chunks until the next one would reach the last 2606 rows, then 1075
        self.assertEqual([st["p"] for st in steps], [2048] * 8 + [tail_cap] * 3 + [20_000 - 8 * 2048 - 3 * tail_cap])

    def test_tail_weighting_needs_an_exact_replay_guard(self) -> None:
        self.assertEqual(spc.replay_geometry(with_replay(make_scheduler())), (2606, 21, 40))
        self.assertEqual(spc.replay_geometry(with_replay(make_scheduler(), rows=128)), (0, 1, 1))  # per-chunk mode
        self.assertEqual(spc.replay_geometry(make_scheduler()), (0, 1, 1))  # replay off: no guard
        for geo, chunks in (((0, 1, 1), [2048, 1952]), ((2606, 21, 40), [1075, 1075, 1075, 775])):
            eng = Engine(arm(make_scheduler(), cap=2048))
            eng.s._dsv41_replay_geometry = geo
            decoders(eng)
            eng.add("p", 4000)
            self.assertEqual([st["p"] for st in eng.until_prefilled("p")], chunks, geo)
        s = make_scheduler()
        s.kv_cache_manager._dsv41_replay = types.SimpleNamespace(rows=2606)  # no model config: weighting off
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(spc._geometry(s), (0, 1, 1))
        self.assertIn("no decoder replay tail weighting", out.getvalue())

    def test_a_waiting_short_turn_counts_as_a_tail(self) -> None:
        eng = Engine(arm(with_replay(make_scheduler()), cap=2048))
        with contextlib.redirect_stdout(io.StringIO()):
            decoders(eng, 2)
            eng.add("p", 40_000)
            eng.step()
            eng.add("e", 500)  # its 500 rows all run every layer
            st = eng.step()
        self.assertEqual((st["p"], st["e"]), (1075, 500))

    def test_the_threshold_is_restored_when_schedule_raises(self) -> None:
        s = arm(make_scheduler(), cap=2048)
        eng = Engine(s)
        decoders(eng)
        eng.add("p", 20_000)
        s.kv_cache_manager.fail_next = True
        with self.assertRaises(RuntimeError):
            s.schedule()
        self.assertEqual(s.scheduler_config.long_prefill_token_threshold, 0)


class DecodeStepTests(unittest.TestCase):
    def test_one_decode_only_step_after_each_mixed_step(self) -> None:
        for async_ in (True, False):
            eng = Engine(arm(make_scheduler(async_), cap=2048, steps=1))
            ids = decoders(eng)
            eng.add("p", 10_000)
            steps = eng.until_prefilled("p")
            self.assertEqual([st.get("p", 0) for st in steps], [2048, 0] * 4 + [10_000 - 4 * 2048], async_)
            for st in steps:
                self.assertEqual([st[r] for r in ids], [1 + K] * 3)
            after = eng.run(3)  # prefill done: plain decode steps for all four
            self.assertTrue(all(len(st) == 4 and set(st.values()) == {1 + K} for st in after), after)

    def test_two_decode_only_steps_on_uncapped_chunks(self) -> None:
        eng = Engine(arm(make_scheduler(), steps=2))
        decoders(eng)
        eng.add("p", 20_000)
        steps = eng.until_prefilled("p")
        rest = B - 3 * (1 + K + DRAFT_SLOTS) - DRAFT_SLOTS
        self.assertEqual([st.get("p", 0) for st in steps], [rest, 0, 0, rest, 0, 0, 20_000 - 2 * rest])

    def test_no_decode_only_steps_without_a_decoding_request(self) -> None:
        eng = Engine(arm(make_scheduler(), cap=2048, steps=2))
        eng.add("p1", 10_000)
        eng.add("p2", 6000)
        steps = eng.until_prefilled("p2")
        self.assertTrue(all(st.get("p1", 0) + st.get("p2", 0) > 0 for st in steps), steps)

    def test_a_new_turn_waits_out_the_decode_only_step(self) -> None:
        eng = Engine(arm(make_scheduler(), cap=2048, steps=1))
        decoders(eng, 2)
        eng.add("p", 40_000)
        eng.step()  # mixed
        eng.add("e", 500)
        dec, mixed = eng.step(), eng.step()
        self.assertNotIn("p", dec)
        self.assertNotIn("e", dec)
        self.assertEqual((mixed["p"], mixed["e"]), (2048, 500))


class KnobAndWiringTests(unittest.TestCase):
    def test_knobs(self) -> None:
        self.assertEqual(spc.knobs({}), (0, 0))
        self.assertEqual(spc.knobs({spc.ENV_CAP: "", spc.ENV_STEPS: ""}), (0, 0))
        self.assertEqual(spc.knobs({spc.ENV_CAP: "2048", spc.ENV_STEPS: "1"}), (2048, 1))
        for bad in ({spc.ENV_CAP: "100"}, {spc.ENV_CAP: "-1"}, {spc.ENV_STEPS: "9"}, {spc.ENV_STEPS: "-1"}):
            with self.assertRaises(ValueError):
                spc.knobs(bad)
        self.assertEqual(spc.install({}), "off")  # returns before importing vLLM

    def test_threshold_rule(self) -> None:
        self.assertEqual(spc.threshold(0, 0, 4), 0)
        self.assertEqual(spc.threshold(0, 2048, 1), 0)
        self.assertEqual(spc.threshold(0, 2048, 2), 2048)
        self.assertEqual(spc.threshold(1024, 2048, 3), 1024)
        self.assertEqual(spc.threshold(4096, 2048, 3), 2048)
        self.assertEqual(spc.threshold(4096, 2048, 1), 4096)

    def test_carried_prefill(self) -> None:
        self.assertFalse(spc.carried_prefill({"a": 4, "b": 4}, 3))
        self.assertTrue(spc.carried_prefill({"a": 4, "b": 5}, 3))
        self.assertFalse(spc.carried_prefill({}, 3))

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse((PATCH / "sched_prefill_cap.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "functools", "os"})

    def test_markers_and_wiring(self) -> None:
        src = (PATCH / "sched_prefill_cap.py").read_text()
        for marker in (spc.LOG_ENGAGED, spc.LOG_DISARMED):
            self.assertEqual(src.count(marker), 2, marker)  # the constant + the one print (audit convention)
        site = (PATCH / "sitecustomize.py").read_text()
        self.assertIn('_patch("sched_prefill_cap", _p_sched_prefill_cap)', site)
        self.assertLess(site.index('_patch("decoder_replay"'), site.index('_patch("sched_prefill_cap"'))
        audit = (ROOT / "tools" / "engagement_audit.py").read_text()
        self.assertIn('"sched_prefill_cap.py": {"DSV41_SCHED_PREFILL_CAP": "2048"}', audit)
        run_sh = (ROOT / "run.sh").read_text()
        for name, default in ((spc.ENV_CAP, 2048), (spc.ENV_STEPS, 1)):
            self.assertIn(f"  {name}=\n", run_sh)  # FORWARD_ENVS: empty takes the recipe.yaml default
            self.assertIn(f'{name}="${{{name}:-{default}}}"', run_sh)


def _vllm_scheduler():
    try:
        from vllm.v1.core.sched import async_scheduler, scheduler  # noqa: F401
    except Exception:  # noqa: BLE001 - host without the image's vLLM / torch
        return None
    return scheduler, async_scheduler


@unittest.skipUnless(_vllm_scheduler(), "image-only: needs the image's vLLM (run CPU-only in the image)")
class ImageTests(unittest.TestCase):
    def test_pin_matches_the_image(self) -> None:
        import inspect
        import textwrap

        sched_mod, async_mod = _vllm_scheduler()
        seg = segments()
        cls = sched_mod.Scheduler
        for name, src in seg.items():
            if name.startswith("Scheduler."):
                fn = getattr(cls, name.split(".", 1)[1])
                fn = getattr(fn, "__wrapped__", fn)
                self.assertEqual(textwrap.dedent(inspect.getsource(fn)), textwrap.dedent(src), name)
        self.assertEqual(inspect.getsource(async_mod.AsyncScheduler), seg["AsyncScheduler"])

    def test_install_arms_and_is_idempotent(self) -> None:
        sched_mod, _ = _vllm_scheduler()
        cls = sched_mod.Scheduler
        orig = cls.schedule
        try:
            self.assertEqual(spc.install({spc.ENV_CAP: "2048", spc.ENV_STEPS: "1"}), "armed")
            self.assertEqual(cls.schedule._dsv41_prefill_cap, (2048, 1))
            self.assertEqual(spc.install({spc.ENV_CAP: "2048"}), "already installed")
        finally:
            cls.schedule = orig


if __name__ == "__main__":
    unittest.main()
