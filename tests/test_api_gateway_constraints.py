#!/usr/bin/env python3
"""gateway/constraints.py: deepseek-recipe's OutputConstraint -> vLLM `structured_outputs`.

Host (stdlib): structured_outputs_for on stubs of the converted requests in
tests/fixtures/api_gateway/constraints_cases.json (request bodies converted by deepseek-recipe 0.2.0:
constraint kind and schema, the schema as written, the last message, thinking mode), and the shape of
the grammars it emits.
Wheel (skipped without deepseek_recipe): the fixture is what deepseek-recipe produces, and the real
converted requests map exactly like their stubs.
Image (opt-in, DSV41_IMAGE_TESTS=1, docker, no GPU): tests/api_gateway_constraints_image_check.py
compiles every emitted grammar through vLLM's /inference/v1/generate validation and xgrammar backend,
feeds the fixture's upstream-rendered and malformed outputs, and replays the engine's reasoning gate
for a token-in request.

Regenerate the fixture with a python that has the wheel:
  python3 tests/test_api_gateway_constraints.py --write-fixture
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))
sys.path.insert(0, str(ROOT / "tests"))

import api_gateway_constraints_image_check as chk  # noqa: E402
import constraints as c  # noqa: E402

try:
    import deepseek_recipe as dr
except ImportError:
    dr = None

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
PACK = Path.home() / ".cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq"
IMAGE = os.environ.get("DSV41_IMAGE", "dsv41-flash-exl3-sm121:e16-rows")
PREFILL = "\n\n<｜DSML｜ calls>\n"


def _fn(name, params, strict=None):
    f = {"name": name, "description": f"{name} tool.", "parameters": params}
    if strict is not None:
        f["strict"] = strict
    return {"type": "function", "function": f}


WEATHER = {"type": "object", "properties": {
    "city": {"type": "string", "description": "City name"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    "days": {"type": "integer", "minimum": 1, "maximum": 7}}, "required": ["city"]}
SEARCH = {"type": "object", "properties": {
    "query": {"type": "string"},
    "filters": {"type": "object", "properties": {"site": {"type": "string"}, "year": {"type": "integer"}}},
    "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["query"]}
TIME = {"type": "object", "properties": {}}
BOOK = {"type": "object", "properties": {"guest": {"$ref": "#/$defs/Guest"}, "note": {"type": ["string", "null"]}},
        "$defs": {"Guest": {"type": "object", "properties": {"name": {"type": "string"}}}}}
CODE = {"type": "object", "properties": {"code": {"type": "string", "pattern": "[A-Z]{3}-[0-9]{2}"}}}
PERSON = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                           "email": {"type": "string", "format": "email"}}, "required": ["name", "age"]}
ORDER = {"type": "object", "properties": {"guest": {"$ref": "#/$defs/Guest"}, "note": {"type": ["string", "null"]}},
         "$defs": {"Guest": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
                             "additionalProperties": False}}}
STRICT_TOOLS = [_fn("get_weather", {k: v for k, v in WEATHER.items() if k != "required"}, True),
                _fn("web_search", {k: v for k, v in SEARCH.items() if k != "required"}, True),
                _fn("get_time", TIME, True), _fn("book", BOOK, True), _fn("lookup", CODE, True)]


def _user(text):
    return [{"role": "user", "content": text}]


def _rf(schema, strict=None):
    js = {"name": "answer", "schema": schema}
    if strict is not None:
        js["strict"] = strict
    return {"type": "json_schema", "json_schema": js}


def build_fixture(dr) -> dict:
    """Request bodies converted by deepseek-recipe, with upstream-rendered tool-call outputs."""
    opts = dr.ConversionOptions(default_thinking_mode=False).with_model_alias(MODEL, dr.ModelVersion.DeepseekV41Flash)
    enc = dr.DeepseekV41Encoding()

    def dsml(calls, content=""):
        """The assistant turn deepseek-recipe renders for these calls, after `<｜Assistant｜></think>`."""
        tcs = [dr.ToolCall(f"call_{i}", n, json.dumps(a)) for i, (n, a) in enumerate(calls)]
        conv = dr.Conversation([dr.UserMessage("q"), dr.AssistantMessage(content, tool_calls=tcs)], thinking_mode=False)
        p = enc.render_conversation(conv).prompt
        start = p.index("<｜Assistant｜></think>") + len("<｜Assistant｜></think>")
        return p[start:p.index("<｜end▁of▁sentence｜>", start)]

    def inner(block):
        """What the model emits after the required/named prefill."""
        assert block.startswith(PREFILL), block
        return block[len(PREFILL):]

    w_full = {"city": "Paris", "unit": "celsius", "days": 3}
    s_full = {"query": "paris weather", "filters": {"site": "meteo.fr", "year": 2026}, "tags": ["forecast", "france"]}
    wf = dsml([("get_weather", w_full)])
    wp = dsml([("get_weather", {"city": "Paris", "days": 3})])
    two = dsml([("get_weather", w_full), ("web_search", s_full)])
    multi = dsml([("web_search", {"query": "line one\nline two <b>x</b>", "filters": {"site": "a.b", "year": 1}, "tags": []})])
    noarg = dsml([("get_time", {})])
    book = dsml([("book", {"guest": {"name": "Ada"}, "note": "window seat"})])
    book_null = dsml([("book", {"guest": {"name": "Ada"}, "note": None})])
    code = dsml([("lookup", {"code": "ABC-12"})])
    w_twice = dsml([("get_weather", w_full), ("get_weather", {"city": "Lyon", "unit": "celsius", "days": 1})])
    w_said = dsml([("get_weather", w_full)], content="Let me check.")
    wrong_order = dsml([("get_weather", {"days": 3, "city": "Paris", "unit": "celsius"})])
    bad = {
        "unknown_tool": wf.replace("get_weather", "get_wether"),
        "unknown_param": wf.replace('name="city"', 'name="town"'),
        "int_as_string": wf.replace('name="days" string="false">3<', 'name="days" string="true">three<'),
        "int_out_of_range": wf.replace('string="false">3<', 'string="false">10<'),
        "enum_violation": wf.replace(">celsius<", ">kelvin<"),
        "unclosed_invoke": wf.replace("</｜DSML｜ invoke>\n", ""),
        "duplicate_param": wf.replace('<｜DSML｜ parameter name="unit"', '<｜DSML｜ parameter name="city" string="true">'
                                      'Lyon</｜DSML｜ parameter>\n<｜DSML｜ parameter name="unit"'),
        "bad_json_value": two.replace('{"site": "meteo.fr", "year": 2026}', "{site: meteo.fr}"),
        "nested_missing_key": two.replace('{"site": "meteo.fr", "year": 2026}', '{"site": "meteo.fr"}'),
        "nested_extra_key": two.replace('"year": 2026}', '"year": 2026, "lang": "fr"}'),
        "text_after_calls": wf + "\nDone.",
        "noarg_compact": noarg.replace('">\n\n</｜DSML｜ invoke>', '">\n</｜DSML｜ invoke>'),
        "code_pattern": code.replace("ABC-12", "abc-12"),
    }
    prose = "It is sunny in Paris."
    answer = '{"capital": "Paris"}'
    person = '{"name": "Ada Lovelace", "age": 36, "email": "ada@example.org"}'
    order = '{"guest": {"name": "Ada"}, "note": null}'
    weather_tools = [_fn("get_weather", WEATHER), _fn("web_search", SEARCH)]
    specs = [
        # name, protocol, body, expect, accept, reject
        ("chat_text", "chat", {"messages": _user("Hi")}, None, [], []),
        ("chat_json_object", "chat", {"messages": _user("Answer in JSON: the capital of France?"),
                                      "response_format": {"type": "json_object"}}, "json",
         # the API's json_object constraint is any JSON value, an array included
         [answer, '{\n  "capital": "Paris"\n}', f"[{answer}]"], [f"Here: {answer}", f"```json\n{answer}\n```"]),
        ("chat_json_schema", "chat", {"messages": _user("Extract: Ada Lovelace, 36."), "response_format": _rf(PERSON)},
         "json", [person, '{"name": "Ada Lovelace", "age": 36}'],
         ['{"name": "Ada Lovelace", "age": "36"}', person[:-1] + ', "born": 1815}', f"Here: {person}"]),
        ("chat_json_schema_strict_defs", "chat", {"messages": _user("Book for Ada."), "response_format": _rf(ORDER, True)},
         "json", [order, '{"guest": {"name": "Ada"}, "note": "window seat"}'],
         ['{"guest": {"name": "Ada"}}', '{"guest": {"nam": "Ada"}, "note": null}', '{"guest": "Ada", "note": null}']),
        ("responses_text_format_strict_defs", "responses",
         {"input": "Book for Ada.", "text": {"format": {"type": "json_schema", "name": "order", "strict": True,
                                                        "schema": ORDER}}},
         "json", [order], ['{"guest": {"name": "Ada"}}']),
        ("chat_regex", "chat", {"messages": _user("A date?"), "response_format": {"type": "regex", "regex": r"\d{4}-\d{2}-\d{2}"}},
         "regex", ["2026-10-22"], ["2026-1-22", "Date: 2026-10-22"]),
        ("chat_thinking_json_object", "chat", {"messages": _user("Answer in JSON: the capital of France?"),
                                               "thinking": {"type": "enabled"}, "response_format": {"type": "json_object"}},
         "json", [answer], [prose]),
        ("chat_required", "chat", {"messages": _user("Weather in Paris?"), "tool_choice": "required",
                                   "tools": weather_tools + [_fn("get_time", TIME)]}, "structural_tag",
         [inner(b) for b in (wf, wp, two, multi, noarg, wrong_order)],
         [inner(bad[k]) for k in ("unknown_tool", "unclosed_invoke", "bad_json_value", "text_after_calls", "noarg_compact")]
         + [prose, "</｜DSML｜ calls>", wf]),
        ("chat_named", "chat", {"messages": _user("Weather in Paris?"), "tools": weather_tools,
                                "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
         "structural_tag", [inner(wf), inner(w_twice)], [inner(dsml([("web_search", s_full)])), inner(two)]),
        ("chat_required_strict", "chat", {"messages": _user("Weather in Paris?"), "tools": STRICT_TOOLS,
                                          "tool_choice": "required"}, "structural_tag",
         # book: upstream drops $defs from strict tools, so `guest` takes any JSON value ({"nam": ...} too)
         [inner(b) for b in (wf, two, noarg, book, book_null, code, book.replace('{"name": "Ada"}', '{"nam": "Ada"}'))],
         [inner(bad[k]) for k in ("unknown_param", "int_as_string", "int_out_of_range", "enum_violation", "duplicate_param",
                                  "nested_missing_key", "nested_extra_key", "noarg_compact", "code_pattern")]
         + [inner(wp), inner(wrong_order)]),
        ("chat_auto_strict", "chat", {"messages": _user("Weather in Paris?"), "tools": STRICT_TOOLS[:2]}, "structural_tag",
         [prose, w_said, wf, two], [w_said.replace('name="city"', 'name="town"'), bad["enum_violation"]]),
        ("chat_auto", "chat", {"messages": _user("Weather in Paris?"), "tools": weather_tools}, None, [], []),
        ("chat_rf_plus_tools", "chat", {"messages": _user("Weather in Paris? Reply in json."), "tools": weather_tools,
                                        "response_format": {"type": "json_object"}}, None, [], []),
        ("chat_tools_none_rf", "chat", {"messages": _user("Extract: Ada Lovelace, 36."), "tools": weather_tools,
                                        "tool_choice": "none", "response_format": _rf(PERSON)}, "json", [person], [prose]),
        ("chat_prefix_json_schema", "chat", {"messages": _user("Extract: Ada.") + [
            {"role": "assistant", "content": '{"name": ', "prefix": True}], "response_format": _rf(PERSON)}, None, [], []),
        ("chat_prefix_reasoning_regex", "chat", {"messages": _user("A date?") + [
            {"role": "assistant", "content": "", "reasoning_content": "Today is", "prefix": True}],
            "thinking": {"type": "enabled"}, "response_format": {"type": "regex", "regex": r"\d{4}-\d{2}-\d{2}"}},
         "regex", ["2026-10-22"], ["Today: 2026-10-22"]),
        ("responses_required", "responses", {"input": "Weather in Paris?", "tool_choice": "required", "tools": [
            {"type": "function", "name": "get_weather", "description": "Weather.", "parameters": WEATHER}]},
         "structural_tag", [inner(wf)], [inner(bad["unknown_tool"])]),
        ("messages_any", "messages", {"max_tokens": 64, "messages": _user("Weather in Paris?"), "tool_choice": {"type": "any"},
                                      "tools": [{"name": "get_weather", "description": "Weather.", "input_schema": WEATHER}]},
         None, [], []),
    ]
    protocols = {"chat": dr.ChatCompletionRequest, "responses": dr.ResponsesRequest, "messages": dr.MessagesRequest}
    cases = []
    for name, protocol, body, expect, accept, reject in specs:
        req = protocols[protocol](json.dumps(dict(body, model=MODEL))).convert(opts)
        oc = req.inference_options.output_constraint
        last = req.conversation.messages[-1]
        cases.append({"name": name, "protocol": protocol, "body": body, "thinking": req.conversation.thinking_mode,
                      "constraint": None if oc is None else {"kind": oc.kind, "schema": oc.schema},
                      "response_schema": req.conversation.response_schema,
                      "last_message": {"role": last.role, "prefix": last.prefix, "content": last.content},
                      "expect": expect, "accept": accept, "reject": reject})
    return {"deepseek_recipe": dr.__version__, "model": MODEL, "cases": cases}


def _case(name):
    return next(x for x in chk.load_cases() if x["name"] == name)


def _so(name):
    case = _case(name)
    return c.structured_outputs_for(chk.stub(case), thinking=case["thinking"])


def _tag_format(so):
    tag = json.loads(so["structural_tag"])
    assert tag["type"] == "structural_tag"
    return tag["format"]


def _invokes(calls):
    return {t["begin"]: t["content"] for t in calls["tags"]}


class MappingTests(unittest.TestCase):
    def test_every_fixture_request_maps_to_its_kind(self) -> None:
        for case in chk.load_cases():
            so = c.structured_outputs_for(chk.stub(case), thinking=case["thinking"])
            self.assertEqual(chk.result_kind(so), case["expect"], case["name"])
            json.dumps(so)  # the gateway posts it as JSON

    def test_json_object_regex_and_schema_values(self) -> None:
        self.assertEqual(_so("chat_json_object"), {"json": "{}"})
        self.assertEqual(_so("chat_regex"), {"regex": r"\d{4}-\d{2}-\d{2}"})
        self.assertEqual(json.loads(_so("chat_json_schema")["json"]), PERSON)

    def test_strict_json_schema_is_normalized_and_keeps_its_defs(self) -> None:
        for name in ("chat_json_schema_strict_defs", "responses_text_format_strict_defs"):
            schema = json.loads(_so(name)["json"])
            self.assertEqual(schema["required"], ["guest", "note"], name)
            self.assertIs(schema["additionalProperties"], False)
            self.assertEqual(schema["$defs"], ORDER["$defs"])
            self.assertEqual(schema["properties"]["guest"], {"$ref": "#/$defs/Guest"})

    def test_a_ref_that_names_nothing_accepts_any_value(self) -> None:
        case = dict(_case("chat_json_schema_strict_defs"), response_schema=None)
        schema = json.loads(c.structured_outputs_for(chk.stub(case), thinking=False)["json"])
        self.assertEqual(schema["properties"]["guest"], {})
        self.assertNotIn("$defs", schema)

    def test_required_and_named_start_inside_the_calls_block(self) -> None:
        for name, tools in (("chat_required", ["get_weather", "web_search", "get_time"]), ("chat_named", ["get_weather"]),
                            ("responses_required", ["get_weather"])):
            fmt = _tag_format(_so(name))
            self.assertEqual(fmt["type"], "sequence", name)
            calls, end = fmt["elements"]
            self.assertEqual(end, {"type": "const_string", "value": "</｜DSML｜ calls>"})
            self.assertEqual((calls["type"], calls["separator"], calls["at_least_one"], calls["stop_after_first"]),
                             ("tags_with_separator", "", True, False))
            self.assertEqual(list(_invokes(calls)), [f'<｜DSML｜ invoke name="{t}">\n' for t in tools])
            self.assertNotIn("<｜DSML｜ calls>", json.dumps(fmt, ensure_ascii=False))  # the prompt holds the opener

    def test_auto_constrains_only_strict_tools_and_is_triggered(self) -> None:
        self.assertIsNone(_so("chat_auto"))
        self.assertIsNone(_so("messages_any"))
        fmt = _tag_format(_so("chat_auto_strict"))
        self.assertEqual((fmt["type"], fmt["triggers"], fmt["excludes"]),
                         ("triggered_tags", ["<｜DSML｜ calls>"], ["<think>", "</think>"]))
        (block,) = fmt["tags"]
        self.assertEqual((block["begin"], block["end"]), ("<｜DSML｜ calls>\n", "</｜DSML｜ calls>"))

    def test_the_tool_constraint_wins_over_response_format(self) -> None:
        self.assertEqual(_case("chat_rf_plus_tools")["constraint"]["kind"], "tool_call")
        self.assertIsNone(_so("chat_rf_plus_tools"))
        self.assertEqual(json.loads(_so("chat_tools_none_rf")["json"]), PERSON)

    def test_strict_parameters_are_declared_in_order_and_typed(self) -> None:
        inv = _invokes(_tag_format(_so("chat_required_strict"))["elements"][0])
        city, unit, days = inv['<｜DSML｜ invoke name="get_weather">\n']["elements"]
        head = '<｜DSML｜ parameter name="'
        self.assertEqual(city["begin"], head + 'city" string="true">')
        self.assertEqual(city["content"]["type"], "any_text")
        self.assertEqual(unit["begin"], head + 'unit" string="true">')
        self.assertEqual(unit["content"], {"type": "or", "elements": [{"type": "const_string", "value": "celsius"},
                                                                      {"type": "const_string", "value": "fahrenheit"}]})
        self.assertEqual((days["begin"], days["content"]), (head + 'days" string="false">', {
            "type": "json_schema", "json_schema": {"type": "integer", "minimum": 1, "maximum": 7}}))
        self.assertEqual(days["end"], "</｜DSML｜ parameter>\n")
        query, filters, tags = inv['<｜DSML｜ invoke name="web_search">\n']["elements"]
        self.assertEqual(filters["content"]["json_schema"]["required"], ["site", "year"])  # upstream normalization
        guest, note = inv['<｜DSML｜ invoke name="book">\n']["elements"]
        self.assertEqual(guest["begin"], head + 'guest" string="')  # either form: the ref names nothing
        self.assertEqual(guest["content"]["elements"][1]["elements"][1]["json_schema"], {})
        self.assertEqual(note["begin"], head + 'note" string="')  # ["string", "null"]: either form
        (code,) = inv['<｜DSML｜ invoke name="lookup">\n']["elements"]
        self.assertEqual(code["content"], {"type": "regex", "pattern": "[A-Z]{3}-[0-9]{2}"})

    def test_zero_argument_invokes_take_the_trained_blank_line(self) -> None:
        strict = _invokes(_tag_format(_so("chat_required_strict"))["elements"][0])
        self.assertEqual(strict['<｜DSML｜ invoke name="get_time">\n'], {"type": "const_string", "value": "\n"})
        loose = _invokes(_tag_format(_so("chat_required"))["elements"][0])['<｜DSML｜ invoke name="get_time">\n']
        self.assertEqual(loose["type"], "or")
        self.assertEqual(loose["elements"][0], {"type": "const_string", "value": "\n"})
        self.assertEqual(loose["elements"][1]["type"], "plus")

    def test_thinking_mode_keeps_its_grammar(self) -> None:
        self.assertEqual(_so("chat_thinking_json_object"), {"json": "{}"})
        self.assertEqual(_so("chat_prefix_reasoning_regex"), {"regex": r"\d{4}-\d{2}-\d{2}"})

    def test_required_in_thinking_mode_falls_back(self) -> None:
        case = _case("chat_required")
        self.assertIs(c.structured_outputs_for(chk.stub(case), thinking=True), c.FALLBACK)
        self.assertEqual(repr(c.FALLBACK), "FALLBACK")

    def test_prefix_with_answer_text_gets_no_grammar(self) -> None:
        self.assertEqual(_case("chat_prefix_json_schema")["last_message"]["content"], '{"name": ')
        self.assertIsNone(_so("chat_prefix_json_schema"))

    def test_no_constraint_and_unknown_kind(self) -> None:
        self.assertIsNone(_so("chat_text"))
        conv = SimpleNamespace(inference_options=SimpleNamespace(output_constraint=SimpleNamespace(kind="grammar", schema="")))
        with self.assertRaises(ValueError):
            c.structured_outputs_for(conv, thinking=False)


@unittest.skipIf(dr is None, "deepseek_recipe (0.2.0 wheel) not importable")
class WheelTests(unittest.TestCase):
    def test_fixture_is_what_deepseek_recipe_produces(self) -> None:
        self.assertEqual(json.loads(chk.FIXTURE.read_text()), json.loads(json.dumps(build_fixture(dr))))

    def test_real_converted_requests_map_like_their_stubs(self) -> None:
        opts = dr.ConversionOptions(default_thinking_mode=False).with_model_alias(MODEL, dr.ModelVersion.DeepseekV41Flash)
        protocols = {"chat": dr.ChatCompletionRequest, "responses": dr.ResponsesRequest, "messages": dr.MessagesRequest}
        for case in chk.load_cases():
            req = protocols[case["protocol"]](json.dumps(dict(case["body"], model=MODEL))).convert(opts)
            self.assertEqual(c.structured_outputs_for(req, thinking=req.conversation.thinking_mode),
                             c.structured_outputs_for(chk.stub(case), thinking=case["thinking"]), case["name"])


@unittest.skipUnless(os.environ.get("DSV41_IMAGE_TESTS") == "1" and shutil.which("docker") and (PACK / "tokenizer.json").is_file(),
                     "set DSV41_IMAGE_TESTS=1 (needs docker, the serve image and the pack tokenizer)")
class ImageTests(unittest.TestCase):
    def test_grammars_compile_and_gate_inside_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                shutil.copy(PACK / name, Path(d) / name)
            os.chmod(d, 0o755)
            proc = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "--memory", "4g", "--name", f"dsv41-gw-constraints-{os.getpid()}",
                 "-v", f"{ROOT}:/repo:ro",
                 "-v", f"{ROOT}/docker/patch:/opt/dsv41-patch:ro",
                 "-v", f"{ROOT}/docker/patch/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro",
                 "-v", f"{d}:/tok:ro",
                 "--entrypoint", "python3", IMAGE, "/repo/tests/api_gateway_constraints_image_check.py", "/tok"],
                capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-2000:])
        self.assertIn("PASS gate: prompt ending in <think> is free until </think>, masked after it", proc.stdout)
        self.assertNotIn("FAIL", proc.stdout)


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-fixture"]:
        chk.FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        chk.FIXTURE.write_text(json.dumps(build_fixture(dr), indent=1, ensure_ascii=False) + "\n")
        print("wrote", chk.FIXTURE)
    else:
        unittest.main()
