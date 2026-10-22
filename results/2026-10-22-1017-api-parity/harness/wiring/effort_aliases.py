"""What deepseek-recipe 0.2.0 does with each reasoning_effort value and thinking switch (for the README API section)."""
import json
import re

import deepseek_recipe as d

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
opts = d.ConversionOptions(default_thinking_mode=False).with_model_alias(MODEL, d.ModelVersion.DeepseekV41Flash)
enc = d.DeepseekV41Encoding()
cases = [("no thinking field", {})]
cases += [(f"thinking enabled, effort {e}", {"thinking": {"type": "enabled"}, **({"reasoning_effort": e} if e else {})})
          for e in (None, "none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max")]
cases += [(f"no thinking field, effort {e}", {"reasoning_effort": e}) for e in ("none", "low", "high", "max")]
cases += [("thinking disabled, effort high", {"thinking": {"type": "disabled"}, "reasoning_effort": "high"})]
for label, extra in cases:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}
    try:
        conv = d.ChatCompletionRequest(json.dumps(body)).convert(opts)
        r = enc.render_conversation(conv.conversation); prompt = r if isinstance(r, str) else getattr(r, "prompt", None) or getattr(r, "text", None) or str(r)
        m = re.search(r"Reasoning Effort: (\w+)", prompt)
        io = conv.inference_options
        print(f"{label:40s} thinking={conv.conversation.thinking_mode!s:5} effort={m.group(1) if m else '-':4} "
              f"T={io.temperature} top_p={io.top_p} max_tokens(prompt 10)={conv.resolve_max_tokens(10, 500000)}")
    except Exception as e:
        print(f"{label:40s} ERROR {type(e).__name__}: {str(e)[:150]}")
