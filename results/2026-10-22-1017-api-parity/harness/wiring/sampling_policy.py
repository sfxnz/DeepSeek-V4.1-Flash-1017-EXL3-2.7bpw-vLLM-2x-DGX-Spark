"""deepseek-recipe 0.2.0 sampling policy as the gateway applies it (README API section)."""
import json

import deepseek_recipe as d

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
opts = d.ConversionOptions(default_thinking_mode=False).with_model_alias(MODEL, d.ModelVersion.DeepseekV41Flash)
for label, extra in [
    ("chat, T=0 top_p=0.5", {"temperature": 0, "top_p": 0.5}),
    ("chat, T=0.6", {"temperature": 0.6}),
    ("thinking, T=0 top_p=0.5", {"thinking": {"type": "enabled"}, "temperature": 0, "top_p": 0.5}),
    ("thinking, T=0.6 top_p=0.98", {"thinking": {"type": "enabled"}, "temperature": 0.6, "top_p": 0.98}),
    ("chat, max_tokens 393217", {"max_tokens": 393217}),
    ("chat, 17 stop strings", {"stop": [str(i) for i in range(17)]}),
    ("chat, 16 stop strings", {"stop": [str(i) for i in range(16)]}),
]:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}
    try:
        conv = d.ChatCompletionRequest(json.dumps(body)).convert(opts)
        io = conv.inference_options
        print(f"{label:30s} thinking={conv.conversation.thinking_mode!s:5} T={io.temperature} top_p={io.top_p} "
              f"max_tokens={conv.resolve_max_tokens(10, 500000)}")
    except Exception as e:
        print(f"{label:30s} {type(e).__name__}: {getattr(e, 'status_code', '')} {str(getattr(e, 'body', e))[:160]}")
