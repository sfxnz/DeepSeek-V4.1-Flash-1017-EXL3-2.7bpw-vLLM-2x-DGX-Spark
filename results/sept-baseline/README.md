# Sept baseline evidence (copied, read-only)

These files are copies from the September recipe (`/home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark`, commit 3de9146),
kept because the imported tests and `recipe.yaml` measured rows still point at them:

| File | Copied from | Used by |
|---|---|---|
| `boot-lm.sh` | `results/2026-09-22-endgame2/boot-lm.sh` | `tests/test_recipe_ops.py` PROMOTED_BOOT (defaults vs the last measured Sept boot) |
| `round36-headline.json` | `results/2026-09-27-viterbi-adopt/round36-headline.json` | `recipe.yaml` measured rows (Sept 2.0 bpw pack numbers) |

They describe the **Sept 2.0 bpw pack** (`sfxnz/DeepSeek-V4.1-Flash-EXL3` @ `2.0bpw-mcg-viterbi-lmhead-mxfp8`),
not the 1017 pack. TODO phase D: replace with the 1017 promoted boot and the 1017 measured headline,
then delete this directory.
