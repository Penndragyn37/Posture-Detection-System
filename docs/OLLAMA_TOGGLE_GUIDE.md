# Ollama Inference Toggle Guide

This guide explains every inference-behaviour setting under `gemma4:` in
`config.yaml`, why it exists, and how to test it independently on macOS.
All of these were added after debugging real failures during development;
each section names the symptom it fixes.

---

## Quick reference

| Setting | Default | What it does | Symptom if disabled |
|---|---|---|---|
| `force_json` | `true` | Forces Ollama to emit valid JSON | Model adds prose, JSON parsing fails |
| `think` | `false` | Asks Ollama to skip chain-of-thought | Often ignored — see `use_prefill` |
| `use_prefill` | `true` | Prefills assistant turn with `{` | Empty `content`, posture defaults to absent |
| `temperature` | `0.1` | Sampling randomness | Higher = less consistent classification |
| `num_predict` | `200` | Max output tokens | Too low risks truncated JSON |

---

## 1. `use_prefill` — the thinking-model fix

### What it does
Gemma 4 (both the 26B MoE and the E4B variant) has a built-in chain-of-thought
"thinking" capability. When it activates, the model writes its reasoning to a
separate `thinking` field in the Ollama response and leaves the `content`
field empty until it has finished "thinking out loud." For a JSON-only
classification task this means `content` can come back completely empty,
which crashes the JSON parser and the system falls back to `absent` with
0% confidence.

`think: false` is supposed to disable this, but in practice it is not
reliably honoured by every Ollama build — some versions ignore it for
vision requests.

The reliable fix is **assistant turn prefilling**. Instead of sending only
a user message with the image, a second message is appended:

```python
{"role": "assistant", "content": "{"}
```

This tells Ollama "the assistant has already started its reply with an
opening brace." The model has no choice but to continue directly from
that brace into JSON tokens, because from its perspective the thinking
phase has already been skipped, the reply is already in progress.

### How to test it on your Mac

**Reproduce the bug** (prefill off, see the empty response):
```bash
# Edit config.yaml: set use_prefill: false
python run.py --approach gemma4_server
```
Watch the terminal. You should see:
```
[PDSDA] Empty response from Ollama — check terminal for details
```

**Confirm the fix** (prefill on, default):
```bash
# Edit config.yaml: set use_prefill: true
python run.py --approach gemma4_server
```
Detections should now classify normally.

**Direct curl test**, bypassing the app entirely, to see the raw difference:
```bash
# Without prefill — may return thinking-only output
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4:26b",
  "messages": [{"role":"user","content":"Reply with only this JSON: {\"test\":\"ok\"}"}],
  "stream": false,
  "format": "json"
}' | python3 -m json.tool

# With prefill — forces immediate JSON continuation
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4:26b",
  "messages": [
    {"role":"user","content":"Reply with only this JSON: {\"test\":\"ok\"}"},
    {"role":"assistant","content":"{"}
  ],
  "stream": false,
  "format": "json"
}' | python3 -m json.tool
```
Compare the `message.content` field in both responses. With prefill,
`content` should contain the rest of the JSON immediately. Without it,
you may see `thinking` populated and `content` empty.

### For the final report
This is a strong candidate for an ablation table in the Implementation
section: run a small batch (20-30 frames) with `use_prefill: true` vs
`false` and report the empty-response rate for each. This demonstrates
the fix quantitatively rather than just describing it.

---

## 2. `force_json`

### What it does
Adds `"format": "json"` to the Ollama request body. This is a
constrained-decoding instruction: Ollama restricts the model's token
sampling to only sequences that form valid JSON syntax.

### How to test it on your Mac
```bash
# Edit config.yaml: set force_json: false
python run.py --approach gemma4_server
```
Watch the terminal for `JSON parse failed` errors. Without this flag, the
model is free to add conversational text like "The person appears to be
sitting at their desk. Here's the classification:" before the JSON object,
which the parser cannot handle even though the underlying classification
might be correct.

**Recommendation**: keep this `true` always. Disabling it is useful only
for one-off debugging of prompt wording, never for actual data collection.

---

## 3. `think`

### What it does
Adds `"think": false` directly to the request body (separate from
`use_prefill`, which is a structural workaround). This is the "ask nicely"
approach: some Ollama versions respect this flag and skip thinking
entirely, others ignore it for vision inputs.

### How to test it on your Mac
```bash
ollama --version
```
Check the Ollama changelog for your installed version to see if
`think: false` is documented as supported for vision requests. If your
version is older or this flag is unsupported, `use_prefill` is doing all
the real work and `think: false` has no effect either way (harmless to
leave on).

---

## 4. `temperature`

### What it does
Standard LLM sampling temperature. `0.1` means the model almost always
picks its highest-probability next token, producing consistent,
repeatable classifications across runs of the same image.

### How to test it on your Mac
```bash
# Edit config.yaml: set temperature: 0.8
python run.py --images ./test_images/sitting_001.jpg
```
Run the same single image multiple times in a row (use the Prev/Next
buttons to go back to it) and watch the `reasoning` field and confidence
score vary between runs. At `0.1` they should be nearly identical each time.

**Recommendation**: keep at `0.1` for evaluation runs, since the benchmark
needs reproducible results. Raising it is only useful for qualitatively
exploring how the model's stated reasoning changes.

---

## 5. `num_predict`

### What it does
Caps the number of tokens Ollama will generate per response. The JSON
output (`posture`, `confidence`, `reasoning`) is roughly 30-60 tokens, so
`200` leaves comfortable headroom.

### How to test it on your Mac
```bash
# Edit config.yaml: set num_predict: 20
python run.py --approach gemma4_server
```
With a low cap, you should start seeing `JSON parse failed` errors because
the response gets cut off mid-object (e.g. `{"posture":"sitting","confiden`).
This confirms the headroom matters.

---

## General Ollama diagnostics (macOS)

Run this before any benchmarking session to confirm your environment is
healthy:
```bash
python test_ollama.py
```

Check what capabilities a model reports (look for `vision` and `thinking`):
```bash
ollama show gemma4:26b
```

Recommended environment variables for the M5 Pro (add to `~/.zshrc`):
```bash
export OLLAMA_FLASH_ATTENTION=0      # avoids a known Gemma 4 hang in Ollama 0.19
export OLLAMA_NUM_PARALLEL=1         # PDSDA sends one frame at a time
export OLLAMA_MAX_LOADED_MODELS=1    # keep Gemma 4 loaded between frames
```

If you switch between `gemma4:26b` and `gemma4:e4b` frequently during
benchmarking, consider raising `OLLAMA_MAX_LOADED_MODELS` to `2` so both
stay resident and you avoid a multi-second reload penalty each time you
change the model dropdown in the UI.
