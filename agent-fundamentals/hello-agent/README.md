# hello-agent — Tool Use / Function Calling

**Concept demonstrated:** the structured tool-calling loop that almost every agent framework is built on — the model returns a `tool_calls` object naming a function and its arguments, the caller executes it locally, and the result is fed back as a `role: "tool"` message until the model stops requesting tools.

**What this is *not*:** ReAct. That distinction, and the literature it maps to (Toolformer, ReAct, Reflexion, MemGPT), is covered in [`vs-literature.md`](./vs-literature.md) — read that alongside this file.

## Files

| File | Purpose |
|---|---|
| `agent.py` | `FunctionCallingAgent` — the loop itself |
| `tools.py` | Tool functions, their JSON schemas, and argument validation |
| `tests/test_tools.py` | Unit tests for the safe math evaluator and schema validation |
| `vs-literature.md` | What this implementation gets right/wrong against the papers it resembles |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY
python agent.py
```

## What changed from the original version

The first version of this folder worked but had four issues, all fixed here (see `vs-literature.md`'s changelog section for the reasoning behind each):

1. `calculate_expression` used `eval()` — now a restricted AST walk (same approach `agent-in-streamlit/tools/math_tools.py` already used).
2. Tool arguments were trusted straight out of `json.loads()` — now validated against the declared schema (`validate_arguments`) before a tool is ever called.
3. Malformed JSON from the model would raise and crash the loop — now caught and returned as a tool-visible error instead.
4. Hitting the iteration cap ended the loop silently — now returns an explicit `"max_iterations_reached"` status with a real final answer.

## Next in this concept series

This folder deliberately stays narrow — function calling only, no reasoning trace. The next fundamentals folder (`react-agent/`) will implement literal ReAct — the model required to emit a `Thought:` string every turn via prompting, not the native tool-call API — so the two mechanisms can be run on the same tasks and compared directly rather than conflated.