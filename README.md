# LLM Tool Calling Demo

A minimal Python project showing how to call an OpenAI model with a
**tool / function call** round-trip. The model receives a user question,
decides on its own whether it needs the `calculator` tool, the tool is
executed locally in Python, and the model produces the final natural-language
answer using the tool's result.

## How tool calling works

The flow implemented in `app.py`:

1. **User → Model (1st call).** The prompt plus the tool schema is sent to
   the Chat Completions API. The tool schema declares a `calculator` function
   that accepts `{ "expression": string }`.
2. **Model decides.** The model returns either a normal message *or* a
   `tool_calls` array describing which tools to run and with what arguments.
3. **Tool execution.** If `tool_calls` is present, the script parses the JSON
   arguments and runs the local `calculator` function (a safe AST-based
   evaluator — it does **not** use `eval`).
4. **Tool → Model (2nd call).** The assistant message (including
   `tool_calls`) and one `role: "tool"` message per call (with the JSON result
   and matching `tool_call_id`) are appended to the history and sent back.
5. **Final answer.** The model turns the tool output into a human-readable
   response, which is printed to stdout.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env    # Windows: copy .env.example .env
# then edit .env and put in your OPENAI_API_KEY
```

## Usage

Pass the prompt as a CLI argument:

```bash
python app.py "What is (128 * 7) + 42?"
```

Or run with no arguments to be prompted interactively:

```bash
python app.py
Ask something: How much is 15% of 240?
```

### Example session

```
$ python app.py "Compute (2**10 - 24) / 8 and tell me if it is even."
The result of (2**10 - 24) / 8 is 125.0, which is odd.
```

Behind the scenes the model emitted a `calculator` tool call with
`{"expression": "(2**10 - 24) / 8"}`, the script executed it, returned
`{"result": 125.0}`, and the model wrote the final sentence.

## Project layout

```
app.py              # Main script: OpenAI client + calculator tool + CLI
requirements.txt    # Python dependencies
.env.example        # Template for environment variables
.gitignore
README.md
```

## Safety note on the calculator

`calculator()` parses the expression with `ast.parse(..., mode="eval")` and
walks the tree, allowing only numeric literals and the operators
`+ - * / // % **` (plus unary `+`/`-`). Names, attribute access and function
calls are rejected, so arbitrary code cannot be executed even if the model
suggests a malicious expression.
