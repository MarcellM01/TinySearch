## MCP Knowledge Pack (tinysearch)

This environment uses the **tinysearch** MCP server. It exposes **one** tool.

### Available tool

**`research(query)`**

- **Input:** a single string field **`query`** only. Pass the user’s question **as-is**: do not rewrite, spell-correct, add dates, expand abbreviations, translate, or “improve” the wording before calling.
- **Output:** `{"answer": "<prompt string>"}`. The `answer` is a **search-grounded prompt** (not a finished article): it aggregates ranked web results, crawled page text, and chunk context. **Your job** is to answer the user from that prompt and **cite source URLs** that appear in the blocks.

There is **no** `access_site`, `search_web`, `lite_*`, or `mode` / `max_results` on this server. Everything goes through **`research(query)`**.

---

### Tool routing (most important)

#### Compound questions about “what do we use + is there something newer/better?”
Always split into two sequential steps:

1. **Codebase first** — search/read local files to find what the project actually uses. Never skip this on the assumption you already know.
2. **MCP second** — call **`research(query)`** with a focused question when you need **live web** evidence (docs, releases, comparisons). Use the returned `answer` prompt as the evidence base.

This order is mandatory. Reversing it (or only doing the MCP half) mis-describes the project.

#### When to use codebase tools (search/read/list files)
- “What model / config / setting does this project use for X?”
- “Where is X configured / called / defined?”
- “Which version / provider / endpoint does the code target?”
- Anything answerable from files in the repo.

#### When to use `research(query)`
Use it when you need **up-to-date external facts** and primary sources:
- “Is there a newer version of X?”
- “What does the vendor doc say about Y?”
- “What are the alternatives to Z?”
- “What changed between versions?”

Prefer **URLs and short quotes** from the prompt text. External claims should be grounded in what `research` surfaced.

#### Source hygiene
- Prefer **official docs** over blogs when the prompt includes them.
- Prefer **changelogs / release notes** for “what’s new”.
- If sources in the prompt conflict, report the conflict and cite both.

---

### Strategy with a single tool

1. If you need **broader discovery**, call **`research(query)`** once with a clear question aligned to the user’s goal.
2. **Answer from `answer`**: synthesize the user’s reply from the embedded search snippets and crawled chunks; pull **URLs** from the prompt for citations.
3. If the prompt is thin on one angle, **refine `query`** and call **`research` again** with a narrower follow-up (do not assume other MCP tools exist).

#### If the user gave an exact URL
You **cannot** pass a URL-only “fetch this page” call. Frame **`research(query)`** so discovery still finds that page or the same domain (e.g. include the hostname or product name from the URL in natural language), or rely on the user-pasted URL in chat while you use `research` for **surrounding** context.

#### If results look partial or empty
- Retry **`research`** at most once with a **tightened or alternate `query`**.
- After two failures, say what you tried and ask for guidance.

---

## Structured approach for “what we use / something better?”

```
Step 1 – Search the codebase for the relevant config/constant/import.
Step 2 – Read the specific file(s) to confirm current value/behavior.
Step 3 – Call research(query) with a focused external question.
Step 4 – Answer from the answer prompt; cite URLs present in the prompt.
Step 5 – Synthesize: "Project uses X. Sources in the research prompt suggest Y. Upgrade path Z."
```

Do not emit a final recommendation until the applicable steps are done.

---

## Tool-loop prevention

- **Same tool, same args → stop.** If `research(query)` with identical arguments returned the same class of result twice, change the query or approach—do not spam identical calls.
- **Three-strike rule.** After three consecutive tool calls with no new actionable information, pause and reassess.
- **No circular read→tool→read chains** that don’t add facts.
- **Failed `research`.** At most one retry with a refined `query`; then report attempts and limitations.
- **Progress gate.** Before each call, ask what new information it will add; if unclear, don’t call.

---

## PowerShell command guidelines (Windows environment)

When executing commands on Windows, use **PowerShell** syntax:

### Command style
- Use concise commands; avoid noisy output when possible.
- Prefer pipelines over needless intermediate variables.
- Use `--` to separate options from positional arguments that may start with `-`.

### Common patterns
```powershell
command && echo "Success" || echo "Failed"
command 2>&1 | Out-Null
Get-Content -Path "file.txt"
Set-Content -Path "file.txt" -Value "content"
Add-Content -Path "file.txt" -Value "more content"
Get-ChildItem -Recurse
Select-String -Pattern "pattern" -Path "*.py" | Select-Object -First 10
Start-Process powershell -ArgumentList "command" -Verb RunAs
```

### Error handling
- Check exit codes for critical commands.
- Use `try/catch` where failures are expected.
- Use `2>&1` when capturing stderr with stdout.

### Linter / typecheck / tests
After edits, verify quality as the project expects, for example:
```powershell
flake8 path/to/file.py
mypy path/to/file.py --ignore-missing-imports
pytest tests/ -q
```

---

## General code behaviour

- When in doubt about a service or config value, search the codebase before guessing.
- Do not assume the project matches vendor defaults — verify from source.
- For external facts, rely on **`research(query)`** and cite URLs from the returned prompt.
