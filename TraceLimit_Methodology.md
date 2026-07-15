# TraceLimit — Research Methodology
## Positional Bias ("Lost in the Middle") in Automated Code Debugging

**Version:** 3.0 · **Type:** 2-month computational research project  
**Methods:** Programmatic Bug Injection, Automated Benchmarking  
**Stack:** Python 3.11+, Pytest, Docker, LLM APIs

---

## Table of Contents

1. [Hypothesis](#hypothesis)
2. [Models Dataset](#models-dataset)
3. [Code Repository Dataset](#code-repository-dataset)
4. [Bug Injection Dataset](#bug-injection-dataset)
5. [Control Cases](#control-cases)
6. [Step-by-Step Methodology](#step-by-step-methodology)
7. [Execution Guide](#execution-guide)
8. [Measurement & Analysis](#measurement--analysis)
9. [Fairness Controls](#fairness-controls)

---

## Hypothesis

> When a bug is physically located in the **middle 40–60%** of a compiled multi-file context window, an LLM's ability to identify and fix it drops significantly compared to bugs placed at the beginning or end of the same context.

Depth points tested: `0% · 5% · 25% · 50% · 75% · 95% · 100%`  
(0% and 100% are Control C anchor points — see Control Cases section)

Expected shape: **U-shaped curve** — high FSR at edges, lowest at midpoint.

---

## Models Dataset

| # | Model | Provider | HuggingFace ID | Params | Context | GPU | License |
|---|-------|----------|----------------|--------|---------|-----|---------|
| 1 | **Llama-3.1-8B-Instruct** | Meta | meta-llama/Meta-Llama-3.1-8B-Instruct | 8B | 128k | dual-T4 (GQA) | Apache 2.0 |
| 2 | **Yi-Coder-9B-Chat** | 01.AI | 01-ai/Yi-Coder-9B-Chat | 9B | 128k | dual-T4 (GQA) | Apache 2.0† |
| 3 | **DeepSeek-Coder-V2-Lite-Instruct** | DeepSeek | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | 16B MoE | 128k | single T4/P100 (MLA) | DeepSeek custom‡ |

**Fairness:** All 128k context · temperature 0.0 · identical prompt · local inference only (no API rate limits or batching variance) · 4-bit quantization (nf4) applied uniformly.

†Yi-Coder-9B: 01.AI Apache 2.0 license with a registration clause for services exceeding certain MAU thresholds — not applicable at research scale.  
‡DeepSeek-Coder-V2-Lite: subject to the DeepSeek Model License (custom commercial-use terms, see model card). Permissive for non-commercial research.

### GPU assignment and architecture rationale

The three models use two different KV-cache architectures, which determines their VRAM footprint at long context:

- **GQA (Grouped-Query Attention)** — Llama-3.1-8B and Yi-Coder-9B. The KV cache grows linearly with sequence length. At full 128K context, combined weights + KV cache exceeds a single T4's 16 GB. These models **must run on a Kaggle dual-T4 session** (2×16 GB, sharded via `device_map="auto"` / accelerate).
- **MLA (Multi-head Latent Attention)** — DeepSeek-Coder-V2-Lite. MLA projects keys and values through a low-rank bottleneck, compressing the KV cache substantially. Full 128K context fits comfortably on a **single T4 or P100** (16 GB).

This is a disclosed hardware constraint, not a fairness gap: all three models see identical context content and identical prompts. On a single-GPU session, depths above 50% are skipped for GQA models (logged inline) rather than crashing silently.

```python
# config.py — actual configuration in use
MODELS = {
    "llama-3.1-8b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "gpu_requirement": "dual_t4",    # GQA — needs 2×16 GB for full 128K
        "trust_remote_code": False,
    },
    "yi-coder-9b-chat": {
        "hf_id": "01-ai/Yi-Coder-9B-Chat",
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "gpu_requirement": "dual_t4",    # GQA — needs 2×16 GB for full 128K
        "trust_remote_code": False,
    },
    "deepseek-coder-v2-lite-instruct": {
        "hf_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "gpu_requirement": "single_gpu", # MLA — fits 128K on one T4/P100
        "trust_remote_code": True,       # required for custom MLA kernels
    },
}
```

`trust_remote_code` is an explicit opt-in required by DeepSeek-Coder-V2-Lite because its MLA attention architecture ships as custom Python inside the model repo, outside the standard `transformers` package. Llama and Yi-Coder load with standard `AutoModelForCausalLM` classes and do not require it.

---

## Code Repository Dataset

### Selection criteria (all must pass)

1. **Zero compiled extensions** — no `.so` or `.pyd` files (verified by install inspection)
2. **≥ 100 passing pytest tests** on `main` branch
3. **Test suite < 60 seconds** (scoped if necessary)
4. **Active maintenance** — commit within last 12 months
5. **Permissive license** — MIT, Apache 2.0, BSD-2, BSD-3, or ISC
6. **2,000–40,000 LOC** Python source
7. **Standard pytest** — no custom test runners

### The 20 repositories — all verified pure Python, zero .so files

| # | Package | GitHub | Domain | License | Tests | LOC | Bug Type | Injection Target |
|---|---------|--------|--------|---------|-------|-----|----------|-----------------|
| 01 | **isort** | [PyCQA/isort](https://github.com/PyCQA/isort) | Import sorter | MIT | ~400 | 9.7k | A — Off-by-one | `isort/output.py` — `range()` in `sorted_imports()` |
| 02 | **httpx** | [encode/httpx](https://github.com/encode/httpx) | HTTP client | BSD-3 | ~600 | 8.8k | A — Off-by-one | `httpx/_decoders.py` — `range()` in `decode()` |
| 03 | **arrow** | [arrow-py/arrow](https://github.com/arrow-py/arrow) | Date/time | Apache 2.0 | ~500 | 10.4k | A — Off-by-one | `arrow/arrow.py` — `range()` in `interval()` |
| 04 | **loguru** | [Delgan/loguru](https://github.com/Delgan/loguru) | Logging | MIT | ~200 | 4.9k | A — Off-by-one | `loguru/_get_frame.py` — `range()` in `get_frame_fallback()` |
| 05 | **more-itertools** | [more-itertools/more-itertools](https://github.com/more-itertools/more-itertools) | Itertools | MIT | ~500 | 7.1k | A — Off-by-one | `more_itertools/more.py` — `range()` in `divide()` |
| 06 | **rich** | [Textualize/rich](https://github.com/Textualize/rich) | Terminal UI | MIT | ~400 | 38.5k | B — Boolean flip | `rich/spinner.py` — `if` condition in `update()` |
| 07 | **cerberus** | [pyeve/cerberus](https://github.com/pyeve/cerberus) | Validation | ISC | ~300 | 3.1k | B — Boolean flip | `cerberus/validator.py` — `if` in `_validate_regex()` |
| 08 | **cattrs** | [python-attrs/cattrs](https://github.com/python-attrs/cattrs) | Serialization | MIT | ~400 | 6.7k | B — Boolean flip | `src/cattrs/converters.py` — `if` in `_get_dis_func()` |
| 09 | **attrs** | [python-attrs/attrs](https://github.com/python-attrs/attrs) | Classes | MIT | ~400 | 6.2k | B — Boolean flip | `src/attr/_make.py` — `if` condition in `validator()` |
| 10 | **click** | [pallets/click](https://github.com/pallets/click) | CLI | BSD-3 | ~300 | 11.1k | B — Boolean flip | `src/click/core.py` — `if` in `_format_deprecated_suffix()` |
| 11 | **returns** | [dry-python/returns](https://github.com/dry-python/returns) | Functional/Result types | MIT | ~400 | 16.8k | C — Operator swap | `returns/contrib/pytest/plugin.py` — `==`/`!=` in `_trace_function()` |
| 12 | **marshmallow** | [marshmallow-code/marshmallow](https://github.com/marshmallow-code/marshmallow) | Serialization | MIT | ~500 | 5.0k | C — Operator swap | `src/marshmallow/schema.py` — `==`/`!=` in `_run_validator()` |
| 13 | **deepdiff** | [seperman/deepdiff](https://github.com/seperman/deepdiff) | Diff/compare | MIT | ~300 | 9.6k | C — Operator swap | `deepdiff/diff.py` — `==`/`!=` in `_diff_uuids()` |
| 14 | **sortedcontainers** | [grantjenks/python-sortedcontainers](https://github.com/grantjenks/python-sortedcontainers) | Data structures | Apache 2.0 | ~500 | 4.2k | C — Operator swap | `src/sortedcontainers/sortedlist.py` — `==`/`!=` in `bisect_left()` |
| 15 | **glom** | [mahmoud/glom](https://github.com/mahmoud/glom) | Data access | BSD-3 | ~300 | 9.5k | C — Operator swap | `glom/core.py` — `==`/`!=` in `_unpack_stack()` |
| 16 | **typer** | [fastapi/typer](https://github.com/fastapi/typer) | CLI builder | MIT | ~300 | 13.8k | D — Wrong variable | `typer/main.py` — variable swap in `solve_typer_info_defaults()` |
| 17 | **funcy** | [Suor/funcy](https://github.com/Suor/funcy) | Functional utils | BSD-2 | ~300 | 2.4k | D — Wrong variable | `funcy/colls.py` — variable swap in `zipdict()` |
| 18 | **python-dateutil** | [dateutil/dateutil](https://github.com/dateutil/dateutil) | Date parsing | Apache 2.0 | ~600 | 7.6k | D — Wrong variable | `src/dateutil/relativedelta.py` — variable swap in `__init__()` |
| 19 | **structlog** | [hynek/structlog](https://github.com/hynek/structlog) | Structured logging | Apache 2.0 | ~300 | 7.2k | D — Wrong variable | `src/structlog/stdlib.py` — variable swap in `filter_by_level()` |
| 20 | **apscheduler** | [agronholm/apscheduler](https://github.com/agronholm/apscheduler) | Job scheduler | MIT | ~300 | 5.6k | D — Wrong variable | `src/apscheduler/triggers/interval.py` — variable swap in `__setstate__()` |

**Bug type balance:** A×5 · B×5 · C×5 · D×5

**Replaced from v2 (reason):**
- `black` → `isort` — black compiles linegen.py, nodes.py and 16 other modules to Cython .so wheels
- `pydantic v1` → `cerberus` — even v1.10.21 compiles validators.py and 24 other files to .so
- `networkx` → `click` — 191k LOC far exceeds the context window budget; impossible to pad to target depths
- `schema` → `deepdiff` — single 961-LOC file; too small to build a multi-file context stack
- `toolz/boltons/schematics/tenacity/dateutil(rrule)/hypothesis/schedule` — replaced in prior audits (see v2 notes)

---

## Bug Injection Dataset

### Tool: `libcst` (concrete syntax tree — format-preserving)

All mutations use `libcst`, not `ast` or regex. This preserves original whitespace, comments, and formatting so models cannot detect mutations via formatting anomalies.

### Mandatory gate before every LLM trial

```python
def validate_injection(repo_path, mutated_file, target_file):
    failures = run_pytest(repo_path, mutated_file)
    assert 1 <= len(failures) <= 10, (
        f"Mutation invalid: {len(failures)} test failures "
        f"(need 1–10 for clean signal)"
    )
    return failures
```

If this gate fails → pick a different expression in the same function, re-validate. Never proceed with 0 failures (invisible mutation) or >10 failures (too destructive).

### Type A — Off-by-one

```python
# ORIGINAL
for i in range(len(items)):
    process(items[i])

# INJECTED
for i in range(len(items) - 1):   # silently skips last item
    process(items[i])
```

**libcst target:** `Call` nodes where `func.id == 'range'`  
**Pre-screen:** confirm at least one test checks exhaustive iteration or output length  
**Validity:** HIGH — deterministic, always produces off-by-one in bounded iteration  
**Repos:** isort · httpx · arrow · loguru · more-itertools

### Type B — Boolean flip

```python
# ORIGINAL
if self.follow_redirects:
    return self._redirect(request)

# INJECTED
if not self.follow_redirects:
    return self._redirect(request)
```

**libcst target:** `Constant(value=True/False)` or `If` test nodes  
**Pre-screen:** run coverage check — confirm flipped branch is exercised by ≥1 test  
**Validity:** MEDIUM-HIGH — fails if target boolean is never reached in tests  
**Repos:** rich · cerberus · cattrs · attrs · click

### Type C — Operator swap

```python
# ORIGINAL
if left == right:
    return True

# INJECTED
if left != right:     # equality inverted
    return True
```

**libcst target:** `Compare` nodes — prefer `Eq ↔ NotEq`; avoid `And ↔ Or` (may be logically equivalent)  
**Validity:** MEDIUM — **highest false-positive risk**. `== → !=` on an untested code path causes 0 failures. Dry-run gate is MANDATORY for this type.  
**Repos:** returns · marshmallow · deepdiff · sortedcontainers · glom

### Type D — Wrong variable

```python
# ORIGINAL
result = compute(start=start, end=end)

# INJECTED
result = compute(start=end, end=start)   # variables transposed
```

**libcst target:** `Name` nodes — use `rope` for scope analysis to find same-type, same-scope variable pairs  
**Pre-screen:** inspect test fixtures and confirm the two variables hold **asymmetric values** (e.g., `start=1, end=10` — not `start=end=5`)  
**Validity:** MEDIUM — fails silently if variables happen to be equal in all tests  
**Repos:** typer · funcy · python-dateutil · structlog · apscheduler

---

## Control Cases

Three controls are required. Without them, failures cannot be attributed to positional bias.

### Control A — Capability isolation (60 calls)

Send **only** the single buggy function — no file context, no padding. If the model cannot fix it here, the model lacks the capability for this repo. Discard that repo/model pair from all depth trials.

```python
prompt = f"Fix the bug in this function. Return only the corrected function.\n\n{buggy_function}"
result = call_model(model, prompt, temperature=0.0)
assert tests_pass(result), "DISCARD — capability failure, not positional bias"
```

Calls: 20 repos × 3 models = **60 calls**

### Control B — Hallucination check (20 calls)

Send the full padded context at 50% depth with **no bug injected**. All tests pass. If the model edits working code anyway, it is hallucinating bugs under deep context — a separate confound that must be reported.

Calls: 20 repos × 1 model (llama-3.1-8b-instruct spot-check) = **20 calls**

### Control C — Curve anchors (120 calls)

Inject the bug at exactly **0%** (first token of first file) and **100%** (last token of last file). These set the theoretical performance ceiling. Without them the U-curve has no endpoints and cannot be fitted.

```python
DEPTHS = [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]
#          ^^^^                                  ^^^^  ← Control C anchor points
```

Calls: 20 repos × 2 depths × 3 models = **120 calls**

### Total calls

| Run type | Calls |
|----------|-------|
| Main experiment (5 depths) | 300 |
| Control A (capability) | 60 |
| Control B (hallucination) | 20 |
| Control C (anchors) | 120 |
| **Total** | **500** |

---

## Step-by-Step Methodology

### Phase 1 — Environment setup (Day 1–2)

```bash
mkdir tracelimit && cd tracelimit && mkdir repos results logs

# requirements.txt covers: experiment packages, ML inference stack,
# pytest plugins, and all test-suite deps for the 20 repos.
pip install -r requirements.txt

# Confirm zero .so files across all 20
python -c "
import site, os
for pkg in ['isort','httpx','arrow','loguru','returns','rich','cerberus',
            'cattrs','attr','click','marshmallow','deepdiff','sortedcontainers',
            'more_itertools','glom','typer','funcy','dateutil','structlog','apscheduler']:
    for sp in site.getsitepackages():
        path = os.path.join(sp, pkg)
        if os.path.isdir(path):
            so = [f for r,_,fs in os.walk(path) for f in fs if f.endswith(('.so','.pyd'))]
            print(f'{pkg}: {\"CLEAN\" if not so else so}')
            break
"

# .env
echo 'TOGETHER_API_KEY=your_key' >> .env
echo 'MISTRAL_API_KEY=your_key'  >> .env
```

### Phase 2 — Baseline (Control A, Day 3)

```python
# baseline.py — run before any depth trial
for repo in REPOS:
    for model in MODELS:
        buggy_fn = inject_and_extract_function(repo)
        result = call_model(model, f"Fix this function:\n{buggy_fn}", temperature=0.0)
        if not tests_pass(repo, result):
            EXCLUDED.add((repo['name'], model['name']))
            print(f"EXCLUDED: {repo['name']} × {model['name']} — capability failure")
```

### Phase 3 — Injection pipeline (Day 4–7)

```python
# injector.py
import libcst as cst

DEPTHS = [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]

def inject_at_depth(repo_path, target_file, target_fn, bug_type, depth):
    files = sorted(collect_python_files(repo_path))   # alphabetical — deterministic
    context = concatenate(files)
    context = pad_to_depth(context, target_file, target_fn, depth)  # ±2% tolerance
    mutated = apply_mutation(read(target_file), target_fn, bug_type)  # libcst
    return context, mutated

def validate(repo_path, mutated_file):
    failures = run_pytest(repo_path, mutated_file)
    assert 1 <= len(failures) <= 10
    return failures
```

### Phase 4 — Evaluation loop (Weeks 3–5)

```python
# evaluate.py
results = []

for repo in REPOS:
    for depth in DEPTHS:
        context, mutated = inject_at_depth(repo, depth)
        failures = validate(repo, mutated)          # gate — abort if invalid

        for model in MODELS:
            if (repo['name'], model['name']) in EXCLUDED:
                continue

            response = call_model(model, build_prompt(context, failures), temperature=0.0)
            fixed = extract_code_block(response)
            success = apply_and_test(repo, fixed)

            results.append({
                "repo": repo["name"], "model": model["name"],
                "depth": depth, "bug_type": repo["bug_type"],
                "success": int(success),
                "context_tokens": count_tokens(context),
            })

        time.sleep(RATE_LIMITS[model["name"]]["sleep"])

pd.DataFrame(results).to_csv("results/raw_results.csv", index=False)
```

### Prompt template (identical for all 3 models)

```
SYSTEM:
You are an expert Python debugger. Given a large multi-file Python codebase
and failing test output, find and fix the single bug. Return ONLY the corrected
version of the file that contains the bug inside a ```python block.
Do not modify other files. Do not explain.

USER:
--- FAILING TEST OUTPUT ---
{pytest_output}

--- REPOSITORY SOURCE ---
{full_context}

Return the corrected file: {target_filename}
```

### Phase 5 — Analysis (Week 8)

```python
# analyse.py
df = pd.read_csv("results/raw_results.csv")
pivot = df.groupby(["model","depth"])["success"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
for model, grp in pivot.groupby("model"):
    ax.plot(grp["depth"]*100, grp["success"]*100, marker="o", linewidth=2, label=model)

ax.axvspan(40, 60, alpha=0.08, color="red", label="Predicted trough zone")
ax.set_xlabel("Bug depth in context window (%)")
ax.set_ylabel("Fix success rate (%)")
ax.set_xticks([0, 5, 25, 50, 75, 95, 100])
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig("results/positional_bias_curve.png", dpi=150)
```

---

## Execution Guide

```bash
# Week 1–2: setup
pip install -r requirements.txt
python verify_clean.py          # confirms zero .so across all 20 packages
python verify_baselines.py      # must show 20/20 PASS per model

# Pre-flight
python injector.py --dry-run    # confirms 1–10 test failures per mutation

# Week 3–5: run (Kaggle dual-T4 session required for llama and yi-coder)
python evaluate.py --model llama-3.1-8b-instruct        # 140 calls (100 main + 40 controls)
python evaluate.py --model yi-coder-9b-chat              # 140 calls
python evaluate.py --model deepseek-coder-v2-lite-instruct  # 140 calls (single GPU ok)
python control_b.py                         # 20 calls (hallucination check, llama spot-check)

# Week 8: analysis
python analyse.py
```

### Rate limits

All three models run locally via Kaggle GPU sessions — no API rate limits apply. Sleep is 0 between calls; throughput is limited only by GPU inference speed.

| Model | GPU session | Sleep | Notes |
|-------|------------|-------|-------|
| Llama-3.1-8B-Instruct | dual-T4 | 0 s | GQA; needs 2×16 GB for full 128K |
| Yi-Coder-9B-Chat | dual-T4 | 0 s | GQA; needs 2×16 GB for full 128K |
| DeepSeek-Coder-V2-Lite-Instruct | single T4/P100 | 0 s | MLA; fits 128K on 16 GB |

---

## Measurement & Analysis

**Primary metric:** Fix Success Rate (FSR) = passing trials / total trials per depth point.

**Secondary metrics:**
- Token depth at FSR < 0.5 (exact failure threshold)
- Bug type sensitivity — 2-way ANOVA: model × bug_type × depth
- False fix rate — model patches tests to pass without fixing the actual bug (detect by diffing against known correct patch)

**Expected results (hypothesis):**

| Depth | Expected FSR |
|-------|-------------|
| 0% | 0.88 – 0.98 |
| 5% | 0.80 – 0.93 |
| 25% | 0.65 – 0.80 |
| 50% | 0.35 – 0.55 ← trough |
| 75% | 0.60 – 0.75 |
| 95% | 0.75 – 0.90 |
| 100% | 0.85 – 0.97 |

---

## Fairness Controls

1. **Zero compiled extensions** — all 20 packages verified with zero `.so`/`.pyd` files
2. **Temperature 0.0** — all models, all trials
3. **Identical prompt** — same system + user message across all 3 models
4. **Identical 128k context** — no truncation at any depth for any model
5. **One bug per repo per trial** — model never sees two bugs simultaneously
6. **Baseline gate (Control A)** — repo excluded if model can't fix bug in isolation
7. **Injection gate** — 1–10 test failures required before any LLM call
8. **Alphabetical file ordering** — deterministic context construction, no randomness
9. **Depth tolerance ±2%** — context padded precisely before every call
10. **Docker isolation** — fresh container per trial, no state carryover
11. **Balanced bug types** — exactly 5 repos per type, prevents difficulty confounds
12. **Disclosed GPU split** — Llama-3.1-8B and Yi-Coder-9B run on dual-T4 (GQA attention requires 2×16 GB at 128K context); DeepSeek-Coder-V2-Lite runs on single T4/P100 (MLA compressed KV fits 128K in 16 GB). All models receive identical context; the split is a hardware constraint, not a capability advantage. On single-GPU sessions, depths >50% are skipped for GQA models and logged — not silently dropped.

---

*TraceLimit v3.2 · AI × Software Engineering · 2025*  
*Models: Llama-3.1-8B-Instruct · Yi-Coder-9B-Chat · DeepSeek-Coder-V2-Lite-Instruct*  
*All 20 repos: verified pure Python, zero compiled extensions*  
*Injection targets updated to validated dry-run values; trust_remote_code added to MODELS config*
