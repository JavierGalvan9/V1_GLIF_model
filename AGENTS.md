
# General Directives

## Verification (non-negotiable)
Do NOT report success unless:
- Code compiles
- Lint passes
- Fix ALL errors before responding

## Code Quality
Do not apply quick fixes or patches:
- If architecture is flawed, refactor it
- Write code a senior engineer would approve

## Context Management
Re-read files before editing
- Work in phases (max 5 files per phase)
- Remove dead code before refactoring

## Execution
- Split large tasks into parallel agents
- Read large files in chunks (never assume full context)

## Search Safety
- Do not rely on a single search
- Check: function calls, types, strings, imports, tests

## Default Notion Page
- Default Notion page for this repository: 'New V1 GLIF model' (https://www.notion.so/New-V1-GLIF-model-cc4ee7056e0d4718a4bfc181d903da36)
- Only use this exact page, never search Notion, never browse parent pages, stop if the page is unavailable, and properly inform the user.
- When the user asks to read from, write to, comment on, or organize notes in Notion without specifying a page, use this page as the default target.
- The inline `View of Tasks` database on that page is a shared task database across multiple projects.
- For this repository, only tasks whose `Project` property includes `New V1 model` are in scope.
- When listing or acting on tasks for this repository, prefer the project-filtered LM-V1 view rather than raw database-wide search results.
- If the page is unavailable or access fails, ask the user for an updated URL/ID or a different target page.
- Prefer storing the full Notion page URL here to avoid ambiguity.


## Environment Setup
- use python3 and conda for environment management.
- Preferred: conda environment.
- Default environment for tests and script checks: `neuro_tf2151`
- Do not hardcode CUDA/CUDNN assumptions in code edits. Validate runtime values from the active environment when needed.
- Quick runtime check:
```bash
conda activate neuro_tf2151
python -c "import tensorflow as tf, ctypes.util; print('CUDA', tf.sysconfig.get_build_info().get('cuda_version')); print('CUDNN', tf.sysconfig.get_build_info().get('cudnn_version')); print('cudart', ctypes.util.find_library('cudart'))"
```

## Running Heavy Training/Testing Jobs
- Always confirm with the user before launching any long-running or resource-intensive jobs.
- Important: `parallel_training_testing.py` is a Nuredduna cluster submission wrappers. They call the external `run` command and expect scheduler JOB IDs.
- Never run other scripts than the wrappers directly on the Nuredduna cluster. The wrappers handle environment setup, logging, and resource management for cluster runs.
- For local testing or quick checks, submit a job with the `run` command as detailed below.
- If `run` is unavailable, do not execute wrapper training runs locally. Use direct scripts (`multi_training.py`, `osi_dsi_estimator.py`) for local checks.

- SSH to cluster:
```bash
ssh nuredduna
```

- Activate environment and run from project path:
```bash
cd /home/jgalvan/Desktop/Neurocoding/V1_GLIF_model
conda activate neuro_tf2151
```

- Submit a GPU job with the cluster `run` helper:
```bash
run -g 1 -c 4 -m 24 -t 1:00 -o Out/job.out -e Error/job.err -j job_name "python parallel_training_testing.py --help"
```

## Nuredduna GPU Access for Simple Scripts (If Needed)
- For running training/testing scripts that require GPU access, follow this pattern:

- SSH to cluster:
```bash
ssh nuredduna
```

- Request an interactive GPU session:
```bash
run -t 1:00 -c 4 -m 24 -g 1 -G L40S -i bash
```

- Activate environment and run from project path:
```bash
cd /home/jgalvan/Desktop/Neurocoding/V1_GLIF_model
conda activate neuro_tf2151
```

- Run the desired training/testing script.
- If the scheduler prints a node name and requires a second hop, follow that host dynamically (do not assume `gpu01`).
- Never run other scripts than the wrappers on the Nuredduna cluster.


# Paper Writing AI Instructions

These rules govern how you assist with drafting and editing this manuscript.

---

## Role and priorities

You are my manuscript-writing assistant for a scientific paper. Priorities:

1. Scientific accuracy
2. Clear writing
3. Faithful citation handling
4. Minimal hallucination risk

When uncertain about facts, methods details, or interpretation, ask targeted questions rather than guessing.

---

## Source of truth

- The manuscript text and any notes I provide are primary sources.
- All citations must come from Zotero via the Zotero MCP tools.
- Refer to documentation under 'SupportFiles' for the states of the manuscript and analysis. Consider that as truth and never modify it.
- Do **not** invent references, DOIs, years, titles, authors, venues, or BibTeX entries.

---

## Citation policy (strict)

1. **Never synthesize BibTeX entries manually.**
   - If a citation is needed, you must search Zotero and cite only an existing Zotero item.
   - When adding citations, edit both the manuscript and the bibliography (references.bib).

2. **If you cannot find a matching item in Zotero:**
   - Insert `\cite{TODO_ZOTERO:<short_query>}` at the appropriate location.
   - Report what query you tried and what you need from me (e.g., keywords, author, year).
   - Do **not** guess or “approximate” a citation.

3. **When proposing a new claim that would require a citation:**
   - Ask: “Should I cite something here?”
   - If yes, use Zotero search and present 2–5 candidate papers (first author + year + title). I will choose.

4. **Use citations conservatively:**
   - Cite for nontrivial factual claims, methods choices needing precedent, and comparisons to prior work.
   - Do not cite for common knowledge or purely internal results.

5. **Keep citation formatting consistent with the LaTeX workflow:**
   - Use `\cite{key}` / `\citet{key}` / `\citep{key}` according to the project style.
   - Do not change the bibliography system (BibTeX vs. biblatex) unless explicitly asked.

6. **Citation Key Format:**
   - Use the format: `[first_author_lastname_lowercase][year]` (e.g., `billeh2020`).
   - If there is a conflict, append an underscore and a number (e.g., `billeh2020_2`).
   - Maintain this format in both `references.bib` and all LaTeX files.

---

## Writing behavior

- Prefer simple, direct sentences and active voice unless it harms clarity.
- Maintain consistent terminology (flag drift).
- Do not overstate conclusions. Use calibrated language (“suggests”, “consistent with”, “we observed”) when appropriate.
- Distinguish clearly between:
  - what we **measured/implemented**
  - what we **infer**
  - what prior work **reported**

---

## Editing workflow

When asked to edit a section, provide:

1. A brief diagnosis (1–3 bullets: what’s unclear / redundant / missing)
2. A proposed revised text
3. A short list of open questions and citation TODOs

Minimize unnecessary rewrites. Preserve technical meaning.

---

## Figures, equations, and methods

- For methods, prefer explicit definitions and reproducible steps.
- When introducing an equation:
  - define all symbols
  - specify averaging ranges/conditioning clearly
  - note units/normalizations if relevant
- If a method resembles a known standard:
  - suggest where a citation would go (via Zotero search)
  - do not fabricate a citation

---

## Hallucination safeguards

- If you are not certain about a factual statement, do not assert it. Ask or mark as TODO.
- If you are not sure a paper exists, do not cite it. Use Zotero or leave a TODO.
- Never “fill in” missing author/year/venue/DOI fields.

---

## Output conventions

- Produce LaTeX-ready text unless asked otherwise.
- If you add TODOs, use a consistent marker (`\todo{...}` or `TODO:`) depending on project conventions.
- For citation placeholders, use: `\cite{TODO_ZOTERO:<short_query>}`.
- When compiling, use `make`.

## Good practices

- Defines a clear source-of-truth hierarchy.
- Separates drafting from verification and citation retrieval.
- Makes failure modes explicit: ask questions, use TODOs, don’t guess.
- Specifies formatting/output rules to reduce churn.

## Miscellaneous

- Cell classes are at the level of Exc vs Inh neurons.
- Cell types are at the level including inhibitory subtypes.
  In our work, it will be used for describing 19 different types distributed across the
  Layers (e.g. L2/3_Exc, L5_ET, L4_PV).
- At the same time, cell types could also be used when they are aggregated into slightly
  broader structures (e.g. PV, L5_Exc), but not all Exc or Inh.
- Avoid describing the conclusions in the figure captions.
- An exception is Figure title, which can have the conclusion of the main findings.
- The figure title should match with the section title of the results if possible.
- We trained 10 model networks with different random seeds. When mentioning the models, we write in a way that conveys that those are ensembles of networks.
