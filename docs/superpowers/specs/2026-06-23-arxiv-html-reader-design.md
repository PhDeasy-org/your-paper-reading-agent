# Read Papers from arXiv HTML (Replace HF CLI Content + PyMuPDF Figures)

## Summary

Replace the paper-content and figure-extraction pipeline. Today the pipeline
reads full text via `hf papers read` and extracts figures from the PDF via
PyMuPDF caption-detection (which mis-crops frequently, then a vision LLM
attempts to pick the best crops). The new design reads the **arXiv HTML
version** of the paper, which provides clean author-authored text *and*
author-provided figures in one fetch — no mis-cropping, no vision LLM required.

**Scope (decided in brainstorming):**

| Concern | Decision |
|---|---|
| HF CLI removal | **Content + figures only.** HF CLI stays for paper discovery (`list_papers`, `search_papers`) and metadata (`paper_info`). |
| Figure storage | **Local files** in `report_dir/figures/figure_N.<ext>`, referenced by relative path (matches today). |
| Vision LLM | **Dropped entirely.** The `vision` LLM role, the `figure_selector` agent, and `chat_vision()` are deleted. |
| Fallback when no HTML | **PDF text-only fallback.** Older papers without arXiv HTML get text via the existing `pdf.extract_text`, and **no figures**. |
| Figure → section mapping | **Deterministic**, based on the paper section the figure appears in (with caption keywords as tiebreaker). No LLM call. |
| Figure cap | **Configurable** — new `report.max_figures` setting, default 8. |

---

## 1. Architecture & Module Layout

### New module: `src/ppagent/arxiv_html.py`

Single owner of arXiv HTML fetch + parse. Public API:

```python
@dataclass
class ParsedHtml:
    markdown: str                          # HTML body → markdown for Writer/Criticizer
    figures: list[Figure]                  # caption + image_path (already downloaded)
    figure_sections: dict[int, str]        # figure_number → FIGURE_SECTIONS key

class HtmlUnavailable(Exception): ...      # HTTP 404, network failure on the HTML page
class ParseError(Exception): ...           # 200 OK but body unrecognizable as a paper

def fetch_and_parse(
    paper_id: str,
    out_dir: Path,
    *,
    max_figures: int = 8,
) -> ParsedHtml: ...
```

It does three things, in order:

1. **Fetch** `https://arxiv.org/html/{paper_id}` (no version suffix; arXiv redirects
   to the latest version and `httpx` follows with `follow_redirects=True`). One GET.
2. **Parse** the body with stdlib `html.parser.HTMLParser` into (a) a markdown
   string and (b) a list of figures, each tagged with the section heading it
   appeared under.
3. **Download** each figure's image into `out_dir/figures/figure_N.<ext>` and
   set `Figure.image_path = "figures/figure_N.<ext>"` — the exact shape
   `figures.py` produces today, so `assembler.py` and the Jinja templates need
   **zero** changes.

### Data symbol migration

`Figure`, `SelectedFigure`, and `FIGURE_SECTIONS` currently live in
`figures.py`. Because `assembler.py` (kept) imports them, they **migrate** to
`arxiv_html.py` rather than disappearing. `assembler.py` and `pipeline.py`
update their imports from `ppagent.figures` → `ppagent.arxiv_html`.

### Deterministic section mapping

Lives in `arxiv_html.py`:

```python
def _map_section(paper_section_title: str, caption: str) -> str
# Returns one of FIGURE_SECTIONS: "method" | "evaluation" | "benchmarks" | "previous_works"
```

Rule-based. Inputs are lowercased and stripped of arXiv's leading numeric
prefix (e.g. `"3.4 The Training Cost"` → `"the training cost"`) before
matching. The paper section is the primary signal; caption keywords break ties
when the section title is generic.

Keyword matching is prefix/substring on the normalized title+caption, in table
order; the first matching row wins:

| Paper section keyword (lowercased substring) | Maps to |
|---|---|
| `intro`, `method`, `approach`, `framework`, `model`, `architecture`, `preliminar` | `method` |
| `eval`, `result`, `experiment`, `finding`, `analysis` | `evaluation` |
| `benchmark`, `dataset`, `setup`, `data` | `benchmarks` |
| `related`, `prior`, `previous work` | `previous_works` |
| (no match) | `method` (today's fallback) |

### What gets deleted

- `src/ppagent/figures.py` — gone (PyMuPDF cropping; symbols migrate to `arxiv_html.py`).
- `src/ppagent/agents/figure_selector.py` — gone.
- `FIGURE_SELECTOR_SYSTEM_PROMPT` / `FIGURE_SELECTOR_USER_PROMPT_TEMPLATE` in `prompts.py` — gone.
- `vision` LLM role in `config.py` (`LLMsConfig.vision`, `_vision_default`,
  `AGENT_LLM_ROLE["figure_selector"]`) — gone.
- `LLMClient.chat_vision()` + `_image_to_data_uri()` in `llm.py` — gone.
- `FigureSelectorAgent` import in `agents/__init__.py` — gone.
- Vision-LLM menu entries in `tui.py` — gone (see §4).

### What stays untouched

`assembler.py`, both Jinja templates, `models.py`, `hf.py` (discovery +
metadata), all other agents (writer/finder/criticizer/classifier/searcher),
`pdf.py` (fallback text extraction).

---

## 2. Parsing Strategy

A single `html.parser.HTMLParser` subclass walks the document in order,
maintaining a small amount of state. Two outputs in one pass.

### Output 1 — Markdown text (for the Writer/Criticizer)

| HTML | Markdown |
|---|---|
| `<h1>` | `# ` (strip arXiv's "1 " numeric prefix, keep words) |
| `<h2>`, `<h3>` | `## `, `### ` (numeric prefix stripped) |
| `<p>` | paragraph, blank-line separated |
| `<ul>` / `<ol>` / `<li>` | `- ` / `1. ` list items |
| `<table>` | pipe-table (the existing `render_markdown_with_math` already enables the `tables` extension) |
| `<math>` block | `$$...$$` — extract original LaTeX from the inner `<annotation encoding="application/x-tex">` element when present; otherwise skip the block |
| `<math>` inline | `$...$` (same `<annotation>` extraction) |
| `<figure>`, `<figcaption>`, `<img>` | **Skip** in the markdown output (figures render in their own blocks; the Writer doesn't need their captions inline) |
| `<section class="ltx_bibliography">` (References) | **Skip** entirely — saves context tokens; the Finder agent already does its own related-work discovery |
| `<nav>`, arXiv chrome header/footer | **Skip** |

### Output 2 — Figures list

For each `<figure>` element that contains a descendant `<img>`:

- **`src`**: resolved relative URL via `urllib.parse.urljoin(page_url, src)`.
  - Skip `data:image/...;base64,...` URIs (inline icons, not figures).
  - Skip `/static/...` arXiv chrome logos.
  - Absolute `https://...` URLs used as-is.
- **caption**: the `<figcaption>` text, kept verbatim (including the
  "Figure N:" label — templates render it, and the showcase shows captions like
  "Figure 1: Taxonomy...").
- **section context**: the most-recent `<h2>`/`<h3 class="ltx_title">` text seen
  before the `<figure>`. Verified reliable on the sample paper.
- **`figure_number`**: Nth figure-with-image encountered (1-indexed), **not** the
  number in the caption text. Stable across papers with unnumbered sub-figures.
- **Sub-figures**: a `<figure>` may wrap multiple `<img>`s under one caption
  (e.g. `x2.png` + `x3.png`). Each `<img>` becomes its own `Figure`,
  inheriting the parent caption + section.
- **Tables-as-figures**: silently skip any `<figure>` with no `<img>` descendant
  (their content is already captured as a markdown table in Output 1).

---

## 3. Pipeline Integration & Fallback

### Phase reorganization (8 → 6 phases)

| New # | Old # | Phase |
|-------|-------|-------|
| 1 | 1 | Fetch metadata (HF/arXiv API — unchanged) |
| 2 | 2 | **Fetch + parse arXiv HTML** → markdown + figures (new behavior) |
| 3 | 3 | Classify paper type (unchanged) |
| 4 | 4 | Writer ‖ Finder (unchanged) |
| 5 | 5 | Criticizer (unchanged) |
| 6 | 8 | **Assemble** + publish (old Phase 6 "Ensure PDF" and old Phase 7 "Extract+select figures" deleted) |

### Phase 2 logic

```python
paper_dir = self.storage.paper_dir(paper.title, paper.published_at)
try:
    parsed = arxiv_html.fetch_and_parse(
        paper_id, paper_dir, max_figures=self.config.report.max_figures
    )
    content_md = parsed.markdown
    selected = [
        SelectedFigure(fig, parsed.figure_sections[fig.figure_number])
        for fig in parsed.figures
    ]
    source = "arXiv HTML"
except (arxiv_html.HtmlUnavailable, arxiv_html.ParseError) as exc:
    # Fallback: PDF text only, no figures.
    if self.config.report.download_pdf:
        pdf_path = pdf.download_pdf(paper, self.config.pdf_cache_dir)
        content_md = pdf.extract_text(pdf_path)
    else:
        content_md = paper.summary or "Paper content unavailable."
    selected = []
    source = f"arXiv PDF (no HTML: {exc})"
```

Key points:

- **One HTTP fetch** for both text and figures (HTML page contains everything).
  Image downloads happen inside `fetch_and_parse`, sequentially.
- **`paper_dir`** computed identically to today, so figures land next to
  `report.html` as now.
- **All parsed figures selected by default.** Every figure-with-image becomes a
  `SelectedFigure`, assigned to its deterministic section. No "is this worth
  inserting?" gating — arXiv HTML contains only figures the authors chose to
  include.
- **`max_figures` cap**: applied during parsing — only the first
  `max_figures` figures (in document order) are downloaded and returned.

### `pdf.py` simplification

With `figures.py` gone, `pdf.py` keeps only `download_pdf` + `extract_text`.
**PyMuPDF stays as a dependency** (still needed for fallback text extraction).
The `report.download_pdf` config flag still gates whether the fallback path may
download; if a user disables it *and* HTML is missing, they get a content-only
report from the abstract (existing behavior at `pipeline.py:256-260`).

### "HTML unavailable" — `HtmlUnavailable` raised when:

- HTTP 404 on `https://arxiv.org/html/{id}` (older papers, withdrawn).
- Network/timeout error on the HTML fetch.

### `ParseError` raised when:

- HTTP 200 but the body has no recognizable paper structure (no `<article>`, no
  body text). Treated identically to `HtmlUnavailable` by the pipeline → PDF
  text-only fallback.

---

## 4. Config, TUI & Removals

### `config.py`

`LLMsConfig` loses the `vision` role:

```python
class LLMsConfig(BaseModel):
    text: LLMConfig = Field(default_factory=LLMConfig)
    searcher: LLMConfig = Field(default_factory=LLMConfig)
    # vision: removed
    saved_vendors: dict[...] = ...
```

Ripple effects:
- `for_role()` — drop `"vision"` from the allowed set: `("text", "searcher")`.
- `_LLM_ROLES` constant — drop `"vision"`.
- `_apply_env_overrides()` — `PPA_LLM_*` env vars apply to text + searcher only.
- `_migrate_legacy_llm()` — clones flat `[llm]` into text + searcher (was text + vision + searcher).
- `_vision_default()` helper — deleted.
- `AGENT_LLM_ROLE` — drop the `"figure_selector": "vision"` entry.

`ReportConfig` gains:

```python
max_figures: int = Field(
    default=8,
    description="Maximum number of figures to extract from the paper's arXiv HTML and insert into the report.",
)
```

`AppConfig.pdf_cache_dir` property stays (fallback path uses it).

### `tui.py`

The config menu has a "Vision LLM" entry (figure selector) and several regex
patterns that match `llm_(text|vision|searcher)_...` menu IDs. Changes:
- Drop the `llm_vision_vendor` menu entry and its description.
- Update the three regexes from `text|vision|searcher` → `text|searcher`:
  - `^llm_(text|vision|searcher)_vendor$`
  - `^llm_(text|vision|searcher)_([a-z0-9_]+)_latest$`
  - `^llm_(text|vision|searcher)_([a-z0-9_]+)$`
- Drop the `"vision": "Vision LLM"` label mapping.
- Update the top-level menu description to mention only text + searcher roles.

### `cli.py`

`config_show` drops its "Vision LLM (figure_selector)" print line.

### `agents/__init__.py`

Drop `from ppagent.agents.figure_selector import FigureSelectorAgent`.

### `pyproject.toml`

No dependency changes. `PyMuPDF` stays (used by `pdf.py` fallback);
`huggingface_hub[cli]` stays (used for discovery/metadata).

### `README.md`

Update the "How it works" section: figure extraction now reads arXiv HTML, not
PyMuPDF; remove the "vision agent" mention; the install instructions drop the
implicit need for a vision-capable model.

---

## 5. Edge Cases & Risks

1. **arXiv HTML URL & versioning.** Fetch `https://arxiv.org/html/{paper_id}`
   (no version). arXiv serves the latest version and redirects to
   `/html/{id}vN`; `httpx` follows with `follow_redirects=True`. The final URL
   becomes the base for resolving relative image `src` values.

2. **Image URL resolution.** `urllib.parse.urljoin(page_url, src)` handles all
   three cases: relative (`2606.01075v2/x1.png`), absolute (`https://...`),
   and we explicitly skip `data:` URIs and `/static/...` chrome logos.

3. **Image format.** arXiv images are typically `.png`, sometimes `.jpg`/`.gif`.
   Preserve the original extension in the local filename
   (`figure_1.png`, `figure_2.jpg`). Template `<img src>` is extension-agnostic.

4. **SVG figures.** Skip `.svg` images in v1 (rare; raster download is already
   complex enough). Debug log, revisit if it becomes a real gap.

5. **Image download failure.** If one image 404s or times out, skip that single
   figure (don't fail the whole parse). Each download is independent; partial
   success is the norm.

6. **Rate limiting.** ≤ (1 HTML page + ~8 images) requests per paper,
   sequential, with `User-Agent: ppagent/1.0` and the existing 120s timeout.
   No concurrency in v1 — YAGNI.

7. **Parser robustness.** `HTMLParser` never raises on unexpected tags; unknown
   tags pass through silently (their text still flows into markdown). The only
   hard failure is "page fetched but no recognizable body" → `ParseError`.

8. **Memory.** ~350KB HTML parsed in one pass is trivial. Images are streamed
   to disk (`response.iter_bytes()` → file), not held in memory.

9. **`Figure.image_path` portability.** Stays `figures/figure_N.<ext>` (relative).
   No template or storage changes.

10. **Backward compatibility.** Existing reports on disk are untouched. A
    re-run regenerates from HTML (possibly different figure numbers, which is
    expected on regeneration).

---

## 6. Test Plan

### New: `tests/test_arxiv_html.py`

| Test | What it verifies |
|---|---|
| `test_parser_unit` | Synthetic HTML fixture (`<figure>` with `<img>` + `<figcaption>` under an `<h2>`) → caption captured, section mapped to `method`, image URL resolved, heading in markdown as `## Method`. |
| `test_section_mapping` | Table-driven: caption/section strings → expected `FIGURE_SECTIONS` key. Covers intro/method/eval/results/benchmark/dataset/related + default fallback. |
| `test_mathml_to_latex` | Synthetic `<math><annotation encoding="application/x-tex">...</annotation></math>` → `$...$` inline and `$$...$$` block. |
| `test_skip_data_uri_and_static` | `data:image/...` and `/static/...` image sources are skipped. |
| `test_subfigures_share_caption` | One `<figure>` with two `<img>`s → two `Figure`s, both with the parent caption + section. |
| `test_max_figures_cap` | HTML with 12 figures, `max_figures=4` → only 4 downloaded/returned (first 4 in document order). |
| `test_html_unavailable_raises` | `fetch_and_parse` raises `HtmlUnavailable` on HTTP 404 (via mocked `httpx`). |
| `test_parse_error_raises` | 200 OK but empty/non-paper body → raises `ParseError`. |
| `test_fetch_and_parse_live` (skip-if-no-network) | Real fetch of a known HTML paper (e.g. `2606.01075`); asserts markdown non-empty, ≥1 figure, figure files on disk, all `figure_sections` values ∈ `FIGURE_SECTIONS`. |

### Deleted

- `tests/test_figures.py` — tests the removed `figures.py` and `FigureSelectorAgent`.

### Regression (must keep passing, no edits expected)

`test_agent_base.py`, `test_classifier.py`, `test_prompts.py`,
`test_finder_tools.py`, `test_xai_tool.py`, `test_streaming.py`,
`test_llm_errors.py`, `test_providers.py`, `test_pricing.py`,
`test_publishers.py`, `test_latex_rendering.py`, `test_config_persistence.py`,
`test_tui_config.py`, `test_cli.py`.

### Quality gates

- `ruff check` clean on all new and modified files.
- All function signatures and class definitions fully typed (AGENTS.md).
- `uv run pytest` fully green.
