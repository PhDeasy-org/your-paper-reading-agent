# ppagent

Personalized arXiv paper discovery and automated report generation using AI agents.

`ppagent` fetches trending papers from [HuggingFace Daily Papers](https://huggingface.co/papers), ranks them against your research profile, and generates structured reading reports (Markdown + HTML) via a multi-agent pipeline.

---

## Features

- **Personalized paper search** — ranks papers by relevance to your research profile
- **Multi-agent report generation** — Writer, Finder, and Criticizer agents produce:
  - Metadata, keywords, affiliations, benchmarks
  - TL;DR, method details, performance evaluation
  - Related works discovery
  - Critical analysis and limitations
- **Dual output formats** — Markdown and styled HTML reports
- **Auto-fetch scheduler** — run the pipeline on a cron schedule
- **Optional publishing** — push reports to WeChat, Notion, or a personal blog
- **Extensible agents** — drop in custom agents via Python plugins
- **TOML-based configuration** — all settings in one file, env var overrides for secrets

---

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url>
cd daily-paper-reading
uv sync
```

Also requires the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli):

```bash
uv pip install huggingface_hub[cli]
```

---

## Quick Start

```bash
# 1. Create default config
uv run ppagent config init

# 2. Add your LLM API key
#    Edit config/settings.toml → set llm.api_key
#    Or: export PPA_LLM_API_KEY=sk-...

# 3. Customize your research profile
#    Edit config/profile.md

# 4. Discover papers
uv run ppagent search

# 5. Generate a report for a specific paper
uv run ppagent report 2506.12345

# 6. Full pipeline: search + report all matches
uv run ppagent run
```

---

## CLI Reference

```
ppagent [OPTIONS] COMMAND [ARGS]...
```

| Command | Description |
|---------|-------------|
| `search` | Discover and rank papers by relevance to your profile |
| `report <paper_id>` | Generate a detailed report for a specific paper |
| `run` | Full pipeline: search + report generation |
| `config show` | Show current configuration |
| `config init` | Create a default `config/settings.toml` |

### Options

| Option | Description |
|--------|-------------|
| `--version`, `-V` | Show version |
| `--verbose`, `-v` | Enable debug logging |

### `ppagent search`

```bash
ppagent search [--date YYYY-MM-DD] [--limit N] [--profile path/to/profile.md]
```

### `ppagent report`

```bash
ppagent report <paper_id_or_url> [--output path/to/dir]
```

Accepts an arXiv ID (`2506.12345`) or full URL (`https://arxiv.org/abs/2506.12345`).

### `ppagent run`

```bash
ppagent run [--date YYYY-MM-DD] [--limit N] [--schedule]
```

`--schedule` starts auto-fetch mode, running the pipeline at the configured cron time.

---

## Configuration

All settings live in `config/settings.toml`. Copy `config/settings.example.toml` as a starting point.

```toml
[llm]
base_url = "https://api.openai.com/v1"   # any OpenAI-compatible endpoint
api_key = "sk-..."                         # or set PPA_LLM_API_KEY
model = "gpt-4o"

[search]
default_limit = 50
sort = "trending"
profile_path = "config/profile.md"
relevance_threshold = 0.6
max_reports_per_run = 5

[report]
output_dir = "output"
formats = ["md", "html"]
download_pdf = true
language = "English"  # e.g. "Chinese", "Japanese", "French"

[scheduler]
enabled = false
cron_hour = 8
cron_minute = 0
timezone = "Asia/Shanghai"

[publish.wechat]
enabled = false
appid = ""
secret = ""

[publish.notion]
enabled = false
api_key = ""
database_id = ""

[publish.blog]
enabled = false
webhook_url = ""
```

**Environment variable overrides** (take precedence over TOML):

| Variable | Overrides |
|----------|-----------|
| `PPA_LLM_API_KEY` | `llm.api_key` |
| `PPA_LLM_BASE_URL` | `llm.base_url` |
| `PPA_LLM_MODEL` | `llm.model` |
| `PPA_NOTION_API_KEY` | `publish.notion.api_key` |
| `PPA_WECHAT_APPID` | `publish.wechat.appid` |
| `PPA_WECHAT_SECRET` | `publish.wechat.secret` |
| `PPA_BLOG_API_KEY` | `publish.blog.api_key` |

---

## Research Profile

Edit `config/profile.md` to define your interests. The Searcher agent uses this to score papers.

```markdown
# Research Profile

## Research Areas
- Large Language Models
- Multimodal AI
- AI Agents and Tool Use

## Keywords
transformer, fine-tuning, RLHF, vision-language, reasoning

## Disinterests (exclude papers about)
- Pure theoretical mathematics
- Hardware-only contributions

## Preferred Venues
NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR
```

---

## Output Structure

Reports are saved to `output/{date}/{paper_id}/`:

```
output/
└── 2026-06-10/
    └── 2506.12345/
        ├── report.md        # Markdown report
        ├── report.html      # Styled HTML report
        └── metadata.json    # Structured metadata
```

---

## Architecture

```
User Profile (.md)
       │
       ▼
┌──────────────┐    hf papers ls
│  Searcher    │◄──────────────── HuggingFace CLI
│  Agent       │
└──────┬───────┘
       │ filtered papers
       ▼
┌──────────────────────────────┐
│  Paper Pipeline (per paper)  │
│                              │
│  ┌─────────┐  ┌──────────┐  │
│  │ Writer  │  │ Finder   │  │  ◄── parallel
│  └────┬────┘  └─────┬────┘  │
│       │              │       │
│       ▼              ▼       │
│  ┌──────────────┐            │
│  │ Criticizer   │            │  ◄── needs Writer output
│  └──────┬───────┘            │
│         │                    │
│         ▼                    │
│  ┌──────────────┐            │
│  │ Assembler    │            │  ◄── deterministic, no LLM
│  └──────┬───────┘            │
└─────────┼────────────────────┘
          │
          ▼
  report.md + report.html
```

**Agents**:
- **Searcher** — scores papers against your profile using structured LLM output
- **Writer** — extracts metadata, benchmarks, TL;DR, method, evaluation
- **Finder** — searches related works via tool-calling (`hf papers search`)
- **Criticizer** — audits the paper for limitations and weaknesses
- **Assembler** — renders all sections into Markdown + HTML via Jinja2 templates

---

## Custom Agents

Place a Python file in `~/.config/ppagent/agents/`:

```python
# ~/.config/ppagent/agents/my_summarizer.py
from ppagent.agents.base import AgentBase, register_agent
from ppagent.models import AgentResult

@register_agent
class MySummarizer(AgentBase):
    name = "my_summarizer"
    description = "Produces bullet-point summaries"

    def run(self, **kwargs) -> AgentResult:
        content = kwargs.get("content")
        # your logic here
        return AgentResult(agent_name=self.name, success=True, data={"summary": "..."})
```

Reference it in `settings.toml`:

```toml
[report]
custom_agents = ["my_summarizer"]
```

## Custom Publishers

Same pattern — drop a file in `~/.config/ppagent/publishers/`:

```python
from ppagent.publishers.base import PublisherBase, register_publisher
from ppagent.models import PaperReport

@register_publisher
class MyPublisher(PublisherBase):
    name = "my_publisher"

    def publish(self, report: PaperReport, *, md_content: str, html_content: str) -> bool:
        # push to your platform
        return True
```

---

## Project Structure

```
daily-paper-reading/
├── main.py                          # Entry point
├── pyproject.toml                   # Dependencies + build config
├── config/
│   ├── settings.toml                # Your config (gitignored)
│   ├── settings.example.toml        # Template
│   └── profile.md                   # Your research profile
├── templates/
│   ├── report.md.jinja2             # Markdown report template
│   └── report.html.jinja2           # HTML report template
├── src/ppagent/
│   ├── cli.py                       # Typer CLI
│   ├── config.py                    # Config loading
│   ├── models.py                    # Pydantic data models
│   ├── llm.py                       # OpenAI-compatible LLM client
│   ├── hf.py                        # HuggingFace CLI wrapper
│   ├── pdf.py                       # PDF download + extraction
│   ├── pipeline.py                  # Multi-agent orchestration
│   ├── scheduler.py                 # Auto-fetch mode
│   ├── storage.py                   # Report file management
│   ├── agents/                      # Searcher, Writer, Finder, Criticizer, Assembler
│   └── publishers/                  # WeChat, Notion, Blog
└── output/                          # Generated reports (gitignored)
```

---

## License

MIT
