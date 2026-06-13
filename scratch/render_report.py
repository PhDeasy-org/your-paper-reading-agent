import json
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Mock render_markdown_with_math
def render_markdown_with_math(text: str) -> str:
    import markdown as md_lib
    import re
    placeholders = {}
    block_pattern = re.compile(r'(\$\$.*?\$\$|\\\[.*?\\\])', re.DOTALL)
    inline_pattern = re.compile(r'(\$(?!\s)[^\$\n]+?(?<!\s)\$|\\\(.*?\\\))')
    temp_text = text or ""
    def replace_match(match: re.Match) -> str:
        placeholder = f"<!--MATH_PLACEHOLDER_{len(placeholders)}-->"
        placeholders[placeholder] = match.group(0)
        return placeholder
    temp_text = block_pattern.sub(replace_match, temp_text)
    temp_text = inline_pattern.sub(replace_match, temp_text)
    html = md_lib.markdown(temp_text, extensions=["tables", "fenced_code"])
    for placeholder, original in placeholders.items():
        html = html.replace(placeholder, original)
    return html

class DotDict(dict):
    """Allow dot notation for dict keys (e.g. obj.title)"""
    def __getattr__(self, name):
        if name in self:
            val = self[name]
            if isinstance(val, dict):
                return DotDict(val)
            if isinstance(val, list):
                return [DotDict(item) if isinstance(item, dict) else item for item in val]
            return val
        raise AttributeError(f"No attribute {name}")

def main():
    workspace = Path("/Users/sn/ws/daily-paper-reading")
    metadata_path = workspace / "output/2026-06-13/2606.01075/metadata.json"
    if not metadata_path.exists():
        metadata_path = workspace / "output/2026-06-12/2606.01075/metadata.json"
        
    print(f"Reading from {metadata_path}")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    
    # Parse generated_at
    gen_at = datetime.fromisoformat(data["generated_at"])
    
    # Create the template context
    paper = DotDict(data["paper"])
    # Convert published_at to datetime object if it is string
    if paper.get("published_at"):
        paper.published_at = datetime.fromisoformat(paper.published_at)
    else:
        paper.published_at = None
        
    tldr = DotDict(data["tldr"])
    benchmarks = DotDict(data["benchmarks"])
    previous_works = DotDict(data["previous_works"])
    method = DotDict(data["method"])
    evaluation = DotDict(data["evaluation"])
    critique = DotDict(data["critique"])
    
    # Extract affiliations and keywords from metadata/writer_data
    # In the real assembler, these are extracted from writer_data
    # Let's mock them or extract them from metadata.content
    meta_content = data["metadata"]["content"]
    # Parse affiliations and keywords from markdown table in meta_content
    affiliations = []
    keywords = []
    for line in meta_content.split('\n'):
        if "Affiliations" in line:
            parts = line.split('|')
            if len(parts) >= 3:
                affiliations = [x.strip() for x in parts[2].split(',')]
        if "Keywords" in line:
            parts = line.split('|')
            if len(parts) >= 3:
                keywords = [x.strip() for x in parts[2].split(',')]
                
    related_works = [DotDict(rw) for rw in data.get("related_works", [])]
    
    cost_report = data.get("cost_report")
    if not cost_report:
        # Mock a cost report for visual design testing
        cost_report = {
            "provider": "open_router",
            "model": "deepseek/deepseek-r1",
            "input_cost": 0.00014,
            "output_cost": 0.00085,
            "total_cost": 0.00099
        }
    usage = data.get("usage")
    if not usage:
        usage = {
            "prompt_tokens": 12500,
            "completion_tokens": 3400,
            "total_tokens": 15900
        }
        
    context = {
        "paper": paper,
        "metadata": DotDict(data["metadata"]),
        "benchmarks": benchmarks,
        "tldr": tldr,
        "previous_works": previous_works,
        "method": method,
        "evaluation": evaluation,
        "critique": critique,
        "related_works": related_works,
        "generated_at": gen_at,
        "model_used": data["model_used"],
        "keywords": keywords,
        "affiliations": affiliations,
        "finder_narrative": "This paper is highly relevant to generator-verifier dynamics and closed-loop training. It is closely related to SPIN (Self-Play Fine-Tuning) and AZR (Absolute Zero).",
        "usage": usage,
        "cost_report": cost_report,
    }
    
    templates_dir = workspace / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)
    env.filters["markdown"] = render_markdown_with_math
    
    template = env.get_template("report.html.jinja2")
    rendered_html = template.render(**context)
    
    output_path = workspace / "scratch/test_report.html"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(rendered_html, encoding="utf-8")
    print(f"Rendered test HTML report to {output_path}")

if __name__ == "__main__":
    main()
