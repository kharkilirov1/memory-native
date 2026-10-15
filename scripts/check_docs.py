"""Check local file links in the maintained documentation entry points.

Historical report bodies may cite missing artifacts explicitly; their existence is
tracked by the artifact manifest rather than this navigation check. Remote URLs
and heading fragments are outside this check's scope.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        self.urls.extend(value for key, value in attrs if key in {"href", "src"} and value)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pages = [root / name for name in ("README.md", "CONTRIBUTING.md", "SECURITY.md", "CLAUDE.md")]
    pages += sorted((root / "docs").glob("*.md"))
    pages += [root / folder / "README.md" for folder in ("scripts", "examples", "notebooks", "research")]
    pages += sorted((root / "research").glob("*/README.md"))
    errors = []
    for page in pages:
        text = re.sub(r"```.*?```", "", page.read_text(), flags=re.S)
        urls = re.findall(r"\]\(([^\s)]+)\)", text)
        html = HTMLLinks()
        html.feed(text)
        for url in urls + html.urls:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            target = page.parent / unquote(parts.path)
            if not target.exists():
                errors.append(f"{page.relative_to(root)}: missing {url}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Local file links resolve in {len(pages)} maintained documentation pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
