"""Full-text search over the dashboard's own text (sidebar Search box).

The dashboard script runs top to bottom under Streamlit, so its text cannot
be collected from a rendered page: only the selected Welcome mode, region,
etc. would be seen. Instead the index is built from the source: every string
the app can display (labels, captions, headings, documentation dicts, help
tooltips), located by the tab and section it belongs to.

Keep this file free of streamlit imports (it is unit-tested).
"""
from __future__ import annotations

import ast
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD = re.compile(r"[*_`>#|]+")
# Strings that are code, not text: keys, file names, colormaps, format specs.
_CODE_LIKE = re.compile(r"^[\w./{}:%\-\[\]=]+$")


def _normalize_map(s: str) -> tuple[str, list[int]]:
    """Normalized text and, for each of its characters, the index in `s`."""
    out, idx = [], []
    for i, ch in enumerate(s):
        for c in unicodedata.normalize("NFKD", ch.lower()):
            if not unicodedata.combining(c):
                out.append(c)
                idx.append(i)
    return "".join(out), idx


def normalize(s: str) -> str:
    """Lower case with accents removed, so 'periode' also finds 'période'."""
    return _normalize_map(s)[0]


def clean(s: str) -> str:
    """Display text: markdown links → their text, tags and markup removed."""
    s = _LINK.sub(r"\1", s)
    s = _TAG.sub(" ", s)
    s = _MD.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_text(s: str) -> bool:
    s = s.strip()
    if len(s) < 3 or not re.search(r"[A-Za-z]", s) or re.fullmatch(r"#?[0-9A-Fa-f]{6}", s):
        return False
    if " " in s:
        return not s.startswith(("http://", "https://")) or " " in s.split("://", 1)[1]
    return not _CODE_LIKE.match(s) or (s[0].isupper() and "_" not in s and "." not in s)


def build_index(source: str, tab_names: dict[str, str],
                doc_locations: dict[str, tuple[str, str]],
                default_location: tuple[str, str] = ("Page", "")) -> list[dict]:
    """Index every displayable string of a Streamlit script.

    tab_names: `with <name>:` block → tab label (e.g. {"tab_maps": "Spatial maps"});
        `with st.sidebar:` is labelled "Sidebar".
    doc_locations: module-level variable → (tab, section) for its strings
        (e.g. {"METHOD_DOCS": ("Welcome", "Methods")}). Other module-level
        variables take the location where they are first used inside a tab.
    Returns entries {"tab", "section", "text", "norm", "line"}.
    """
    tree = ast.parse(source)
    blocks, assigns, headers = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                name = ast.unparse(item.context_expr)
                if name in tab_names:
                    blocks.append((node.lineno, node.end_lineno, tab_names[name]))
                elif name == "st.sidebar":
                    blocks.append((node.lineno, node.end_lineno, "Sidebar"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                if node.func.attr in ("header", "subheader"):
                    headers.append((node.lineno, clean(a0.value)))
                elif node.func.attr == "markdown" and a0.value.lstrip().startswith("#"):
                    headers.append((node.lineno, clean(a0.value.strip().split("\n")[0])))
        elif (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
              and node.left.id.endswith("_mode") and node.comparators
              and isinstance(node.comparators[0], ast.Constant)):
            headers.append((node.lineno, str(node.comparators[0].value)))
    headers.sort()
    first_use: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if any(b[0] <= node.lineno <= b[1] for b in blocks):
                first_use[node.id] = min(first_use.get(node.id, node.lineno), node.lineno)
    var_assigns = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id in doc_locations:
                assigns.append((node.lineno, node.end_lineno, doc_locations[t.id]))
            elif isinstance(t, ast.Name) and t.id in first_use:
                var_assigns.append((node.lineno, node.end_lineno, first_use[t.id]))

    # Strings that are never shown: docstrings, widget keys, dict keys of docs.
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
        elif isinstance(node, ast.keyword) and node.arg in ("key", "format", "width", "unsafe_allow_html"):
            for sub in ast.walk(node.value):
                skip.add(id(sub))
        elif isinstance(node, ast.Dict):
            for k in node.keys:
                if k is not None:
                    skip.add(id(k))
        elif isinstance(node, ast.JoinedStr):
            for sub in node.values:
                skip.add(id(sub))
        elif isinstance(node, ast.Subscript):
            skip.add(id(node.slice))

    def locate(line: int) -> tuple[str, str]:
        inner = [b for b in blocks if b[0] <= line <= b[1]]
        if inner:
            start, _, tab = max(inner, key=lambda b: b[0])
            sec = [h for ln, h in headers if start <= ln <= line]
            return tab, (sec[-1] if sec else "")
        for start, end, loc in assigns:
            if start <= line <= end:
                return loc
        for start, end, use in var_assigns:
            if start <= line <= end:
                return locate(use)
        return default_location

    entries = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            raw = node.value
        elif isinstance(node, ast.JoinedStr):
            raw = " … ".join(v.value for v in node.values
                             if isinstance(v, ast.Constant) and isinstance(v.value, str))
        else:
            continue
        if not _is_text(raw):
            continue
        text = clean(raw)
        if not text:
            continue
        tab, section = locate(node.lineno)
        entries.append({"tab": tab, "section": section, "text": text,
                        "norm": normalize(text), "line": node.lineno})
    entries.sort(key=lambda e: e["line"])
    return entries


def search(entries: list[dict], query: str, max_locations: int = 25) -> list[dict]:
    """Entries containing every word of `query` (matched from a word start,
    so "hann" finds "Hann-tapered" but not "channels"), one per (tab, section).

    Returns [{"tab", "section", "snippet", "count"}] in app order; the snippet
    marks the first word in bold.
    """
    words = [w for w in normalize(query).split() if w]
    if not words:
        return []
    pats = [re.compile(r"(?<!\w)" + re.escape(w)) for w in words]
    out: dict[tuple[str, str], dict] = {}
    for e in entries:
        if not all(p.search(e["norm"]) for p in pats):
            continue
        key = (e["tab"], e["section"])
        if key in out:
            out[key]["count"] += 1
            continue
        norm, pos = _normalize_map(e["text"])
        j = pats[0].search(norm).start()
        i, k = pos[j], pos[j + len(words[0]) - 1] + 1     # match span in the original text
        t = e["text"]
        a, b = max(0, i - 60), min(len(t), k + 80)
        snippet = (("…" if a else "") + t[a:i] + "**" + t[i:k] + "**" + t[k:b]
                   + ("…" if b < len(t) else ""))
        out[key] = {"tab": e["tab"], "section": e["section"], "snippet": snippet, "count": 1}
        if len(out) >= max_locations:
            break
    return list(out.values())
