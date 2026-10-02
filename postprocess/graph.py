import argparse
import html
import json
from pathlib import Path

COLORS = {
    "pending": "#e2e8f0",
    "cached": "#bbf7d0",
    "done": "#bbf7d0",
    "running": "#bfdbfe",
    "failed": "#fecaca",
    "blocked": "#fed7aa",
}


def render(graph: dict) -> str:
    if graph.get("schema") != "contract-graph-v1":
        raise ValueError("expected a contract-graph-v1 manifest")
    nodes = {node["id"]: node for node in graph["nodes"]}
    parents = {key: [] for key in nodes}
    for edge in graph["edges"]:
        if edge["consumer"] not in nodes or edge["dependency"] not in nodes:
            raise ValueError("edge references an unknown node")
        parents[edge["consumer"]].append(edge["dependency"])
    ranks = {}
    while len(ranks) < len(nodes):
        ready = [
            key
            for key in nodes
            if key not in ranks and all(p in ranks for p in parents[key])
        ]
        if not ready:
            raise ValueError("graph contains a cycle")
        for key in ready:
            ranks[key] = max((ranks[p] + 1 for p in parents[key]), default=0)
    columns = {}
    positions = {}
    for key, rank in ranks.items():
        row = columns.get(rank, 0)
        positions[key] = (40 + rank * 295, 90 + row * 110)
        columns[rank] = row + 1
    width = max(700, 80 + len(columns) * 295)
    height = max(240, 120 + max(columns.values(), default=0) * 110)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<g font-family="system-ui,sans-serif" fill="#0f172a">',
        '<text x="40" y="32" font-size="20">Contract dependency graph</text>',
        '<text x="40" y="55" font-size="12">Arrows: dependency → consumer. Outlined nodes are requested roots. State is printed in each node.</text>',
    ]
    for edge in graph["edges"]:
        x, y = positions[edge["dependency"]]
        tx, ty = positions[edge["consumer"]]
        parts.append(
            f'<path d="M {x + 250} {y + 35} C {x + 270} {y + 35}, {tx - 20} {ty + 35}, {tx} {ty + 35}" fill="none" stroke="#64748b" marker-end="url(#arrow)"/>'
        )
    for key, node in nodes.items():
        x, y = positions[key]
        state = node["state"]
        color = COLORS.get(state, "#e2e8f0")
        border = 3 if key in graph["roots"] else 1
        parts.extend(
            [
                f"<g><title>{html.escape(node['label'] + ': ' + node.get('reason', ''))}</title>",
                f'<rect x="{x}" y="{y}" width="250" height="70" rx="8" fill="{color}" stroke="#475569" stroke-width="{border}"/>',
                f'<text x="{x + 12}" y="{y + 27}" font-size="12">{html.escape(node["label"][:34])}</text>',
                f'<text x="{x + 12}" y="{y + 49}" font-size="11">{html.escape(state)} · {html.escape(key[:8])}</text></g>',
            ]
        )
    return "\n".join([*parts, "</g></svg>"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    svg = render(json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)


if __name__ == "__main__":
    main()
