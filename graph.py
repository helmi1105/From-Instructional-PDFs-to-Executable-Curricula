import argparse
from pathlib import Path

from graphviz import Source


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one ECG DOT file into PNG.")
    parser.add_argument("--dot", required=True, help="Path to input .dot file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--name", default="ecg_graph", help="Output filename without extension")
    args = parser.parse_args()

    dot_path = Path(args.dot)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dot_text = dot_path.read_text(encoding="utf-8")
    src = Source(dot_text)
    src.render(filename=args.name, directory=str(out_dir), format="png", cleanup=True)
    print("Graph saved to:", out_dir / f"{args.name}.png")


if __name__ == "__main__":
    main()
