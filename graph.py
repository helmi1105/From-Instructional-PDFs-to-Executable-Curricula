import os
from graphviz import Source

# Add Graphviz binary folder to PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz-14.1.4-win64\bin"

dot_path = r"C:\Users\hbaaz\OneDrive\Bureau\PHD\ecg\project\outputs\clean_ecg\ecg.dot"
out_dir = r"C:\Users\hbaaz\OneDrive\Bureau\PHD\ecg\project\outputs\clean_ecg\graphs"

with open(dot_path, "r", encoding="utf-8") as f:
    dot_text = f.read()

src = Source(dot_text)

# Render as PNG
src.render(
    filename="ecg_graph",
    directory=out_dir,
    format="png",
    cleanup=True
)

print("Graph saved to:", os.path.join(out_dir, "ecg_graph.png"))