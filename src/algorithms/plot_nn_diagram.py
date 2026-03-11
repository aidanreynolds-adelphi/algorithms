#!/usr/bin/env python3
"""
Generate a neural network architecture diagram using PlotNeuralNet (TikZ/LaTeX).

Uses the actual MLP architecture from obesity_nn (input size, hidden sizes, classes).
Clones PlotNeuralNet into repo root if needed. Outputs .tex and .png to figures/.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path

# Type alias for (label, size) per layer
_LayerSpec = tuple[str, int]

# Repo root (this file lives in src/algorithms/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Where to find or clone PlotNeuralNet (under repo root)
PLOTNN_DIR = _REPO_ROOT / "PlotNeuralNet"
PLOTNN_URL = "https://github.com/HarisIqbal88/PlotNeuralNet.git"


def ensure_plotneuralnet() -> Path:
    """Clone PlotNeuralNet into repo root (PlotNeuralNet/) if not present."""
    if PLOTNN_DIR.is_dir() and (PLOTNN_DIR / "pycore").is_dir():
        return PLOTNN_DIR
    print("PlotNeuralNet not found; cloning into PlotNeuralNet/ ...", file=sys.stderr)
    subprocess.run(
        ["git", "clone", "--depth", "1", PLOTNN_URL, str(PLOTNN_DIR)],
        check=True,
        cwd=str(_REPO_ROOT),
    )
    return PLOTNN_DIR


def pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 200) -> bool:
    """Convert a single-page PDF to PNG. Tries pdftoppm, then ImageMagick. Returns True if done."""
    # pdftoppm (poppler): outputs base-1.png for single page
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        try:
            stem = pdf_path.with_suffix("").name
            out_stem = str(pdf_path.parent / stem)
            subprocess.run(
                [pdftoppm, "-png", "-r", str(dpi), str(pdf_path), out_stem],
                check=True,
                capture_output=True,
            )
            from_pdftoppm = pdf_path.parent / f"{stem}-1.png"
            if from_pdftoppm.exists():
                from_pdftoppm.rename(png_path)
                return True
        except subprocess.CalledProcessError:
            pass
    # ImageMagick convert
    convert = shutil.which("convert")
    if convert:
        try:
            subprocess.run(
                [convert, "-density", str(dpi), str(pdf_path), str(png_path)],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            pass
    return False


def draw_mlp_matplotlib(png_path: Path, layers: list[_LayerSpec]) -> None:
    """Draw the obesity MLP with the given layer specs and save as PNG (tight layout)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n_layers = len(layers)
    # Tighter horizontal spacing (hidden layers have short labels)
    x_spacing = 1.65
    x_pos = [i * x_spacing for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(9, 5))
    # Extra right margin so output label (class names) is not clipped
    ax.set_xlim(-0.5, (n_layers - 1) * x_spacing + 1.4)
    ax.set_ylim(-1.82, 1.15)  # minimal room below for input/output labels
    ax.set_aspect("equal")
    ax.axis("off")

    for i, (label, size) in enumerate(layers):
        x = x_pos[i]
        n_show = min(size, 12)
        y_vals = [(-0.85 + 1.7 * j / (n_show + 1)) for j in range(1, n_show + 1)]
        for y in y_vals:
            circle = Circle((x, y), 0.06, color="steelblue", ec="navy", zorder=2)
            ax.add_patch(circle)
        if size > n_show:
            ax.text(x, -0.9, "...", ha="center", fontsize=9)
        is_input = i == 0
        is_output = i == n_layers - 1
        if is_input or is_output:
            ax.text(
                x, -1.0, label, ha="center", va="top", fontsize=6, fontweight="bold"
            )
        else:
            ax.text(x, 1.02, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i in range(n_layers - 1):
        ax.annotate(
            "",
            xy=(x_pos[i + 1] - 0.28, 0),
            xytext=(x_pos[i] + 0.28, 0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _wrap_list(items: list[str], per_line: int = 5) -> str:
    """Format a list as comma-separated lines (for diagram labels)."""
    lines = []
    for i in range(0, len(items), per_line):
        lines.append(", ".join(items[i : i + per_line]))
    return "\n".join(lines)


def _build_layer_specs(
    n_input: int,
    hidden_sizes: tuple[int, ...],
    n_classes: int,
    class_names: list[str],
    feature_names: list[str],
) -> list[_LayerSpec]:
    """Build (label, size) list for input, hidden, and output layers (full labels)."""
    in_label = f"Input\n({n_input})"
    if feature_names:
        in_label += "\n" + _wrap_list(feature_names, per_line=5)
    layers: list[_LayerSpec] = [(in_label, n_input)]
    for h in hidden_sizes:
        layers.append((str(h), h))
    out_label = f"Output\n({n_classes} classes)"
    if class_names:
        out_label += "\n" + _wrap_list(class_names, per_line=4)
    layers.append((out_label, n_classes))
    return layers


def main() -> None:
    from algorithms.obesity_nn import get_mlp_architecture

    n_input, hidden_sizes, n_classes, class_names, feature_names = get_mlp_architecture()
    layer_specs = _build_layer_specs(
        n_input, hidden_sizes, n_classes, class_names, feature_names
    )

    plotnn = ensure_plotneuralnet()
    if str(plotnn) not in sys.path:
        sys.path.insert(0, str(plotnn))

    from pycore import tikzeng  # type: ignore[import-not-found]

    projectpath = str(plotnn.resolve())
    input_caption = (
        f"Input ({n_input}): " + ", ".join(feature_names)
        if feature_names
        else f"Input ({n_input})"
    )
    # Input layer
    arch = [
        tikzeng.to_head(projectpath),
        tikzeng.to_cor(),
        tikzeng.to_begin(),
        tikzeng.to_Conv(
            name="input",
            s_filer=str(n_input),
            n_filer=str(n_input),
            offset="(0,0,0)",
            to="(0,0,0)",
            width=1,
            height=8,
            depth=8,
            caption=input_caption,
        ),
    ]
    # Hidden layers
    prev = "input"
    for i, size in enumerate(hidden_sizes):
        name = f"hidden{i + 1}"
        arch.append(
            tikzeng.to_Conv(
                name=name,
                s_filer=str(size),
                n_filer=str(size),
                offset="(2,0,0)",
                to=f"({prev}-east)",
                width=2,
                height=10 if i == 0 else 8,
                depth=10 if i == 0 else 8,
                caption=str(size),
            )
        )
        arch.append(tikzeng.to_connection(prev, name))
        prev = name
    # Output layer
    arch.append(
        tikzeng.to_SoftMax(
            name="output",
            s_filer=n_classes,
            offset="(2,0,0)",
            to=f"({prev}-east)",
            width=1.5,
            height=4,
            depth=12,
            opacity=0.8,
            caption=f"{n_classes} classes",
        )
    )
    arch.append(tikzeng.to_connection(prev, "output"))
    arch.append(tikzeng.to_end())

    out_dir = _REPO_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "obesity_mlp.tex"
    with io.StringIO() as buf:
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            tikzeng.to_generate(arch, str(tex_path))
        finally:
            sys.stdout = old_stdout
    print(f"Wrote {tex_path}")

    # Compile to PDF then PNG, or fall back to matplotlib PNG
    pdf_path = out_dir / "obesity_mlp.pdf"
    png_path = out_dir / "obesity_mlp.png"
    png_done = False
    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                str(out_dir),
                str(tex_path),
            ],
            check=True,
            cwd=str(out_dir),
            capture_output=True,
        )
        if pdf_path.exists():
            print(f"Compiled to {pdf_path}")
            png_done = pdf_to_png(pdf_path, png_path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    if not png_done:
        draw_mlp_matplotlib(png_path, layer_specs)
        print(f"Wrote {png_path} (matplotlib fallback)")


if __name__ == "__main__":
    main()
