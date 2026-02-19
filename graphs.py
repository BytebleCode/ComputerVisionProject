"""
graphs.py - Generate training graphs for a YOLOv6 project.

Parses the training results CSV and produces:
  - Loss curves (train box/obj/cls loss, val box/obj/cls loss)
  - mAP curves (mAP@0.5, mAP@0.5:0.95)
  - Learning rate curve

Usage:
  python graphs.py --project cars                  # view graphs
  python graphs.py --project cars --save-only      # save PNGs only
  python graphs.py --project cars --compare faces   # overlay two projects
"""

import argparse
import os
import sys
import glob

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(ROOT, "projects")

# Colors for plotting
COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4",
    "#795548", "#607D8B", "#E91E63", "#3F51B5",
]


def find_results_csv(project_dir):
    """Find the most recent results CSV in a project's runs."""
    runs_dir = os.path.join(project_dir, "runs")
    # YOLOv6 may write results as CSV or we parse from logs
    patterns = ["**/results.csv", "**/results.txt"]
    for pattern in patterns:
        matches = glob.glob(os.path.join(runs_dir, pattern), recursive=True)
        if matches:
            return max(matches, key=os.path.getmtime)
    return None


def load_results(csv_path):
    """Load and clean the results CSV."""
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df


def find_columns(df, keywords):
    """Find DataFrame columns that contain any of the given keywords."""
    matches = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw.lower() in col_lower for kw in keywords):
            matches.append(col)
    return matches


def plot_loss(ax, df, label_prefix=""):
    """Plot loss curves on the given axes."""
    epoch_col = find_columns(df, ["epoch"])
    epoch = df[epoch_col[0]] if epoch_col else df.index

    # Find loss columns
    train_loss_cols = find_columns(df, ["train/box", "train/obj", "train/cls",
                                         "box_loss", "obj_loss", "cls_loss"])
    val_loss_cols = find_columns(df, ["val/box", "val/obj", "val/cls"])

    # If we can't find specific loss columns, look for generic loss
    if not train_loss_cols:
        train_loss_cols = find_columns(df, ["loss"])

    color_idx = 0
    for col in train_loss_cols:
        label = f"{label_prefix}{col}" if label_prefix else col
        ax.plot(epoch, df[col], label=label, color=COLORS[color_idx % len(COLORS)])
        color_idx += 1

    for col in val_loss_cols:
        label = f"{label_prefix}{col}" if label_prefix else col
        ax.plot(epoch, df[col], label=label, color=COLORS[color_idx % len(COLORS)],
                linestyle="--")
        color_idx += 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_map(ax, df, label_prefix=""):
    """Plot mAP curves on the given axes."""
    epoch_col = find_columns(df, ["epoch"])
    epoch = df[epoch_col[0]] if epoch_col else df.index

    map_cols = find_columns(df, ["mAP", "map"])

    color_idx = 0
    for col in map_cols:
        label = f"{label_prefix}{col}" if label_prefix else col
        ax.plot(epoch, df[col], label=label, color=COLORS[color_idx % len(COLORS)],
                linewidth=2)
        color_idx += 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("mAP Over Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_lr(ax, df, label_prefix=""):
    """Plot learning rate curve on the given axes."""
    epoch_col = find_columns(df, ["epoch"])
    epoch = df[epoch_col[0]] if epoch_col else df.index

    lr_cols = find_columns(df, ["lr"])

    color_idx = 4  # offset to get different colors
    for col in lr_cols:
        label = f"{label_prefix}{col}" if label_prefix else col
        ax.plot(epoch, df[col], label=label, color=COLORS[color_idx % len(COLORS)])
        color_idx += 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def generate_graphs(project_name, save_only=False, compare_name=None):
    project_dir = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.isdir(project_dir):
        print(f"[ERROR] Project '{project_name}' not found.")
        sys.exit(1)

    csv_path = find_results_csv(project_dir)
    if not csv_path:
        print(f"[ERROR] No training results found for '{project_name}'.")
        print("  Train the model first: python train.py --project " + project_name)
        sys.exit(1)

    print(f"[GRAPHS] Loading results from: {csv_path}")
    df = load_results(csv_path)
    print(f"  Found {len(df)} epochs of data, columns: {list(df.columns)}")

    # Load comparison data if requested
    compare_df = None
    if compare_name:
        compare_dir = os.path.join(PROJECTS_DIR, compare_name)
        compare_csv = find_results_csv(compare_dir)
        if compare_csv:
            compare_df = load_results(compare_csv)
            print(f"  Comparison: {compare_name} ({len(compare_df)} epochs)")
        else:
            print(f"  [WARN] No results found for comparison project '{compare_name}'")

    if save_only:
        matplotlib.use("Agg")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Results: {project_name}" +
                 (f" vs {compare_name}" if compare_name else ""),
                 fontsize=14, fontweight="bold")

    prefix = f"{project_name}: " if compare_name else ""

    # Plot main project
    plot_loss(axes[0], df, label_prefix=prefix)
    plot_map(axes[1], df, label_prefix=prefix)
    plot_lr(axes[2], df, label_prefix=prefix)

    # Overlay comparison project
    if compare_df is not None:
        compare_prefix = f"{compare_name}: "
        plot_loss(axes[0], compare_df, label_prefix=compare_prefix)
        plot_map(axes[1], compare_df, label_prefix=compare_prefix)
        plot_lr(axes[2], compare_df, label_prefix=compare_prefix)

    plt.tight_layout()

    # Save graphs
    graphs_dir = os.path.join(project_dir, "runs", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    combined_path = os.path.join(graphs_dir, "training_summary.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Saved: {combined_path}")

    # Also save individual plots
    for idx, name in enumerate(["loss", "mAP", "lr"]):
        individual_fig, individual_ax = plt.subplots(1, 1, figsize=(8, 5))
        if name == "loss":
            plot_loss(individual_ax, df, label_prefix=prefix)
            if compare_df is not None:
                plot_loss(individual_ax, compare_df, label_prefix=f"{compare_name}: ")
        elif name == "mAP":
            plot_map(individual_ax, df, label_prefix=prefix)
            if compare_df is not None:
                plot_map(individual_ax, compare_df, label_prefix=f"{compare_name}: ")
        elif name == "lr":
            plot_lr(individual_ax, df, label_prefix=prefix)
            if compare_df is not None:
                plot_lr(individual_ax, compare_df, label_prefix=f"{compare_name}: ")

        individual_path = os.path.join(graphs_dir, f"{name}.png")
        individual_fig.savefig(individual_path, dpi=150, bbox_inches="tight")
        plt.close(individual_fig)
        print(f"  [OK] Saved: {individual_path}")

    if not save_only:
        print()
        print("  Showing graphs (close the window to continue)...")
        plt.show()
    else:
        plt.close(fig)

    print("[OK] Graphs complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate YOLOv6 training graphs")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--save-only", action="store_true",
                        help="Save PNGs without opening display window")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare with another project (overlay on same plots)")
    args = parser.parse_args()

    generate_graphs(args.project, save_only=args.save_only, compare_name=args.compare)


if __name__ == "__main__":
    main()
