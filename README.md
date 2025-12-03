# jigsaw_puzzle_matching

A robust jigsaw puzzle reconstruction system using **LAB color**, **gradient features**, **best-buddy matching**, and **iterative refinement**.
Supports **visual** and **non-visual** execution modes.

---

## 📸 Overview

This solver takes an input image, cuts it into an **N×N grid**, extracts enhanced border descriptors, computes edge compatibility scores, then reconstructs the puzzle using:

* LAB + Gradient Magnitude + Gradient Direction + Laplacian edges
* Pairwise border compatibility
* Mutual best-buddies
* Greedy placement
* Segmentation
* Multi-seed shifter optimization

---

## ⚙️ Installation

```bash
pip install opencv-python numpy matplotlib
```

---

## ▶️ Usage

### **Basic (no visualization)**

```bash
python puzzle_solver_edges_refactored.py --image input.jpg --grid 4
```

### **Enable visualization**

```bash
python puzzle_solver.py --image input.jpg --grid 4 --vis
```

### **Force no visualization**

```bash
python puzzle_solver.py --image input.jpg --grid 4 --no-vis
```

---

## 🧰 Command-Line Options

| Argument          | Type | Description                              |
| ----------------- | ---- | ---------------------------------------- |
| `--image PATH`    | str  | Path to input image                      |
| `--grid N`        | int  | Grid size (e.g., 4 for 4×4)              |
| `--vis`           | flag | Enables detailed visualization           |
| `--no-vis`        | flag | Disables all visualization               |
| `--strip-width W` | int  | Border thickness for feature extraction  |
| `--iters K`       | int  | Number of shifter improvement iterations |

---

## 🔍 What Visualization Shows (`--vis`)

* Puzzle piece grid
* LAB + gradient border channels
* Normalized strips
* Compatibility heatmaps
* Segment grouping
* Final assembled puzzle preview

---

## 📦 Example

```bash
python puzzle_solver.py --image sample.jpg --grid 2 --vis
```

---

## 📂 Output

* Final piece placement array
* Best-buddy score (quality metric)
* Segment groupings
* Optional visual previews

---

## 📝 Notes

* Rotation is **not** handled; pieces must be upright.
* Works best with detailed images (texture & edges).
* The solver is deterministic per seed but uses multi-seed shifter passes to improve quality.

