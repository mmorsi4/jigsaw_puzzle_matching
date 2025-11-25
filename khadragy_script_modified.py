# puzzle_solver_edges.py (Replaced - full script with exact preprocessing on split pieces)
import cv2
import numpy as np
import os
import argparse
import time
from itertools import permutations

# ----------------------------
# This file replaces the previous canvas file and implements the
# exact preprocessing you requested applied to each split piece.
# Preprocessing per piece (no additions or removals):
#   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   img_clahe = clahe.apply(img_gray)
#   gray = cv2.bilateralFilter(img_clahe, 3, 75, 75)
#   edges = cv2.Canny(gray, 40, 180)
# ----------------------------

# ----------------------------
# Preprocessing: process a list of image groups exactly as given
# ----------------------------

def process_image_groups(image_groups):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    output = []

    for group in image_groups:
        processed_group = []
        for img in group:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_clahe = clahe.apply(img_gray)
            gray = cv2.bilateralFilter(img_clahe, 3, 75, 75)
            edges = cv2.Canny(gray, 40, 180)
            processed_group.append(edges)
        output.append(processed_group)

    return output

# ----------------------------
# Utilities / Visualization
# ----------------------------

def show_images(images, titles=None, figsize=(12,6), grayscale=None):
    import matplotlib.pyplot as plt
    n = len(images)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(n)]
    if grayscale is None:
        grayscale = [False] * n
    elif isinstance(grayscale, bool):
        grayscale = [grayscale] * n
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(1, n, i+1)
        if len(img.shape) == 3:
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_show = img
        if grayscale[i]:
            plt.imshow(img_show, cmap='gray')
        else:
            plt.imshow(img_show)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ----------------------------
# 1) Cut image into grid pieces
# ----------------------------

def cut_into_grid(img, grid_n):
    rows = cols = grid_n
    h, w = img.shape[:2]
    ph, pw = h // rows, w // cols
    pieces = []
    coords = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r*ph, (r+1)*ph
            x0, x1 = c*pw, (c+1)*pw
            piece = img[y0:y1, x0:x1].copy()
            pieces.append(piece)
            coords.append((y0,y1,x0,x1))
    return pieces, coords, (ph, pw)

# ----------------------------
# 2) Extract border strips from preprocessed (edge) pieces
# ----------------------------

def extract_borders(piece, strip_width=16):
    # piece is expected to be a single-channel edge map (H, W)
    p = piece.copy().astype(np.float32)
    h, w = p.shape
    sw = min(strip_width, h//2, w//2)
    top = p[0:sw, :]
    bottom = p[h-sw:h, :]
    left = p[:, 0:sw]
    right = p[:, w-sw:w]
    return {0: top, 1: right, 2: bottom, 3: left}

# ----------------------------
# 3) Border similarity metric
# ----------------------------

def normalize_strip(s):
    if s.size == 0:
        return s.flatten()
    arr = s.flatten().astype(np.float32)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-6:
        return arr - mean
    return (arr - mean) / std


def border_distance(stripA, stripB, sideA, sideB):
    def orient(strip, side):
        s = strip.copy()
        if side in (1, 3):
            s = s.T
        return normalize_strip(s)

    a = orient(stripA, sideA)
    b = orient(stripB, sideB)

    if a.size == 0 or b.size == 0:
        return 1e6

    if a.size != b.size:
        la, lb = a.size, b.size
        if la < lb:
            a = np.interp(np.linspace(0, la-1, lb), np.arange(la), a)
        else:
            b = np.interp(np.linspace(0, lb-1, la), np.arange(lb), b)

    ssd = np.mean((a - b)**2)
    ssd_flip = np.mean((a - b[::-1])**2)
    return min(ssd, ssd_flip)

# ----------------------------
# 4) Precompute compatibility matrices
# ----------------------------

def build_compatibility(pieces_edges, strip_width=16):
    n = len(pieces_edges)
    borders = [extract_borders(p, strip_width) for p in pieces_edges]
    compat = {s: np.full((n,n), 1e6, dtype=np.float32) for s in range(4)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            compat[0][i,j] = border_distance(borders[i][0], borders[j][2], 0, 2)
            compat[1][i,j] = border_distance(borders[i][1], borders[j][3], 1, 3)
            compat[2][i,j] = border_distance(borders[i][2], borders[j][0], 2, 0)
            compat[3][i,j] = border_distance(borders[i][3], borders[j][1], 3, 1)
    return compat

# ----------------------------
# 5) Solve 2x2 by brute-force (guaranteed)
# ----------------------------

def solve_bruteforce(pieces_edges, compat, grid_n):
    n = grid_n*grid_n
    best_perm = None
    best_score = 1e12
    for perm in permutations(range(n)):
        score = 0.0
        valid = True
        for pos, pid in enumerate(perm):
            r = pos // grid_n
            c = pos % grid_n
            if c > 0:
                left_pid = perm[pos-1]
                score += compat[1][left_pid, pid]
                if score >= best_score:
                    valid = False
                    break
            if r > 0:
                top_pid = perm[pos-grid_n]
                score += compat[2][top_pid, pid]
                if score >= best_score:
                    valid = False
                    break
        if not valid:
            continue
        if score < best_score:
            best_score = score
            best_perm = perm
    return list(best_perm), best_score

# ----------------------------
# 6) Backtracking solver for larger grids
# ----------------------------

def solve_backtracking(pieces_edges, compat, grid_n, top_k=10, time_limit=30.0):
    n = grid_n * grid_n
    start_time = time.time()

    placed = [-1] * n
    used = [False] * n

    best_solution = None
    best_score = 1e12

    def backtrack(pos, current_score):
        nonlocal best_solution, best_score
        if time.time() - start_time > time_limit:
            return
        if pos == n:
            if current_score < best_score:
                best_score = current_score
                best_solution = placed.copy()
            return
        r = pos // grid_n
        c = pos % grid_n
        candidates = []
        for pid in range(n):
            if used[pid]:
                continue
            score_local = 0.0
            if c > 0:
                left_pid = placed[pos-1]
                score_local += compat[1][left_pid, pid]
            if r > 0:
                top_pid = placed[pos-grid_n]
                score_local += compat[2][top_pid, pid]
            candidates.append((score_local, pid))
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:top_k]
        for score_local, pid in candidates:
            new_score = current_score + score_local
            if new_score >= best_score:
                continue
            placed[pos] = pid
            used[pid] = True
            backtrack(pos+1, new_score)
            used[pid] = False
            placed[pos] = -1
            if time.time() - start_time > time_limit:
                return

    backtrack(0, 0.0)
    return best_solution, best_score

# ----------------------------
# 7) Reconstruct image from ordering
# ----------------------------

def reconstruct_from_order(pieces, order, grid_n, piece_shape):
    ph, pw = piece_shape
    recon = np.zeros((grid_n*ph, grid_n*pw, 3), dtype=np.uint8)
    for pos, pid in enumerate(order):
        r = pos // grid_n
        c = pos % grid_n
        y0, y1 = r*ph, (r+1)*ph
        x0, x1 = c*pw, (c+1)*pw
        piece = pieces[pid]
        if piece.shape[0] != ph or piece.shape[1] != pw:
            piece_placed = cv2.resize(piece, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            piece_placed = piece
        recon[y0:y1, x0:x1] = piece_placed
    return recon

# ----------------------------
# 8) Full pipeline wrapper
# ----------------------------

def solve_image(img, grid_n, strip_width=16, top_k=12, time_limit=30.0, visualize=True):
    pieces, coords, (ph, pw) = cut_into_grid(img, grid_n)
    n = len(pieces)
    print(f"[+] Grid {grid_n}x{grid_n} -> {n} pieces, piece size {ph}x{pw}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    pieces_edges = [
        (lambda p: (lambda img_gray, img_clahe, gray, edges: edges)(
            cv2.cvtColor(p, cv2.COLOR_BGR2GRAY),
            clahe.apply(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)),
            cv2.bilateralFilter(clahe.apply(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)), 3, 75, 75),
            cv2.Canny(cv2.bilateralFilter(clahe.apply(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)), 3, 75, 75), 40, 180)
        ))(p) for p in pieces
    ]

    print("[+] Preprocessed pieces with exact pipeline (CLAHE -> bilateral -> Canny).")

    compat = build_compatibility(pieces_edges, strip_width=strip_width)
    print("[+] Built compatibility matrices (border distances).")

    if grid_n == 2:
        print("[*] Solving 2x2 by brute-force permutations.")
        order, score = solve_bruteforce(pieces_edges, compat, grid_n)
    else:
        print(f"[*] Solving {grid_n}x{grid_n} by backtracking (top_k={top_k}, time_limit={time_limit}s).")
        order, score = solve_backtracking(pieces_edges, compat, grid_n, top_k=top_k, time_limit=time_limit)

    if order is None:
        print("[-] No complete solution found within limits (returning best partial if any).")
        order = list(range(n))

    print("[+] Solution score:", score)
    recon = reconstruct_from_order(pieces, order, grid_n, (ph, pw))

    if visualize:
        sample = min(4, n)
        show_images([img, recon], titles=["Original", f"Reconstructed {grid_n}x{grid_n}"], figsize=(14,6))
    return order, recon, compat

# ----------------------------
# Main CLI (updated for multiple images)
# ----------------------------

def main(folder, filename, grids, strip_width, top_k, time_limit, visualize, limit=None):
    files = []

    if filename:
        path = filename if os.path.isabs(filename) else os.path.join(folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not load {path}")
        files = [path]
    else:
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                        if f.lower().endswith(('.png','.jpg','.jpeg'))])
        if not files:
            raise FileNotFoundError("No images found in folder.")
        if limit:
            files = files[:limit]

    for path in files:
        img_name = os.path.basename(path)
        print("\n" + "="*60)
        print(f"[+] Processing {img_name}")
        img = cv2.imread(path)
        if img is None:
            print(f"[-] Could not read {img_name}, skipping.")
            continue
        for g in grids:
            print(f"\n[*] Grid {g}x{g}")
            start = time.time()
            order, recon, compat = solve_image(
                img, g, strip_width=strip_width, top_k=top_k, time_limit=time_limit, visualize=visualize
            )
            print(f"[+] Finished grid {g}x{g} in {round(time.time()-start,2)}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-based puzzle solver (accurate for 2x2, practical for 4x4 & 8x8).")
    parser.add_argument("--folder", type=str, default=".", help="Folder containing images (if --file not set uses first).")
    parser.add_argument("--file", type=str, default=None, help="Specific image filename to use (optional).")
    parser.add_argument("--grids", type=str, default="2,4,8", help="Comma-separated grid sizes to attempt, e.g. '2,4,8'.")
    parser.add_argument("--strip", type=int, default=4, help="Border strip width in pixels (default 16).")
    parser.add_argument("--topk", type=int, default=12, help="Top-k candidates per slot for backtracking (pruning).")
    parser.add_argument("--timelimit", type=float, default=30.0, help="Time limit per grid solve (seconds). Increase for harder puzzles.)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization.")
    parser.add_argument("--limit", type=int, default=None, help="Number of images from the folder to process (default: all).")
    args = parser.parse_args()

    grids = [int(x) for x in args.grids.split(",") if x.strip().isdigit()]
    main(args.folder, args.file, grids,
        strip_width=args.strip, top_k=args.topk,
        time_limit=args.timelimit, visualize=not args.no_vis,
        limit=args.limit)
