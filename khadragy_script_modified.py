# puzzle_solver_edges.py
import cv2
import numpy as np
import os
import argparse
import time
from itertools import permutations

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
    """Return list of pieces (row-major) and their coords (y0,y1,x0,x1)."""
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
# 2) Extract border strips
# ----------------------------
def extract_borders(piece, strip_width=16):
    """
    Extracts border strips as grayscale arrays:
    returns dict: {0: top, 1: right, 2: bottom, 3: left}
    Each strip is shaped (strip_width, width) or (height, strip_width) depending on side.
    """
    # Preprocess piece
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(piece_gray)
    smmothed = cv2.bilateralFilter(img_clahe, 3, 75, 75)
    edges = cv2.Canny(smmothed, 40, 180)
    p = piece_gray
    h, w = p.shape
    sw = min(strip_width, h//2, w//2)
    top = p[0:sw, :].astype(np.float32)
    bottom = p[h-sw:h, :].astype(np.float32)
    left = p[:, 0:sw].astype(np.float32)
    right = p[:, w-sw:w].astype(np.float32)
    # For left/right, make them same orientation as top/bottom for comparison:
    # We'll compare right of A with left of B by flipping left/right appropriately in similarity
    return {0: top, 1: right, 2: bottom, 3: left}

# ----------------------------
# 3) Border similarity metric
# ----------------------------
def normalize_strip(s):
    """Zero-mean, unit-norm flattening for robust comparison."""
    if s.size == 0:
        return s.flatten()
    arr = s.astype(np.float32).flatten()
    mean = arr.mean()
    std = arr.std()
    if std < 1e-6:
        return (arr - mean)
    return (arr - mean) / std

def border_distance(stripA, stripB, sideA, sideB):
    """
    Distance between two border strips (lower = more likely match).
    sideA & sideB tell orientation so we flip/transpose as needed.
    We'll compare A's edge to B's opposite edge orientation:
    - top vs bottom: compare rows as-is
    - right vs left: compare columns; we transpose so both flatten same order
    For robustness we also compute mirrored version and take minimum (handles slight flips).
    """
    # Convert to normalized 1D arrays with consistent orientation.
    # For horizontal strips (top/bottom), shape = (sw, width) -> keep as-is
    # For vertical strips (left/right), shape = (height, sw) -> transpose to (sw, height) for comparable flattening
    def orient(strip, side):
        s = strip.copy()
        if side in (1, 3):  # right or left: transpose to become (cols, rows)
            s = s.T
        return normalize_strip(s)

    a = orient(stripA, sideA)
    b = orient(stripB, sideB)

    if a.size == 0 or b.size == 0:
        return 1e6
    # resize shorter to match longer for fair comparison
    if a.size != b.size:
        # simple interpolation resize in 1D via numpy (reshape by length)
        la = a.size; lb = b.size
        if la < lb:
            a = np.interp(np.linspace(0, la-1, lb), np.arange(la), a)
        else:
            b = np.interp(np.linspace(0, lb-1, la), np.arange(lb), b)

    # raw SSD
    ssd = np.mean((a - b)**2)
    # also compare with flipped version for safety (some seams are reversed)
    ssd_flip = np.mean((a - b[::-1])**2)
    return min(ssd, ssd_flip)

def gradient_strip(s):
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return grad

# ----------------------------
# 4) Precompute compatibility matrices
# ----------------------------
def build_compatibility(pieces, strip_width=16):
    """
    For N pieces return compat[side][i][j] = distance of
    piece i side vs piece j opposite side.
    side indices: 0:top,1:right,2:bottom,3:left
    opposite side mapping: 0<->2, 1<->3
    compat[1][i][j] compares right of i and left of j.
    Lower distance -> better match.
    """
    n = len(pieces)
    borders = [extract_borders(p, strip_width) for p in pieces]
    compat = {s: np.full((n,n), 1e6, dtype=np.float32) for s in range(4)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # top of i vs bottom of j (i above j): i.top vs j.bottom
            compat[0][i,j] = border_distance(borders[i][0], borders[j][2], 0, 2)
            # right of i vs left of j (i left of j)
            compat[1][i,j] = border_distance(borders[i][1], borders[j][3], 1, 3)
            # bottom of i vs top of j (i below j)
            compat[2][i,j] = border_distance(borders[i][2], borders[j][0], 2, 0)
            # left of i vs right of j (i right of j)
            compat[3][i,j] = border_distance(borders[i][3], borders[j][1], 3, 1)
    return compat

# ----------------------------
# 5) Solve 2x2 by brute-force (guaranteed)
# ----------------------------
def solve_bruteforce(pieces, compat, grid_n):
    n = grid_n*grid_n
    best_perm = None
    best_score = 1e12
    perm_count = 0
    for perm in permutations(range(n)):
        perm_count += 1
        # early pruning: compute pairwise constraints as we go
        score = 0.0
        valid = True
        for pos, pid in enumerate(perm):
            r = pos // grid_n
            c = pos % grid_n
            # check left neighbor
            if c > 0:
                left_pid = perm[pos-1]
                # left_pid right vs pid left
                score += compat[1][left_pid, pid]
                if score >= best_score:
                    valid = False
                    break
            # check top neighbor
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
# 6) Backtracking solver for larger grids (4x4,8x8) with pruning
# ----------------------------
def solve_backtracking(pieces, compat, grid_n, top_k=10, time_limit=30.0):
    """
    Backtracking with pruning:
    - place pieces row-major
    - when selecting candidate for a slot, consider only top_k best candidates by sum of compat to already placed neighbors
    - prune if partial score exceeds best found
    - time_limit in seconds stops search early (returns best found)
    """
    n = grid_n * grid_n
    start_time = time.time()

    # For faster candidate selection, precompute for each piece lists of best partners per side
    # partner_lists[side][i] = list of j sorted by compat[side][i,j]
    partner_lists = {s: [np.argsort(compat[s][i,:]) for i in range(n)] for s in range(4)}

    best_solution = None
    best_score = 1e12

    placed = [-1] * n
    used = [False] * n

    # order of positions: row-major
    def backtrack(pos, current_score):
        nonlocal best_solution, best_score
        if time.time() - start_time > time_limit:
            return  # time cutoff

        if pos == n:
            # full placement
            if current_score < best_score:
                best_score = current_score
                best_solution = placed.copy()
            return

        r = pos // grid_n
        c = pos % grid_n

        # Build candidate list: all unused pieces with heuristic sort
        candidates = []
        for pid in range(n):
            if used[pid]:
                continue
            # compute local compatibility to existing neighbors
            score_local = 0.0
            # left neighbor
            if c > 0:
                left_pid = placed[pos-1]
                score_local += compat[1][left_pid, pid]  # left.right vs pid.left
            # top neighbor
            if r > 0:
                top_pid = placed[pos-grid_n]
                score_local += compat[2][top_pid, pid]  # top.bottom vs pid.top
            candidates.append((score_local, pid))
        # sort by local score (lower first)
        candidates.sort(key=lambda x: x[0])
        # keep only top_k promising
        candidates = candidates[:top_k]

        for score_local, pid in candidates:
            new_score = current_score + score_local
            if new_score >= best_score:
                continue  # prune
            # place
            placed[pos] = pid
            used[pid] = True
            backtrack(pos+1, new_score)
            used[pid] = False
            placed[pos] = -1
            # early exit if time exceeded
            if time.time() - start_time > time_limit:
                return

    # start backtracking
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
        # if piece dimensions differ slightly, resize for placement only
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

    compat = build_compatibility(pieces, strip_width=strip_width)
    print("[+] Built compatibility matrices (border distances).")

    if grid_n == 2:
        print("[*] Solving 2x2 by brute-force permutations.")
        order, score = solve_bruteforce(pieces, compat, grid_n)
    else:
        # time_limit controls search time; increase for larger puzzles if needed
        print(f"[*] Solving {grid_n}x{grid_n} by backtracking (top_k={top_k}, time_limit={time_limit}s).")
        order, score = solve_backtracking(pieces, compat, grid_n, top_k=top_k, time_limit=time_limit)

    if order is None:
        print("[-] No complete solution found within limits (returning best partial if any).")
        # attempt to return naive order if nothing found
        order = list(range(n))

    print("[+] Solution score:", score)
    recon = reconstruct_from_order(pieces, order, grid_n, (ph, pw))

    if visualize:
        sample = min(4, n)
        show_images([img, recon], titles=["Original", f"Reconstructed {grid_n}x{grid_n}"], figsize=(14,6))
    return order, recon, compat

def compute_error(img1, img2):
    """Compute the mean squared error between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions to compute error.")
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

# ----------------------------
# Main CLI (updated for multiple images)
# ----------------------------
def main(folder, filename, grids, strip_width, top_k, time_limit, visualize, limit=None, correct_folder=None):
    files = []
    false_images = []
    correct = 0
    total = 0
    if filename:
        # Single image mode
        path = filename if os.path.isabs(filename) else os.path.join(folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not load {path}")
        files = [path]
    else:
        # Folder mode
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
            total+=1
            if correct_folder:
                base_name = os.path.splitext(img_name)[0]
                found = False
                for ext in ['.png', '.jpg', '.jpeg']:
                    correct_path = os.path.join(correct_folder, base_name + ext)
                    if os.path.exists(correct_path):
                        correct_img = cv2.imread(correct_path)
                        if correct_img is not None:
                            error = compute_error(recon, correct_img)
                            if error<200:
                                correct+=1
                            else:
                                false_images.append(base_name)
                            print(f"[+] Reconstruction error (MSE) for {g}x{g}: {error:.4f}")
                        else:
                            print(f"[-] Could not read correct image {correct_path}")
                        found = True
                        break
                if not found:
                    print(f"[-] Correct image not found for {img_name} in {correct_folder}")
            print(f"[+] Finished grid {g}x{g} in {round(time.time()-start,2)}s")
            
    print(f"Correct images = {correct}/{total}")
    print(f"Uncorrect image: {false_images}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-based puzzle solver (accurate for 2x2, practical for 4x4 & 8x8).")
    parser.add_argument("--folder", type=str, default="gravity_falls_dataset/puzzle_2x2", help="Folder containing images (if --file not set uses first).")
    parser.add_argument("--correct-folder", type=str, default="gravity_falls_dataset/correct", help="Folder containing correct images.")
    parser.add_argument("--file", type=str, default=None, help="Specific image filename to use (optional).")
    parser.add_argument("--grids", type=str, default="2,4,8", help="Comma-separated grid sizes to attempt, e.g. '2,4,8'.")
    parser.add_argument("--strip", type=int, default=1, help="Border strip width in pixels (default 16).")
    parser.add_argument("--topk", type=int, default=12, help="Top-k candidates per slot for backtracking (pruning).")
    parser.add_argument("--timelimit", type=float, default=30.0, help="Time limit per grid solve (seconds). Increase for harder puzzles.)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization.")
    parser.add_argument("--limit", type=int, default=None, help="Number of images from the folder to process (default: all).")
    args = parser.parse_args()

    grids = [int(x) for x in args.grids.split(",") if x.strip().isdigit()]
    main(args.folder, args.file, grids,
        strip_width=args.strip, top_k=args.topk,
        time_limit=args.timelimit, visualize=not args.no_vis,
        limit=args.limit, correct_folder=args.correct_folder)
