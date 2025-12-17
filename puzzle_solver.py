# puzzle_solver_edges_refactored.py
import cv2
import numpy as np
import os
import argparse
import time
from itertools import permutations
from collections import deque
from typing import List, Tuple, Dict, Optional

# ----------------------------
# Utilities / Visualization
# ----------------------------
def show_images(images: List[np.ndarray], titles: Optional[List[str]] = None, figsize=(12,6), grayscale=None):
    import matplotlib.pyplot as plt
    n = len(images)
    titles = titles or [f"Image {i+1}" for i in range(n)]
    grayscale = [grayscale]*n if isinstance(grayscale, bool) else (grayscale or [False]*n)

    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(1, n, i+1)
        img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        plt.imshow(img_show, cmap='gray' if grayscale[i] else None)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ----------------------------
# 1) Cut image into grid pieces
# ----------------------------
def cut_into_grid(img: np.ndarray, grid_n: int) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]], Tuple[int,int]]:
    h, w = img.shape[:2]
    ph, pw = h // grid_n, w // grid_n
    pieces, coords = [], []
    for r in range(grid_n):
        for c in range(grid_n):
            y0, y1 = r*ph, (r+1)*ph
            x0, x1 = c*pw, (c+1)*pw
            pieces.append(img[y0:y1, x0:x1].copy())
            coords.append((y0,y1,x0,x1))
    return pieces, coords, (ph, pw)


# ----------------------------
# 2) Extract borders with LAB + gradient
# L --> Luminance or Lightness
# AB --> Chrominance (A, red green) (B, yellow blue)
# ----------------------------
def extract_borders(piece: np.ndarray, strip_width: int = 1) -> Dict[int, np.ndarray]:
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    sw = min(strip_width, h//2, w//2)

    def make_grad_patch_enhanced(patch_lab):
        patch_bgr = cv2.cvtColor(patch_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        patch_gray = cv2.cvtColor(patch_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        patch_gray = cv2.GaussianBlur(patch_gray, (3,3), 0)
        
        gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3) 
        gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)

        grad_mag = cv2.magnitude(gx, gy)[..., None]
        grad_dir = cv2.phase(gx, gy, angleInDegrees=True)[..., None]

        lap = cv2.Laplacian(patch_gray, cv2.CV_32F)[..., None]

        return np.concatenate([patch_lab, grad_mag, grad_dir, lap], axis=2)

    return {
        0: make_grad_patch_enhanced(lab[0:sw, :, :]),       # top (height, width, channels)
        1: make_grad_patch_enhanced(lab[:, w-sw:w, :]),     # right
        2: make_grad_patch_enhanced(lab[h-sw:h, :, :]),     # bottom
        3: make_grad_patch_enhanced(lab[:, 0:sw, :])        # left
    }


# ----------------------------
# 3) Normalize strip (zero-mean, unit norm)
# ----------------------------
def normalize_strip_2d(strip: np.ndarray) -> np.ndarray:
    arr = strip.astype(np.float32)
    for ch in range(arr.shape[2]):
        m, sd = arr[..., ch].mean(), arr[..., ch].std()
        arr[..., ch] = (arr[..., ch] - m) / (sd if sd > 1e-6 else 1.0)
    return arr


def mirror_for_side(strip: np.ndarray, side: int) -> np.ndarray:
    if side in (0,2):  # top/bottom
        return strip[:, ::-1, :]
    return strip[::-1, :, :]  # left/right


def border_distance_2d(stripA, stripB, sideA, sideB, p=0.3, q=1/16,
                        w_color=0.4, w_grad_mag=0.2, w_grad_dir=0.2, 
                        w_lap=0.4):
    # orient strips
    def orient(s, side):
        return normalize_strip_2d(np.transpose(s, (1,0,2)) if side in (1,3) else s)

    a, b = orient(stripA, sideA), orient(stripB, sideB)
    if a.size == 0 or b.size == 0:
        return 1e9
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    def dist(x, y):
        d_color = np.sum(np.abs(x[...,:3] - y[...,:3])**p)
        d_grad_mag = np.sum(np.abs(x[...,3:4] - y[...,3:4])**p)
        d_grad_dir = np.sum(np.abs(x[...,4:5] - y[...,4:5])**p)
        d_lap = np.sum(np.abs(x[...,5:8] - y[...,5:8])**p)
        total = (w_color*d_color + w_grad_mag*d_grad_mag + w_grad_dir*d_grad_dir +
                 + w_lap*d_lap)
        return total**(q/p)

    # compute distance and mirrored distance
    d1 = dist(a, b)

    # d2 = dist(a, mirror_for_side(b, sideB)) in case pieces rotated 180 degress or direction aren't direct 
    # d = min(d1, d2) 

    return float(d1)


# ----------------------------
# 4) Precompute compatibility
# ----------------------------
def build_compatibility(pieces: List[np.ndarray], strip_width=8) -> Dict[int, np.ndarray]:
    n = len(pieces)
    
    borders = [extract_borders(p, strip_width) for p in pieces]
    compat = {s: np.full((n,n), 1e9, dtype=np.float32) for s in range(4)}
    
    for i in range(n):
        for j in range(n):
            if i==j: continue
            compat[0][i,j] = border_distance_2d(borders[i][0], borders[j][2], 0,2)
            compat[1][i,j] = border_distance_2d(borders[i][1], borders[j][3], 1,3)
            compat[2][i,j] = border_distance_2d(borders[i][2], borders[j][0], 2,0)
            compat[3][i,j] = border_distance_2d(borders[i][3], borders[j][1], 3,1)
    return compat

# ----------------------------
# Best-buddies helpers
# ----------------------------
def opposite(side):
    return (side + 2) % 4

def best_partner_for(i, side, compat):
    """Return index j that is best (minimal) partner for piece i's given side."""
    arr = compat[side][i]
    # argmin over j (note diagonal is big)
    return int(np.argmin(arr))

def is_best_buddy(i, side, j, compat):
    """Check mutual best-buddy: i(side) -> j and j(opposite) -> i"""
    if i == j:
        return False
    bj = best_partner_for(i, side, compat)
    if bj != j:
        return False
    opp = opposite(side)
    bi = best_partner_for(j, opp, compat)
    return bi == i


def placer_beam(
    n,
    grid_n,
    compat,
    beam_width=10,
    top_k=6,
    used_seeds=None,
    max_seed_trials=100
):
    """
    Enhanced Beam-search based placer with global seed tracking.

    used_seeds : set of (seed_pid, seed_pos) used across runs
    """

    if used_seeds is None:
        used_seeds = set()

    # ----------------------------
    # Precompute neighbors
    # ----------------------------
    neighbor_map = {}
    for pos in range(n):
        r, c = divmod(pos, grid_n)
        neighs = []
        if r > 0: neighs.append((pos - grid_n, 2))
        if r < grid_n - 1: neighs.append((pos + grid_n, 0))
        if c > 0: neighs.append((pos - 1, 1))
        if c < grid_n - 1: neighs.append((pos + 1, 3))
        neighbor_map[pos] = neighs

    # ----------------------------
    # Initial beam (unique seeds)
    # ----------------------------
    beam = []
    trials = 0

    while len(beam) < beam_width and trials < max_seed_trials:
        trials += 1
        seed_pid = np.random.randint(0, n)
        seed_pos = np.random.randint(0, n)
        key = (seed_pid, seed_pos)

        if key in used_seeds:
            continue

        used_seeds.add(key)

        state = {
            "placement": [-1] * n,
            "used": {seed_pid},
            "cost": 0.0
        }
        state["placement"][seed_pos] = seed_pid
        beam.append(state)

    # fallback: allow reuse if exhausted
    if not beam:
        seed_pid = np.random.randint(0, n)
        seed_pos = np.random.randint(0, n)
        state = {
            "placement": [-1] * n,
            "used": {seed_pid},
            "cost": 0.0
        }
        state["placement"][seed_pos] = seed_pid
        beam.append(state)

    # ----------------------------
    # Beam expansion
    # ----------------------------
    for _ in range(1, n):
        candidates = []

        for state in beam:
            placement = state["placement"]
            used = state["used"]
            base_cost = state["cost"]

            # choose empty slot with max filled neighbors
            slots = []
            for pos in range(n):
                if placement[pos] != -1:
                    continue
                filled = [(p, s) for p, s in neighbor_map[pos] if placement[p] != -1]
                if filled:
                    slots.append((-len(filled), pos, filled))

            if not slots:
                continue

            slots.sort()
            _, slot, neighs = slots[0]

            # score unused pieces
            scores = []
            for pid in range(n):
                if pid in used:
                    continue
                cost = sum(
                    compat[side][placement[nb], pid]
                    for nb, side in neighs
                )
                scores.append((cost, pid))

            scores.sort()
            for inc_cost, pid in scores[:top_k]:
                new_state = {
                    "placement": placement.copy(),
                    "used": used.copy(),
                    "cost": base_cost + inc_cost
                }
                new_state["placement"][slot] = pid
                new_state["used"].add(pid)
                candidates.append(new_state)

        if not candidates:
            break

        candidates.sort(key=lambda x: x["cost"])
        beam = candidates[:beam_width]

    beam.sort(key=lambda x: x["cost"])
    return beam[0]["placement"]



# ----------------------------
# Placer (greedy, with best-buddies primary)
# ----------------------------
def placer(
    n,
    grid_n,
    compat,
    seed_placement=None,
    seed_center=True,
    used_seeds=None,          # NEW
    max_seed_tries=100        # safety
):
    """
    Greedy placer with seed tracking.
    - used_seeds: set of (seed_pid, seed_pos) already tried
    """

    if used_seeds is None:
        used_seeds = set()

    placement = [-1] * n
    used = [False] * n

    def opposite(side):
        return (side + 2) % 4

    def mutual_best_buddy(a, side_a, b):
        best_for_a = np.argmin(compat[side_a][a])
        best_for_b = np.argmin(compat[opposite(side_a)][b])
        return best_for_a == b and best_for_b == a

    # -------------------------------------------------
    # STEP 1: place seed(s)
    # -------------------------------------------------
    if seed_placement:
        seed_pos = list(seed_placement.keys())
        rs = [p // grid_n for p in seed_pos]
        cs = [p % grid_n for p in seed_pos]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        seed_h = rmax - rmin + 1
        seed_w = cmax - cmin + 1

        if seed_center:
            top = (grid_n - seed_h) // 2
            left = (grid_n - seed_w) // 2
        else:
            top, left = 0, 0

        for pos_old, pid in seed_placement.items():
            r_old, c_old = pos_old // grid_n, pos_old % grid_n
            r_new = top + (r_old - rmin)
            c_new = left + (c_old - cmin)
            if 0 <= r_new < grid_n and 0 <= c_new < grid_n:
                pos_new = r_new * grid_n + c_new
                placement[pos_new] = pid
                used[pid] = True

    else:
        # ---------- RANDOM SEED WITH TRACKING ----------
        for _ in range(max_seed_tries):
            seed_pid = np.random.randint(0, n)
            seed_pos = np.random.randint(0, n)

            if (seed_pid, seed_pos) not in used_seeds:
                used_seeds.add((seed_pid, seed_pos))
                placement[seed_pos] = seed_pid
                used[seed_pid] = True
                break
        else:
            # fallback: deterministic unused combo
            for pid in range(n):
                for pos in range(n):
                    if (pid, pos) not in used_seeds:
                        used_seeds.add((pid, pos))
                        placement[pos] = pid
                        used[pid] = True
                        break
                else:
                    continue
                break

    # -------------------------------------------------
    # STEP 2: greedy filling (UNCHANGED LOGIC)
    # -------------------------------------------------
    def get_neighbors(pos):
        r, c = pos // grid_n, pos % grid_n
        neighbors = []
        if r > 0 and placement[pos - grid_n] != -1:
            neighbors.append((pos - grid_n, 2))
        if r < grid_n - 1 and placement[pos + grid_n] != -1:
            neighbors.append((pos + grid_n, 0))
        if c > 0 and placement[pos - 1] != -1:
            neighbors.append((pos - 1, 1))
        if c < grid_n - 1 and placement[pos + 1] != -1:
            neighbors.append((pos + 1, 3))
        return neighbors

    slots_filled = sum(p != -1 for p in placement)

    while slots_filled < n:
        empty_slots = []
        for pos in range(n):
            if placement[pos] != -1:
                continue
            neighs = get_neighbors(pos)
            if neighs:
                empty_slots.append((-len(neighs), pos, neighs))

        if not empty_slots:
            pos = placement.index(-1)
            empty_slots = [(0, pos, [])]

        empty_slots.sort()
        chosen = None

        # ---- mutual best buddies ----
        for _, slot_pos, neighs in empty_slots:
            candidates = []
            for pid in range(n):
                if used[pid]:
                    continue
                bb_count = 0
                compat_sum = 0.0
                for neigh_pos, neigh_side in neighs:
                    neigh_pid = placement[neigh_pos]
                    if mutual_best_buddy(neigh_pid, neigh_side, pid):
                        bb_count += 1
                    compat_sum += compat[neigh_side][neigh_pid, pid]
                if bb_count > 0:
                    candidates.append((bb_count, compat_sum, slot_pos, pid))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                chosen = candidates[0]
                break

        # ---- fallback ----
        if chosen is None:
            _, slot_pos, neighs = empty_slots[0]
            best_val = 1e18
            best_pid = None
            for pid in range(n):
                if used[pid]:
                    continue
                ssum = sum(
                    compat[side][placement[nb], pid]
                    for nb, side in neighs
                )
                avg = ssum / max(1, len(neighs))
                if avg < best_val:
                    best_val = avg
                    best_pid = pid
            chosen = (0, best_val, slot_pos, best_pid)

        _, _, slot_pos, chosen_pid = chosen
        placement[slot_pos] = chosen_pid
        used[chosen_pid] = True
        slots_filled += 1

    return placement



# ----------------------------
# Segmenter (region growing using best-buddies predicate)
# ----------------------------
def segmenter(placement, grid_n, compat):
    """
    Given a full placement (position -> piece_id), return list of segments
    Each segment is a list of positions that are connected via best-buddy neighbor relation.
    """
    n_slots = len(placement) # grid_n * grid_n
    visited = [False] * n_slots
    segments = []

    def neighbors(pos):
        r = pos // grid_n
        c = pos % grid_n
        if c > 0: yield pos-1, 3
        if c < grid_n-1: yield pos+1, 1
        if r > 0: yield pos-grid_n, 0
        if r < grid_n-1: yield pos+grid_n, 2

    for pos in range(n_slots): # 0 --> 15
        if visited[pos]:
            continue
        # BFS/region grow using best-buddy predicate
        queue = deque([pos])
        comp = []
        visited[pos] = True
        while queue:
            u = queue.popleft()
            comp.append(u)
            pu = placement[u]
            for v, side_of_u in neighbors(u):
                if visited[v]:
                    continue
                pv = placement[v]
                # For u at side side_of_u facing v, check if u and v are best-buddies:
                # side_of_u is the side index on u that faces v.
                if is_best_buddy(pu, side_of_u, pv, compat):
                    visited[v] = True
                    queue.append(v)
        if comp:
            segments.append(comp)
    return segments

# ----------------------------
# Best-buddies estimation metric
# ----------------------------
def compute_best_buddies_score(placement, grid_n, compat):
    """
    Score = (number of adjacent neighbor pairs that are mutual best buds) / (total number of adjacent neighbor pairs in placement)
    Each undirected neighbor pair counted once.
    """
    n_slots = len(placement)
    bb_count = 0
    total_adj = 0
    for pos in range(n_slots):
        r = pos // grid_n
        c = pos % grid_n
        pid = placement[pos]
        # right neighbor and bottom neighbor to avoid double counting
        if c < grid_n - 1:
            np0 = pos + 1
            pid2 = placement[np0]
            total_adj += 1
            if is_best_buddy(pid, 1, pid2, compat):
                bb_count += 1
        if r < grid_n - 1:
            np1 = pos + grid_n
            pid2 = placement[np1]
            total_adj += 1
            if is_best_buddy(pid, 2, pid2, compat):
                bb_count += 1
    if total_adj == 0:
        return 0.0
    return bb_count / total_adj

# ----------------------------
# Shifter: iterative re-seeding with largest segment
# ----------------------------
def shifter(initial_placement, grid_n, compat, max_iters=10, swap_pass=True, target_score=0.5):
    """
    Iteratively improve placement using segmentation + reseeding + optional local swaps.
    """
    n_slots = len(initial_placement)
    current = initial_placement.copy()
    best_score = compute_best_buddies_score(current, grid_n, compat)
    best_placement = current.copy()
    used_seeds = set()
    
    for it in range(max_iters):
        segments = segmenter(current, grid_n, compat)
        if not segments:
            break

        # Sort segments by size descending
        segments.sort(key=lambda x: -len(x))
        improved = False

        for seg in segments:
            if len(seg) == 0:
                continue
            # Create seed_placement for this segment
            seed_map = {pos: current[pos] for pos in seg}
            # Re-run placer with seed
            placement_new = placer(n_slots, grid_n, compat, seed_placement=seed_map)
            score_new = compute_best_buddies_score(placement_new, grid_n, compat)

            if score_new > best_score + 1e-9:
                current = placement_new
                best_score = score_new
                best_placement = current.copy()
                improved = True
                

        if not improved:
            # optional local swap pass to improve BB-score
            if swap_pass:
                for pos1 in range(n_slots): # 0 -> 15
                    for pos2 in range(pos1+1, n_slots): # 0+1 -> 15
                        new_placement = current.copy()
                        new_placement[pos1], new_placement[pos2] = new_placement[pos2], new_placement[pos1]
                        score_swap = compute_best_buddies_score(new_placement, grid_n, compat)
                        if score_swap > best_score + 1e-9:
                            current = new_placement
                            best_score = score_swap
                            best_placement = current.copy()
                            improved = True
                            break
                    if improved:
                        break

        if not improved: 
            if best_score >= target_score:
                return best_placement, best_score
            current = placer(n_slots, grid_n, compat, used_seeds=used_seeds)
            

    return best_placement, best_score
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

def visualize_placement(pieces, placement, grid_n, piece_shape, title="Placement"):
    recon = reconstruct_from_order(pieces, placement, grid_n, piece_shape)
    show_images([recon], titles=[title], figsize=(6,6))


# ----------------------------
# 8) Full pipeline wrapper (integrates placer, segmenter, shifter)
# ----------------------------
def solve_image(img, grid_n, strip_width=8, top_k=12, time_limit=30.0, visualize=True, seeds=5, shifter_iters=8):
    pieces, coords, (ph, pw) = cut_into_grid(img, grid_n)
    n = len(pieces)
    print(f"[+] Grid {grid_n}x{grid_n} -> {n} pieces, piece size {ph}x{pw}")

    # build compatibility (expensive)
    compat = build_compatibility(pieces, strip_width=strip_width)
    print("[+] Built compatibility matrices (border distances).")

    if grid_n == 2:
        print("[*] Solving 2x2 by brute-force permutations.")
        order, score = solve_bruteforce(pieces, compat, grid_n)
        placement = order
        best_bb_score = compute_best_buddies_score(order, grid_n, compat)
    else:
        # Run multiple random seeds, keep best result by BB-score as paper does
        best_placement = None
        best_bb_score = -1.0
        used_seeds = set()

        for s in range(seeds):
            print(f"[*] Seed run {s+1}/{seeds}")
            # run initial placer with a single random seed
            init_placement = placer_beam(
                n,
                grid_n,
                compat,
                beam_width=top_k,   # reuse CLI param
                top_k=6,
                used_seeds=used_seeds
            )
            bb0 = compute_best_buddies_score(init_placement, grid_n, compat)

            # --- visualize placer ---
            """
            if visualize:
                visualize_placement(pieces, init_placement, grid_n, (ph,pw), title=f"Seed {s+1}: After Placer")
            """
            # --- visualize segments ---
            """
            if visualize:
                segments = segmenter(init_placement, grid_n, compat)
                seg_img = np.zeros_like(img)
                for i, seg in enumerate(segments):
                    color = tuple(int(c) for c in np.random.randint(0,255,3))
                    for pos in seg:
                        r = pos // grid_n
                        c = pos % grid_n
                        y0, y1, x0, x1 = coords[r*grid_n + c]
                        seg_img[y0:y1, x0:x1] = color
                show_images([seg_img], titles=[f"Seed {s+1}: Segments"], figsize=(6,6))
            """

            # run shifter (iteratively improve)
            placement_after_shifter, bb_sh = shifter(init_placement, grid_n, compat, max_iters=shifter_iters)

            # --- visualize after shifter ---
            """
            if visualize:
                visualize_placement(pieces, placement_after_shifter, grid_n, (ph,pw), title=f"Seed {s+1}: After Shifter")
            """
            # take whichever is better
            if bb_sh >= bb0:
                final_placement = placement_after_shifter
                final_bb = bb_sh
            else:
                final_placement = init_placement
                final_bb = bb0

            print(f"    BB-score after seed {s+1}: {final_bb:.4f}")
            if final_bb > best_bb_score:
                best_bb_score = final_bb
                best_placement = final_placement

        placement = best_placement
        # score variable for backward compat compatibility with older API
        score = 0.0

    recon = reconstruct_from_order(pieces, placement, grid_n, (ph, pw))

    if visualize:
        show_images([img, recon], titles=["Original", f"Reconstructed {grid_n}x{grid_n}"], figsize=(14,6))
    return placement, recon, compat

def compute_error(img1, img2):
    """Compute the mean squared error between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions to compute error.")
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)


# ----------------------------
# Main CLI 
# ----------------------------
def main(folder, filename, grids, strip_width, top_k, time_limit, visualize, limit=None, correct_folder=None, seeds=5, iterations=10, save_recon_folder=None, img_range=None):
    files = []
    false_images = []
    correct = 0
    total = 0

    # Load files
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
        # apply limit
        if limit:
            files = files[:limit]
        # apply range
        if img_range:
            start, end = img_range
            files = files[start:end]

    if save_recon_folder:
        os.makedirs(save_recon_folder, exist_ok=True)

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
            placement, recon, compat = solve_image(
                img, g, strip_width=strip_width, top_k=top_k, time_limit=time_limit, visualize=visualize, seeds=seeds, shifter_iters=iterations
            )
            total+=1
            if save_recon_folder:
                base_name = os.path.splitext(img_name)[0]
                save_path = os.path.join(save_recon_folder, f"{base_name}_{g}x{g}_recon.png")
                cv2.imwrite(save_path, recon)
                print(f"[+] Saved reconstructed image to {save_path}")

            if correct_folder:
                base_name = os.path.splitext(img_name)[0]
                found = False
                for ext in ['.png', '.jpg', '.jpeg']:
                    correct_path = os.path.join(correct_folder, base_name + ext)
                    if os.path.exists(correct_path):
                        correct_img = cv2.imread(correct_path)
                        if correct_img is not None:
                            error = compute_error(recon, correct_img)
                            if error<400:
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

# ----------------------------
# CLI parser update
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-based puzzle solver with Placer+Segmenter+Shifter (paper pipeline).")
    parser.add_argument("--folder", type=str, default="gravity_falls_dataset/puzzle_2x2", help="Folder containing images.")
    parser.add_argument("--correct-folder", type=str, default="gravity_falls_dataset/correct", help="Folder containing correct images.")
    parser.add_argument("--file", type=str, default=None, help="Specific image filename to use (optional).")
    parser.add_argument("--grids", type=str, default="2,4,8", help="Comma-separated grid sizes to attempt, e.g. '2,4,8'.")
    parser.add_argument("--strip", type=int, default=1, help="Border strip width in pixels (default 8).")
    parser.add_argument("--topk", type=int, default=12, help="Top-k candidates per slot for backtracking (pruning).")
    parser.add_argument("--timelimit", type=float, default=30.0, help="Time limit per grid solve (seconds).")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization.")
    parser.add_argument("--limit", type=int, default=None, help="Number of images from the folder to process (default: all).")
    parser.add_argument("--range", type=str, default=None, help="Range of images to process, e.g. '0-10'.")
    parser.add_argument("--save-recon-folder", type=str, default=None, help="Folder to save reconstructed images.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random placer seeds to try.")
    parser.add_argument("--iter", type=int, default=10, help="Number of shifter iterations to attempt.")
    args = parser.parse_args()

    grids = [int(x) for x in args.grids.split(",") if x.strip().isdigit()]
    img_range = None
    if args.range:
        parts = args.range.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            img_range = (int(parts[0]), int(parts[1]))

    main(args.folder, args.file, grids,
         strip_width=args.strip, top_k=args.topk,
         time_limit=args.timelimit, visualize=not args.no_vis,
         limit=args.limit, correct_folder=args.correct_folder,
         seeds=args.seeds, iterations=args.iter,
         save_recon_folder=args.save_recon_folder, img_range=img_range)
