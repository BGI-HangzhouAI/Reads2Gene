# filepath: d:\UCAS\genemodel\generator.py
import random
import numpy as np
from collections import Counter
import json
import os
from pathlib import Path
import math

# ---------------- Tunable parameters (kept at top of file) ----------------
# Target sequence length (bp)
TARGET_LEN = 10_000
# Random seed for reproducibility (None for random)
SEED = 42

# State weights: relative probability of each state/region during generation
# biological meaning: most of the human genome is intergenic; proportions of gc_rich/cpg_island/repeat/gene are adjustable
STATE_WEIGHTS = {
    "intergenic": 0.60,
    "gc_rich":    0.20,
    "cpg_island": 0.05,
    "repeat":     0.10,
    "gene":       0.05
}

# Overall repeat insertion fraction (approx. proportion), for post-processing insertions
# biological meaning: controls overall abundance of detectable repeats (e.g., Alu/LINE)
REPEAT_FRACTION = 0.02

# Microsatellite (short tandem repeat) insertion density, per-bp probability
# biological meaning: controls microsatellite density in the sequence
MICROSAT_DENSITY = 0.0004

# CpG island target CpG density (local)
# biological meaning: enrichment of CG dinucleotides in CpG islands; adjustable to simulate island strength
CPG_TARGET_DENSITY = 0.02

# Simple state model (also parameters)
STATES = {
    "intergenic": {"freq": {"A":0.30,"C":0.20,"G":0.20,"T":0.30}, "mean_len":10000},
    "gc_rich":   {"freq": {"A":0.20,"C":0.30,"G":0.30,"T":0.20}, "mean_len":2000},
    "cpg_island":{"freq": {"A":0.18,"C":0.32,"G":0.32,"T":0.18}, "mean_len":400},
    "repeat":    {"freq": {"A":0.25,"C":0.25,"G":0.25,"T":0.25}, "mean_len":500},
    "gene":      {"freq": {"A":0.28,"C":0.22,"G":0.22,"T":0.28}, "mean_len":1500}
}

# Replaceable repeat motif list (example); ideally use real Alu/LINE sequences or load from FASTA
REPEAT_MOTIFS = [
    "AATAT", 
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATAATATATATATATATATTATATATATATATATATATATAT", 
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTT",
    # Short Alu-like example fragment (placeholder for testing)
    "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGG",
]

def sanitize_repeat_motifs(motifs):
    """
    Keep only repeat sequences made of A/C/G/T with length >= 4,
    preventing placeholder or label text from being inserted.
    """
    clean = []
    for m in motifs:
        if not isinstance(m, str):
            continue
        mm = m.strip().upper()
        if len(mm) < 4:
            continue
        if all(ch in "ACGT" for ch in mm):
            clean.append(mm)
    return clean

# Allowed base directory for writes (prevents path traversal)
# Note: do not use raw strings containing \U in Python source (unicodeescape error)
# Use forward slashes or escaped backslashes instead
BASE_OUTPUT_DIR = os.path.abspath("data")
# Subdirectory; default output goes to d:\UCAS\genemodel
OUTPUT_SUBDIR = "genemodel"

def _ensure_safe_path(target_path: str, base_dir: str = BASE_OUTPUT_DIR) -> str:
    """
    Validate and return a safe absolute path; raise if target_path is outside base_dir.
    """
    tp = Path(target_path)
    # Disallow directories; require a file path
    # Resolve relative paths from current working directory
    resolved = tp.resolve()
    base = Path(base_dir).resolve()
    try:
        common = Path(os.path.commonpath([str(resolved), str(base)]))
    except ValueError:
        raise ValueError("Output path resolution error")
    if common != base and str(common) != str(base):
        raise ValueError(f"Writing to path not allowed: {resolved} (must be under {base})")
    return str(resolved)

def sample_state_length(mean_len):
    return max(1, int(random.expovariate(1.0/mean_len)))

def gen_region(seq_len, freqs):
    """Independent base sampling (simple fallback method)."""
    bases = list(freqs.keys())
    probs = [freqs[b] for b in bases]
    return ''.join(np.random.choice(bases, seq_len, p=probs))

def build_simple_dinuc_probs(freqs, cpg_multiplier=1.0):
    """
    Build a simple dinucleotide transition matrix P(next | current) from single-base frequencies.
    Multiply C->G (CpG) by cpg_multiplier to tune CpG abundance (>1 up, <1 down).
    """
    bases = ['A','C','G','T']
    dinuc = {}
    for cur in bases:
        weights = {}
        for nxt in bases:
            w = freqs.get(nxt, 0.25)
            # Apply C->G adjustment factor
            if cur == 'C' and nxt == 'G':
                w *= cpg_multiplier
            weights[nxt] = w
        s = sum(weights.values())
        # Normalize to conditional probabilities
        dinuc[cur] = {b: (weights[b] / s) for b in bases}
    return dinuc

def gen_region_markov(seq_len, dinuc_probs, start_base=None):
    """Generate sequence using a dinucleotide Markov model (P(next|current))."""
    if seq_len <= 0:
        return ""
    bases = ['A','C','G','T']
    if start_base is None:
        # Choose start base (uniform here; could use marginal frequencies)
        start_base = random.choice(bases)
    out = [start_base]
    cur = start_base
    for _ in range(seq_len - 1):
        probs = dinuc_probs.get(cur)
        if not probs:
            out.append(random.choice(bases))
            cur = out[-1]
            continue
        choices, weights = zip(*probs.items())
        out.append(random.choices(choices, weights=weights, k=1)[0])
        cur = out[-1]
    return ''.join(out)

def insert_repeats(seq, fraction=0.01, motifs=None):
    """
    Randomly insert repeat fragments into the sequence.
    motifs: pass REPEAT_MOTIFS or a custom repeat library
    """
    if motifs is None:
        motifs = REPEAT_MOTIFS
    if fraction <= 0 or len(seq) == 0:
        return seq
    total = len(seq)
    n_ins = max(0, int(total * fraction / 50))
    seq_list = list(seq)
    protected_windows = []
    for _ in range(n_ins):
        pos = random.randrange(0, len(seq_list) + 1)
        motif = random.choice(motifs)
        # Avoid dense insertions: simple protection window
        win = (max(0,pos-100), min(len(seq_list), pos+100))
        overlap = any(not (win[1] < p[0] or win[0] > p[1]) for p in protected_windows)
        if overlap:
            continue
        seq_list[pos:pos] = list(motif)
        protected_windows.append(win)
    return ''.join(seq_list)

def strengthen_cpg_island_safe(region, target_cpg_density=0.02, max_attempts_per_kb=50):
    L = len(region)
    if L < 2:
        return region
    cur_cpg = sum(1 for i in range(L-1) if region[i:i+2] == "CG")
    cur_density = cur_cpg / max(1, L)
    if cur_density >= target_cpg_density:
        return region
    needed = int((target_cpg_density - cur_density) * L)
    needed = max(1, needed)
    region_list = list(region)
    attempts = 0
    inserted = 0
    while inserted < needed and attempts < max(1, int(max_attempts_per_kb * (L/1000.0))):
        p = random.randrange(0, L-1)
        if region_list[p] == 'C' and region_list[p+1] == 'G':
            attempts += 1
            continue
        region_list[p] = 'C'
        region_list[p+1] = 'G'
        inserted += 1
        attempts += 1
    return ''.join(region_list)

def insert_microsat(seq, density=MICROSAT_DENSITY):
    seq_list = list(seq)
    i = 0
    motifs = ["A", "CA", "GATA", "TCT"]
    while i < len(seq_list):
        if random.random() < density:
            motif = random.choice(motifs)
            nrep = random.randint(4, 15)
            seq_list[i:i] = list(motif * nrep)
            i += len(motif) * nrep
        i += 1
    return ''.join(seq_list)

# ---------------- Statistics helpers ----------------
def gc_content(seq):
    if not seq:
        return 0.0
    c = seq.count('C') + seq.count('G')
    return c / len(seq)

def cpg_obs_exp(seq):
    if not seq:
        return 0.0
    obs = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == "CG")
    pC = seq.count('C') / len(seq)
    pG = seq.count('G') / len(seq)
    exp = pC * pG * (len(seq) - 1)
    return (obs / exp) if exp > 0 else 0.0

def basic_report(seq, params):
    report = []
    report.append(f"Length: {len(seq)}")
    gc = gc_content(seq)
    report.append(f"GC%: {gc*100:.2f}")
    report.append(f"CpG obs/exp: {cpg_obs_exp(seq):.3f}")
    k = 4
    if len(seq) >= k:
        kmers = Counter(seq[i:i+k] for i in range(len(seq)-k+1))
        top = kmers.most_common(5)
        report.append("Top 4-mers: " + ", ".join(f"{k}:{v}" for k, v in top))
    report.append("Parameters (json):")
    report.append(json.dumps(params, indent=2))
    return "\n".join(report)

def load_repeats_from_fasta(fasta_path):
    """
    Load repeat library from FASTA; return sequences of A/C/G/T with length >= 4.
    """
    seqs = []
    p = Path(fasta_path)
    if not p.exists():
        return seqs
    with p.open('r', encoding='utf-8') as f:
        cur = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur:
                    s = ''.join(cur).upper()
                    if len(s) >= 4 and all(ch in "ACGT" for ch in s):
                        seqs.append(s)
                    cur = []
            else:
                cur.append(line)
        if cur:
            s = ''.join(cur).upper()
            if len(s) >= 4 and all(ch in "ACGT" for ch in s):
                seqs.append(s)
    return seqs

def train_dinuc_from_sequence(seq, pseudocount=1):
    """
    Count dinucleotide transitions from a long sequence and return P(next | current).
    Add pseudocounts to avoid zero probabilities.
    """
    bases = ['A','C','G','T']
    counts = {b:{b2:pseudocount for b2 in bases} for b in bases}
    seq = seq.upper()
    for i in range(len(seq)-1):
        a, b = seq[i], seq[i+1]
        if a in counts and b in counts[a]:
            counts[a][b] += 1
    # Normalize to conditional probabilities
    dinuc = {}
    for a in bases:
        s = sum(counts[a].values())
        dinuc[a] = {b: counts[a][b] / s for b in bases}
    return dinuc

def compute_gc_window_stats(seq, window=1000, step=500):
    """
    Compute sliding-window GC stats: mean, std, min, max, list (for output/plotting).
    """
    n = len(seq)
    if n == 0:
        return {'mean':0,'std':0,'min':0,'max':0,'values':[]}
    vals = []
    i = 0
    while i < n:
        w = seq[i:i+window]
        if len(w) >= 1:
            gc = (w.count('G') + w.count('C')) / len(w)
            vals.append(gc)
        i += step
    if not vals:
        return {'mean':0,'std':0,'min':0,'max':0,'values':[]}
    mean = sum(vals)/len(vals)
    var = sum((x-mean)**2 for x in vals)/len(vals)
    return {'mean': mean, 'std': math.sqrt(var), 'min': min(vals), 'max': max(vals), 'values': vals}

def shannon_entropy(seq):
    """Compute Shannon entropy (bits) from single-base frequencies."""
    if not seq:
        return 0.0
    L = len(seq)
    cnt = Counter(seq)
    ent = 0.0
    for k,v in cnt.items():
        p = v / L
        ent -= p * math.log2(p)
    return ent

# ---------------- Assembly function (Markov / fallback) ----------------
def assemble_chromosome(target_len=100000, seed=None,
                        state_weights=None,
                        repeat_fraction=REPEAT_FRACTION,
                        microsat_density=MICROSAT_DENSITY,
                        cpg_target_density=CPG_TARGET_DENSITY,
                        states=None,
                        repeat_motifs=None,
                        repeat_fasta=None,
                        train_seq_path=None):
    """
    Extension: if repeat_fasta is provided, load repeats from FASTA to replace repeat_motifs;
    if train_seq_path is provided, train a dinucleotide matrix from that sequence.
    """
    if states is None:
        states = STATES
    if repeat_motifs is None:
        repeat_motifs = REPEAT_MOTIFS
    if state_weights is None:
        state_weights = STATE_WEIGHTS

    # Use real repeat library if available
    repeat_motifs = sanitize_repeat_motifs(repeat_motifs)
    if repeat_fasta:
        loaded = load_repeats_from_fasta(repeat_fasta)
        if loaded:
            repeat_motifs = loaded

    # Reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    state_keys = list(states.keys())
    weights = [state_weights.get(k, 0.0) for k in state_keys]
    cur_len = 0
    out = []

    # If training sequence is provided, use it to build base dinuc matrix
    base_dinuc = {}
    if train_seq_path:
        tp = Path(train_seq_path)
        if tp.exists():
            with tp.open('r', encoding='utf-8') as tf:
                lines = [ln.strip() for ln in tf if ln.strip() and not ln.startswith('>')]
                train_seq = ''.join(lines).upper()
            base_dinuc = train_dinuc_from_sequence(train_seq)
        else:
            base_dinuc = None

    # Build dinuc matrix per state (prefer trained matrix; fallback to single-base frequencies)
    dinuc_map = {}
    for k, v in states.items():
        if base_dinuc:
            # Copy trained matrix and adjust C->G
            mult = 1.0
            if k == "cpg_island":
                mult = 3.0
            elif k == "gc_rich":
                mult = 1.2
            elif k == "gene":
                mult = 1.0
            else:
                mult = 0.8
            # Copy and adjust
            m = {}
            for cur, probs in base_dinuc.items():
                m[cur] = dict(probs)  # shallow copy
            # apply C->G multiplier and renormalize each row
            for cur in m:
                if 'G' in m[cur]:
                    m[cur]['G'] *= mult
                s = sum(m[cur].values())
                if s > 0:
                    for b in m[cur]:
                        m[cur][b] = m[cur][b] / s
            dinuc_map[k] = m
        else:
            # Fallback: build from single-base frequencies
            if k == "cpg_island":
                mult = 3.0
            elif k == "gc_rich":
                mult = 1.2
            elif k == "gene":
                mult = 1.0
            else:
                mult = 0.6
            dinuc_map[k] = build_simple_dinuc_probs(v["freq"], cpg_multiplier=mult)

    # Main loop (same as before)
    while cur_len < target_len:
        state = random.choices(state_keys, weights=weights, k=1)[0]
        s = states[state]
        L = sample_state_length(s["mean_len"])
        L = min(L, target_len - cur_len)
        if state in ("gc_rich", "cpg_island", "gene"):
            region = gen_region_markov(L, dinuc_map[state])
        else:
            region = gen_region(L, s["freq"])
        if state == "repeat":
            region = insert_repeats(region, fraction=0.5, motifs=repeat_motifs)
        if state == "cpg_island":
            region = strengthen_cpg_island_safe(region, target_cpg_density=cpg_target_density)
        region = insert_microsat(region, density=microsat_density if state != "repeat" else microsat_density*3)
        out.append(region)
        cur_len += len(region)

    chrom = ''.join(out)[:target_len]
    chrom = insert_repeats(chrom, fraction=repeat_fraction, motifs=repeat_motifs)
    chrom = insert_microsat(chrom, density=microsat_density)
    return chrom

# ---------------- Output and main ----------------
def write_fasta(seq, path, name="synthetic_chr1", width=80):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), width):
            f.write(seq[i:i+width] + "\n")

def safe_write_fasta(seq, path, name="synthetic_chr1", width=80):
    """
    Write sequence to FASTA; path must be under BASE_OUTPUT_DIR or an exception is raised.
    If only a filename is given, default to d:/UCAS/genemodel (use forward slashes to avoid \U escapes).
    """
    p = Path(path)
    # If only filename or relative path, write under BASE_OUTPUT_DIR/OUTPUT_SUBDIR
    if not p.parent or str(p.parent) in ('.', ''):
        p = Path(BASE_OUTPUT_DIR) / OUTPUT_SUBDIR / p.name
    # Get and validate safe path
    safe_path = _ensure_safe_path(str(p))
    parent = Path(safe_path).parent
    parent.mkdir(parents=True, exist_ok=True)
    with open(safe_path, 'w', encoding='utf-8') as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), width):
            f.write(seq[i:i+width] + "\n")

if __name__ == "__main__":
    # Check whether reference files exist (under d:/UCAS/genemodel/)
    base_dir = Path(BASE_OUTPUT_DIR) / OUTPUT_SUBDIR
    repeats_fasta_path = base_dir / "repeats.fa"
    train_seq_path = base_dir / "train_seq.fa"

    use_repeats = repeats_fasta_path.exists()
    use_train = train_seq_path.exists()

    if use_repeats:
        print(f"Using repeats library: {repeats_fasta_path}")
    else:
        print("repeats.fa not found; using built-in REPEAT_MOTIFS")

    if use_train:
        print(f"Using training sequence: {train_seq_path}")
    else:
        print("train_seq.fa not found; using default/frequency-based dinuc matrix")

    # Call generator; pass existing reference file paths when available
    seq = assemble_chromosome(
        TARGET_LEN,
        seed=SEED,
        state_weights=STATE_WEIGHTS,
        repeat_fraction=REPEAT_FRACTION,
        microsat_density=MICROSAT_DENSITY,
        cpg_target_density=CPG_TARGET_DENSITY,
        states=STATES,
        repeat_motifs=REPEAT_MOTIFS,
        repeat_fasta=str(repeats_fasta_path) if use_repeats else None,
        train_seq_path=str(train_seq_path) if use_train else None
    )

    # Write FASTA and report (safe subdir)
    fasta_name = "synthetic_chr.fa"
    safe_write_fasta(seq, fasta_name, name="synthetic_chr1")

    params = {
        "TARGET_LEN": TARGET_LEN,
        "SEED": SEED,
        "STATE_WEIGHTS": STATE_WEIGHTS,
        "REPEAT_FRACTION": REPEAT_FRACTION,
        "MICROSAT_DENSITY": MICROSAT_DENSITY,
        "CPG_TARGET_DENSITY": CPG_TARGET_DENSITY,
        "STATES": STATES,
        "REPEAT_MOTIFS_used": REPEAT_MOTIFS,
        "repeats_fasta_used": str(repeats_fasta_path) if use_repeats else None,
        "train_seq_used": str(train_seq_path) if use_train else None
    }

    # Generate and write report
    report_basic = basic_report(seq, params)
    gc_stats = compute_gc_window_stats(seq, window=1000, step=500)
    ent = shannon_entropy(seq)
    dinuc_counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    top_dinucs = dinuc_counts.most_common(10)
    extra = [
        f"GC window mean: {gc_stats['mean']:.4f}, std: {gc_stats['std']:.4f}, min: {gc_stats['min']:.4f}, max: {gc_stats['max']:.4f}",
        f"Shannon entropy (bits): {ent:.4f}",
        "Top dinucleotides: " + ", ".join(f"{k}:{v}" for k,v in top_dinucs)
    ]
    full_report = report_basic + "\n\n" + "\n".join(extra)

    report_path = base_dir / "generator_report.txt"
    report_safe = _ensure_safe_path(str(report_path))
    Path(report_safe).parent.mkdir(parents=True, exist_ok=True)
    with open(report_safe, "w", encoding="utf-8") as rf:
        rf.write(full_report + "\n")

    print(f"Generation complete: {base_dir / fasta_name}")
    print(f"Report saved: {report_safe}")
