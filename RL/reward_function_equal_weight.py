import re
from Bio import pairwise2
from Bio.Seq import Seq
from difflib import SequenceMatcher
from Bio.Align import PairwiseAligner
from collections import Counter
from collections import defaultdict


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def align_and_calculate_similarity(seq1, seq2):
    """
    Perform global alignment of two DNA sequences and compute similarity.

    Args:
    seq1 (str): First DNA sequence
    seq2 (str): Second DNA sequence

    Returns:
    float: Sequence similarity (0-1)
    str: Formatted alignment string
    """
    # Global alignment with default params (match=1, mismatch=-1, gap=-1)
    alignments = pairwise2.align.globalxx(seq1, seq2)

    # Get best alignment
    best_alignment = alignments[0]

    # Extract aligned sequences
    aligned_seq1 = best_alignment.seqA
    aligned_seq2 = best_alignment.seqB

    # Count matching bases
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)

    # Compute similarity
    similarity = matches / len(aligned_seq1)

    return similarity


def dense_reward(pred: str, truth: str,
                 match=1.0, mismatch=-0.2, gap=-0.2, len_pen_coeff=0.3):
    if not truth:
        return -1.0
    if not pred:
        return -1.0

    # 1. Build equal-length aligned strings (with '-' gaps)
    sm = SequenceMatcher(None, pred, truth)

    align_pred, align_truth = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            align_pred.append(pred[i1:i2])
            align_truth.append(truth[j1:j2])
        elif tag == 'delete':
            align_pred.append(pred[i1:i2])
            align_truth.append('-' * (i2 - i1))
        elif tag == 'insert':
            align_pred.append('-' * (j2 - j1))
            align_truth.append(truth[j1:j2])
        elif tag == 'replace':
            # Pad replace block to equal length with '-'
            ln = max(i2 - i1, j2 - j1)
            align_pred.append((pred[i1:i2]).ljust(ln, '-'))
            align_truth.append((truth[j1:j2]).ljust(ln, '-'))

    align_pred = ''.join(align_pred)
    align_truth = ''.join(align_truth)

    # 2. Score per character (each truth position rewarded at most once)
    rewards = []
    truth_done = set()          # truth indices already scored
    t_idx = 0                   # current truth position

    for p, t in zip(align_pred, align_truth):
        if t == '-':            # insertion
            rewards.append(gap)
            continue
        # truth position
        if t_idx in truth_done:
            rewards.append(0.0)
        else:
            if p == '-':        # deletion
                rewards.append(gap)
            elif p == t:        # match
                rewards.append(match)
            else:               # mismatch
                rewards.append(mismatch)
            truth_done.add(t_idx)
        t_idx += 1

    # 3. Length penalty
    len_pen = abs(len(pred) - len(truth))
    rewards.append(-len_pen_coeff * len_pen)

    return sum(rewards) / len(truth)


class AlignTool:
    """Unified global/local alignment; parameters configured here."""
    # --------- global alignment params ---------
    match_g    = 2
    mismatch_g = -1
    open_gap_g = -2
    extend_g   = -0.5

    # --------- local alignment params ---------
    match_l    = 2
    mismatch_l = -1
    open_gap_l = -2
    extend_l   = -0.5

    # --------- aligner instances ---------
    _global = PairwiseAligner()
    _global.mode = 'global'
    _global.match_score        = match_g
    _global.mismatch_score     = mismatch_g
    _global.open_gap_score     = open_gap_g
    _global.extend_gap_score   = extend_g

    _local = PairwiseAligner()
    _local.mode = 'local'
    _local.match_score        = match_l
    _local.mismatch_score     = mismatch_l
    _local.open_gap_score     = open_gap_l
    _local.extend_gap_score   = extend_l

    # --------- thin wrappers ---------
    @staticmethod
    def similarity(seq1: str, seq2: str) -> float:
        aln = AlignTool._global.align(seq1, seq2)[0]
        matches = sum(a == b for a, b in zip(aln[0], aln[1]))
        return matches / len(aln[0])

    @staticmethod
    def edit_distance(a: str, b: str) -> int:
        aln = AlignTool._global.align(a, b)[0]
        matches = sum(a == b for a, b in zip(aln[0], aln[1]))
        return len(aln[0]) - matches

    @staticmethod
    def coords(read: str, pred: str) -> tuple[int, int]:
        """Global alignment: read mapped on pred (start, end)."""
        aln = AlignTool._global.align(read, pred)[0]
        seg = aln.aligned[1]
        if seg.size == 0:
            return -1, -1
        start = int(seg[0, 0])
        end = int(seg[-1, 1]) - 1
        return start, end

    @staticmethod
    def coords_local(read: str, pred: str, min_identity: float = 0.8) -> tuple[int, int]:
        """Local alignment + identity threshold; return (-1, -1) if not met."""
        aln = AlignTool._local.align(read, pred)[0]
        seg = aln.aligned[1]
        if seg.size == 0:
            return -1, -1
        start = int(seg[0, 0])
        end = int(seg[-1, 1]) - 1
        # identity threshold
        matches = sum(a == b for a, b in zip(aln[0], aln[1]))
        identity = matches / len(aln[0])
        return (start, end) if identity >= min_identity else (-1, -1)


# ---------- format ----------
def sanitize_and_penalize(seq: str):
    """
    Return (clean_seq, fmt_penalty).
    Keep only ATCG characters; delete everything else and penalize by the count removed.
    """
    clean = re.sub(r'[^ATCG]', '', seq.upper())          # 1) remove invalid
    illegal_count = len(seq) - len(clean)               # 2) count

    return clean, illegal_count


# ---------- loop ----------
def loop_penalty_anystart(seq: str, min_unit: int = 200):
    """
    Penalty: from the start of the "longest/most frequent" 200-mer to the end.
    Returns: (penalty_len, (start, end, loop_seq))
    """
    n = len(seq)
    if n < min_unit * 2:
        return 0.0, None

    BASE_CODE = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    pow_w = 4 ** (min_unit - 1)

    counter = defaultdict(int)
    first_pos = {}
    last_pos = {}

    h = 0
    for i in range(min_unit):
        h = h * 4 + BASE_CODE[seq[i]]
    counter[h] = 1
    first_pos[h] = 0
    last_pos[h] = min_unit

    for i in range(1, n - min_unit + 1):
        out_b = BASE_CODE[seq[i - 1]]
        in_b = BASE_CODE[seq[i + min_unit - 1]]
        h = (h - out_b * pow_w) * 4 + in_b

        counter[h] += 1
        if h not in first_pos:
            first_pos[h] = i
        last_pos[h] = i + min_unit

    # Pick the segment with highest count and longest length
    best = max(((h, cnt, last_pos[h] - first_pos[h])
                for h, cnt in counter.items() if cnt >= 2),
               default=(None, 0, 0), key=lambda x: (x[1], x[2]))
    if best[0] is None:
        return 0.0, None

    start = first_pos[best[0]]
    loop_seq = seq[start:start + 200]

    # Penalty length: from this segment start to the end
    loop_len = n - start
    return loop_len, (start, start + 200, loop_seq)


# ---------- short ----------
def detect_repeat_shortcut(pred: str, gt: str, gap_min=40):
    """
    Only count when the short segment before a missing start appears >=2 times in GT -> 1 shortcut.
    """
    if len(gt) < gap_min + 100:
        return 0

    # 1. Traverse all gaps (any length)
    aln = AlignTool._global.align(gt, pred)[0]
    gaps = []  # (start, end) collect all
    prev_gt_end = 0
    for seg in aln.aligned[0]:
        gt_s, gt_e = int(seg[0]), int(seg[1])
        if gt_s > prev_gt_end:  # record any missing region
            gaps.append((prev_gt_end, gt_s))
        prev_gt_end = gt_e
    # tail
    if len(gt) > prev_gt_end:
        gaps.append((prev_gt_end, len(gt)))

    # 2. Check all gaps for "repeat-induced" events
    def _multi_times_gap(gap_start, gap_end, seq, k_min=10, k_max=100, step=4):
        """Return True if any k-mer in gap region (before/inside/after) appears >=2 times."""
        # 1. before 4-100 bp
        if gap_start >= k_max:
            for k in range(k_max, k_min - 1, -step):
                start = gap_start - k
                guide = seq[start:gap_start]
                if seq.count(guide) >= 2:
                    return True

        # 2. inside: missing segment itself (<=k_max scan all; >k_max use head/tail k_max)
        gap_len = gap_end - gap_start
        if gap_len <= k_max:
            for k in range(gap_len, k_min - 1, -step):
                guide = seq[gap_start:gap_start + k]
                if seq.count(guide) >= 2:
                    return True
        else:
            # head and tail k_max
            for k in range(k_max, k_min - 1, -step):
                if seq[gap_start:gap_start + k].count(seq[gap_start:gap_start + k]) >= 2:
                    return True
                if seq[gap_end - k:gap_end].count(seq[gap_end - k:gap_end]) >= 2:
                    return True

        # 3. after 4-100 bp
        if gap_end + k_max <= len(seq):
            for k in range(k_max, k_min - 1, -step):
                guide = seq[gap_end:gap_end + k]
                if seq.count(guide) >= 2:
                    return True

        return False

    # 2. Check all gaps for "repeat-induced" events
    short_cnt = 0
    prev_gap_end = 0  # for adjacent de-duplication
    for g_st, g_en in gaps:
        # 1. De-dup adjacent gaps: too close to previous gap end -> skip (except the first)
        if prev_gap_end != 0 and g_st - prev_gap_end < gap_min:
            continue
        # 2. Actual shortcut detection
        if _multi_times_gap(g_st, g_en, gt):
            short_cnt += 1
        # 3. Update reference point
        prev_gap_end = g_en

    return short_cnt


# ---------- copy num ----------
def copy_num_error_slide(pred: str, gt: str,
                         frag_len=24, min_copy=2,
                         frags_per_kb=40):          # <- auto density
    """
    Sliding window across full GT, dedupe and keep fragments appearing >=min_copy in GT.
    Then sample systematically at frags_per_kb per 1 kb of GT,
    compute |gt count - pred count| penalty and normalize to 0-1.
    """
    from collections import OrderedDict

    L = len(gt)
    if L < frag_len:
        return 0.0

    # 1. Sliding window + dedupe (OrderedDict preserves order)
    unique = OrderedDict()
    for i in range(L - frag_len + 1):
        unique[gt[i:i + frag_len]] = None

    # 2. Keep only fragments repeated in ground truth
    candidates = [f for f in unique if gt.count(f) >= min_copy]
    if not candidates:
        return 0.0

    # 3. Auto compute target count: frags_per_kb per 1 kb
    target_n = max(1, int(L / 1000 * frags_per_kb))
    if len(candidates) > target_n:
        step = max(1, len(candidates) // target_n)
        candidates = candidates[::step][:target_n]

    penalty = 0.0
    for frag in candidates:
        exp = gt.count(frag)
        obs = pred.count(frag)
        err = abs(exp - obs)
        M = max(exp, 1)
        # penalty += (err / M) ** 2
        penalty += err / M

    return penalty / len(candidates)


# ---------- coverage ----------
def coverage_f1_kmergt(pred: str, reads: list[str], gt: str,
                       block: int = 80,
                       min_cov_ratio: float = 0.8,
                       min_identity: float = 0.8,
                       max_align_ratio: float = 1.1,
                       beta: float = 1.0,
                       k: int = 25,
                       min_share: int = 3):
    n = len(pred)
    if n == 0 or not reads or not gt:
        return 0.0

    blocks = (n + block - 1) // block
    block_cov = [False] * blocks
    total = 0
    hit = 0

    # 1. Ground-truth k-mer set (O(L))
    ref_kmers = set(gt[i:i+k] for i in range(len(gt)-k+1))

    # 2. Count per read (O(R×(L-k+1)))
    for i, read in enumerate(reads):
        share = len({read[j:j+k] for j in range(len(read)-k+1)} & ref_kmers)

        if share < min_share:
            continue

        total += 1

        # === 3. Align to pred (local + identity) ===
        start, end = AlignTool.coords_local(read, pred, min_identity=min_identity)
        if start == -1:  # no alignment or identity too low
            continue
        aligned_len = end - start + 1
        # length vs read length threshold
        if aligned_len > len(read) * max_align_ratio or aligned_len < len(read) * min_cov_ratio:
            continue

        hit += 1
        b_start = start // block
        b_end = (end + 1 + block - 1) // block
        for b in range(b_start, b_end):
            if b < blocks:
                block_cov[b] = True

    prec = sum(block_cov) / blocks if blocks else 0
    rec = hit / total if total else 0
    if prec + rec == 0:
        return 0.0
    f1 = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
    return f1


# ---------- reward ----------
def reward_function_equal(pred_seq_str: str, gt_seq_str: str, reads_list: list[str]) -> float:
    # 1) format penalty
    pred_clean, illegals = sanitize_and_penalize(pred_seq_str)  #
    fmt_pen = -1 * illegals
    print('fmt_pen :', fmt_pen)

    # 2) loop penalty
    loop_len, loop = loop_penalty_anystart(pred_clean, min_unit=200)  #
    loop_pen = -1 * loop_len / len(gt_seq_str)  #
    print('loop_pen :', loop_pen)

    # 3) short penalty
    short_cnt = detect_repeat_shortcut(pred_clean, gt_seq_str, gap_min=40)
    short_pen = -1 * short_cnt
    print('short_pen :', short_pen)

    # 4) copy num distance penalty
    copy_num_err = copy_num_error_slide(pred_clean, gt_seq_str, frag_len=15, frags_per_kb=1000)
    copy_num_pen = -1 * copy_num_err
    print('copy_num_pen :', copy_num_pen)

    # 5) edit distance penalty
    similarity = AlignTool.similarity(pred_clean, gt_seq_str)  # range is 0-1
    ed_pen = -1 * (1 - similarity) #
    print('ed_pen :', ed_pen)

    # 6) coverage F1 is in the range 0-1
    cov_r = 1 * coverage_f1_kmergt(pred_clean, reads_list, gt_seq_str,  block=50, min_cov_ratio=0.75, max_align_ratio=1.1, beta=1.0)
    print('basic_reward :', cov_r)  #

    return cov_r + fmt_pen + loop_pen + short_pen + copy_num_pen + ed_pen


# Reward functions
# def compute_score(data_source,solution_str,ground_truth,extra_info):
#     score = 0.0
#     sol_str = extract_xml_answer(solution_str).replace(" ", "").replace("\n", "").replace("\t", "").upper()
#     if sol_str == ground_truth.upper():
#         return 1.0
#     else:
#         for seq in extra_info['dna_sequence']:
#             if seq in solution_str:
#                 score += 1
#             else:
#                 score -= 0.5
#         return score/len(extra_info['dna_sequence']) - 0.1*abs(len(sol_str)-len(ground_truth))


# def compute_score(data_source, solution_str, ground_truth, extra_info):
#     score = 0.0
#     sol_str = extract_xml_answer(solution_str).replace(" ", "").replace("\n", "").replace("\t", "").upper()
#
#     # exactly correct
#     if sol_str == ground_truth.upper():
#         return 1.0
#     # partially correct
#     else:
#         try:
#             similarity = align_and_calculate_similarity(ground_truth.upper(), sol_str)
#             return similarity
#         except:
#             return score


# def compute_score(data_source, solution_str, ground_truth, extra_info):
#     score = 0.0
#     sol_str = extract_xml_answer(solution_str).replace(" ", "").replace("\n", "").replace("\t", "").upper()
#
#     # exactly correct
#     if sol_str == ground_truth.upper():
#         return 1.0
#     # partially correct
#     else:
#         try:
#             score = dense_reward(sol_str, ground_truth.upper(), len_pen_coeff=2.0)
#             return score
#         except:
#             return score


def compute_score(data_source, solution_str, ground_truth, extra_info):
    score = 0.0
    sol_str = extract_xml_answer(solution_str).replace(" ", "").replace("\n", "").replace("\t", "").upper()

    # exactly correct
    if sol_str == ground_truth.upper():
        return 1.0
    # partially correct
    else:
        try:
            score = reward_function_equal(sol_str, ground_truth.upper(), extra_info['dna_sequence'])
            return score
        except:
            return score
