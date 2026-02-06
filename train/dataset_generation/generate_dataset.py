#!/usr/bin/env python3
"""
Dataset generator - integrates generator_complex.py, split.py, and add_noise.py.
Generates training data for the sequence assembly task.
"""

import json
import random
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import tempfile
import os
import hashlib

# Import functionality from existing modules
import sys
sys.path.append('.')

# Import generation function from generator_complex.py
from generator_complex import assemble_chromosome

def generate_complex_sequence(length: int = 2000, seed: int = None) -> str:
    """
    Generate a complex human-like genome sequence.
    Directly uses assemble_chromosome from generator_complex.py with dataset-friendly params.
    """
    # Hyperparameters suitable for dataset generation
    state_weights = {
        "intergenic": 0.60,
        "gc_rich": 0.20,
        "cpg_island": 0.05,
        "repeat": 0.10,
        "gene": 0.05
    }
    
    states = {
        "intergenic": {"freq": {"A": 0.30, "C": 0.20, "G": 0.20, "T": 0.30}, "mean_len": 1000},
        "gc_rich": {"freq": {"A": 0.20, "C": 0.30, "G": 0.30, "T": 0.20}, "mean_len": 500},
        "cpg_island": {"freq": {"A": 0.18, "C": 0.32, "G": 0.32, "T": 0.18}, "mean_len": 200},
        "repeat": {"freq": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}, "mean_len": 300},
        "gene": {"freq": {"A": 0.28, "C": 0.22, "G": 0.22, "T": 0.28}, "mean_len": 800}
    }
    
    # assemble_chromosome may return a sequence string or (sequence, repeat_info)
    result = assemble_chromosome(
        target_len=length, 
        seed=seed,
        state_weights=state_weights,
        states=states,
        repeat_fraction=0.02,  # repeat element fraction
        microsat_density=0.0004,  # microsatellite density
        cpg_target_density=0.02  # CpG island density
    )
    
    # Handle possible return formats
    if isinstance(result, tuple):
        sequence, _ = result
    else:
        sequence = result
    
    return sequence


def shuffle_copy_paste(seq: str,
                       min_len: int,
                       max_len: int,
                       p_geometric: float = 0.7,     # geometric continuation probability
                       p_do_copypaste: float = 0.8,  # probability to do copy-paste
                       seed: int = None) -> Tuple[str, List[str], int]:
    """
    Return (new_seq, list of copied substrings, actual number of operations).
    """
    if seed is not None:
        random.seed(seed)

    L = len(seq)
    s = list(seq)
    copied_chunks: List[str] = [] 

    # 0. Decide whether to do 0 operations
    if random.random() > p_do_copypaste:
        return seq, copied_chunks, 0

    # 1. Sample number of ops >= 1 (truncated geometric)
    max_ops = max(1, L // min_len)
    ops = 1
    while random.random() > p_geometric and ops < max_ops:
        ops += 1

    # 2. Perform ops copy-pastes
    for _ in range(ops):
        length = random.randint(min_len, max_len)
        src = random.randint(0, L - length)
        dst = random.randint(0, L - length)
        chunk = ''.join(s[src:src + length])   # copied fragment
        copied_chunks.append(chunk)
        s[dst:dst + length] = s[src:src + length]

    return ''.join(s), copied_chunks, ops


# Import split functionality from split.py
def split_sequence(sequence: str, coverage: int = 30, read_length_min: int = 50, read_length_max: int = 150) -> List[str]:
    """
    Split a sequence into overlapping fragments.
    Based on the logic in split.py.
    """
    if not sequence:
        return []
    
    # Parameter setup
    piece_length_lower = read_length_min
    piece_length_upper = read_length_max
    piece_length_avg = (piece_length_lower + piece_length_upper) // 2
    
    input_length = len(sequence)
    pieces = []
    covered = [0] * input_length
    
    # Compute number of fragments needed to reach target coverage
    expected_pieces = (coverage * input_length) // piece_length_avg
    
    # Generate main fragments
    for _ in range(expected_pieces):
        if input_length <= piece_length_lower:
            pieces.append(sequence)
            break
            
        # Randomly choose fragment length
        piece_length = random.randint(piece_length_lower, piece_length_upper)
        
        # Randomly choose a start position within the full sequence
        start_index = random.randint(0, input_length - 1)
        
        # Compute max available length from this position to the end
        max_length = input_length - start_index
        
        # If remaining length is too short, try shifting start backward
        if max_length < piece_length_lower:
            # Shift start backward to satisfy minimum length
            start_index = max(0, input_length - piece_length_lower)
            piece_length = piece_length_lower
        else:
            # If chosen length exceeds remaining length, adjust length
            if piece_length > max_length:
                piece_length = max_length
                
        pieces.append(sequence[start_index:start_index + piece_length])
        
        # Mark covered region
        for i in range(start_index, start_index + piece_length):
            if i < input_length:
                covered[i] = 1
    
    # Fill uncovered regions (left-to-right)
    i = 0
    while i < input_length:
        if covered[i] == 0:  # found an uncovered position
            start_index = i
            max_length = input_length - start_index
            
            if max_length < piece_length_lower:
                # If remaining length is too short, shift start backward
                start_index = max(0, input_length - piece_length_lower)
                piece_length = piece_length_lower
            else:
                piece_length = random.randint(piece_length_lower, min(piece_length_upper, max_length))
                
            pieces.append(sequence[start_index:start_index + piece_length])
            
            # Mark covered region
            for j in range(start_index, start_index + piece_length):
                if j < input_length:
                    covered[j] = 1
            
            # Skip covered region
            i = start_index + piece_length
        else:
            i += 1
    
    # Shuffle all fragments
    random.shuffle(pieces)
    
    return pieces

# Modify the last few bp of a sequence
def modify_fragment_tail_all(fragment: str, tail_len: int) -> str:
    """
    Randomly substitute the last tail_len bases of a fragment
    (each base differs from the original).
    """
    if tail_len <= 0 or not fragment:
        return fragment

    nucleotides = ['A', 'T', 'C', 'G']
    frag = list(fragment)
    L = len(frag)
    start = max(0, L - tail_len)

    for p in range(start, L):
        original = frag[p]
        frag[p] = random.choice([b for b in nucleotides if b != original])

    return ''.join(frag)


def add_noise_to_fragments(
    fragments: List[str],
    noise_ratio: float = 0.2,
    modification_ratio: float = 0.2,
    drop_original: bool = False,      # new flag
) -> List[str]:
    """
    Add noise fragments to a fragment list.
    keep_original=True  -> legacy behavior: keep originals, append noise, then shuffle
    keep_original=False -> drop originals: randomly modify a noise_ratio of fragments in-place, then shuffle
    """
    if not fragments or noise_ratio <= 0:
        return fragments

    original_count = len(fragments)

    # Number of fragments to modify
    if not drop_original:
        # Equivalent to previous logic: generate extra noise_count fragments
        if noise_ratio < 1.0:
            noise_count = int(noise_ratio * original_count / (1 - noise_ratio))
        else:
            noise_count = original_count
        # Generate noise
        noise_fragments = [
            modify_fragment(random.choice(fragments), modification_ratio)
            for _ in range(noise_count)
        ]
        merged = fragments + noise_fragments
    else:
        # New logic: randomly modify fragments in-place
        if noise_ratio < 1.0:
            modify_count = int(original_count * noise_ratio)
        else:
            modify_count = original_count
        idx_to_modify = random.sample(range(original_count), modify_count)
        merged = fragments.copy()
        for i in idx_to_modify:
            merged[i] = modify_fragment(fragments[i], modification_ratio)

    random.shuffle(merged)
    return merged


def modify_fragment(fragment: str, modification_ratio: float) -> str:
    """
    Modify a specified ratio of bases in a fragment.
    """
    nucleotides = ['A', 'T', 'C', 'G']
    fragment_list = list(fragment)
    fragment_length = len(fragment)

    # Compute number of bases to modify
    num_changes = int(fragment_length * modification_ratio)

    if num_changes == 0:
        return fragment

    # Randomly choose positions to modify
    positions_to_change = random.sample(range(fragment_length), min(num_changes, fragment_length))

    # Modify bases at selected positions
    for pos in positions_to_change:
        original_base = fragment_list[pos]
        # Choose a random base different from the original
        available_bases = [base for base in nucleotides if base != original_base]
        fragment_list[pos] = random.choice(available_bases)

    return ''.join(fragment_list)


def generate_random_fragments(count: int, length: int = 100) -> List[str]:
    """
    Generate completely random DNA fragments.
    """
    nucleotides = ['A', 'T', 'C', 'G']
    fragments = []
    
    for _ in range(count):
        fragment = ''.join(random.choice(nucleotides) for _ in range(length))
        fragments.append(fragment)
    
    return fragments


def generate_single_sample(
    sequence_length: int = 2000,
    coverage: int = 30,
    add_repeat_ratio: float = 0.3,
    repeat_length_min: int = 10,
    repeat_length_max: int = 30,
    noise_modes: List[str] = None,
    modified_noise_ratio: float = 0.2,
    random_noise_ratio: float = 0.2,
    modification_ratio: float = 0.2,
    drop_original: bool = False,
    random_noise_length: int = 100,
    read_length_min: int = 50,
    read_length_max: int = 150,
    tail_error_ratio: float = 0.3,                       # tail erosion ratio
    tail_error_length_range: Tuple[int, int] = (3, 10),  # tail erosion length range
    seed: int = None,
    noise_mode: str = None  # backward compatibility
) -> Dict:
    """
    Generate a single training sample (supports combined noise).

    Args:
        tail_error_ratio: fraction of fragments to have tail erosion (default 30%)
        tail_error_length_range: length range of tail erosion per fragment (min, max)

    Noise execution order:
        tail_error -> modified -> random
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ===== backward compatibility =====
    if noise_mode is not None:
        import warnings
        warnings.warn("Parameter 'noise_mode' is deprecated; use 'noise_modes' (list).", DeprecationWarning)
        if noise_modes is not None:
            raise ValueError("Cannot pass both 'noise_mode' and 'noise_modes'")
        noise_modes = [noise_mode]
    if noise_modes is None:
        noise_modes = ["none"]

    # ===== normalize noise modes =====
    noise_modes = [m.lower() for m in noise_modes]
    if "none" in noise_modes and len(noise_modes) > 1:
        noise_modes = [m for m in noise_modes if m != "none"]
    noise_modes = list(dict.fromkeys(noise_modes))

    valid_modes = {"tail_error", "modified", "random", "none"}
    execution_order = ["tail_error", "modified", "random"]
    for mode in noise_modes:
        if mode not in valid_modes:
            raise ValueError(f"Invalid noise mode: {mode}, valid: {valid_modes}")
    ordered_modes = [m for m in execution_order if m in noise_modes]

    # ===== 1. Generate original sequence =====
    original_sequence = generate_complex_sequence(sequence_length, seed)

    original_sequence, copypaste_chunks, copypaste_ops = shuffle_copy_paste(
        original_sequence,
        min_len=repeat_length_min,
        max_len=repeat_length_max,
        p_geometric=0.75,  # near 1 -> very few; 0.5 -> possibly ~10+; 0.75 -> 1-2, rarely 4
        p_do_copypaste=add_repeat_ratio,
        seed=seed
    )

    # ===== 2. Split sequence =====
    fragments = split_sequence(original_sequence, coverage, read_length_min, read_length_max)
    original_fragment_count = len(fragments)
    current_fragments = fragments.copy()
    noise_fragments_added = 0

    tail_error_applied = False
    modified_applied = False
    random_applied = False

    # ===== 3. Add noise in order =====
    for mode in ordered_modes:
        if mode == "tail_error":
            num_to_modify = max(1, int(len(current_fragments) * tail_error_ratio))
            indices_to_modify = random.sample(range(len(current_fragments)), num_to_modify)
            for i in indices_to_modify:
                k = random.randint(*tail_error_length_range)
                current_fragments[i] = modify_fragment_tail_all(
                    current_fragments[i], k
                )
            tail_error_applied = True

        elif mode == "modified":
            before = len(current_fragments)
            current_fragments = add_noise_to_fragments(
                current_fragments, modified_noise_ratio, modification_ratio, drop_original=drop_original,
            )
            added = len(current_fragments) - before
            noise_fragments_added += added
            modified_applied = True

        elif mode == "random":
            base_count = len(current_fragments) - noise_fragments_added
            noise_count = (
                int(base_count * random_noise_ratio / (1 - random_noise_ratio))
                if random_noise_ratio < 1.0 else base_count
            )
            random_noise = generate_random_fragments(noise_count, random_noise_length)
            current_fragments.extend(random_noise)
            random.shuffle(current_fragments)
            noise_fragments_added += noise_count
            random_applied = True

    # Ensure final order is randomized regardless of noise
    random.shuffle(current_fragments)

    # ===== Statistics =====
    total_fragments = len(current_fragments)
    actual_noise_ratio = noise_fragments_added / total_fragments if total_fragments > 0 else 0.0
    avg_occurrence = (
        sum(len(f) for f in current_fragments) / len(original_sequence)
        if len(original_sequence) > 0 else 0.0
    )

    # ===== 4. Build text =====
    gene_id = ""
    task_description = (
        "The above are overlapping fragments of a DNA sequence that have been split and shuffled. "
        "The fragments are separated by line breaks. Please assemble all the fragments into a complete sequence "
        "to restore the original DNA sequence. Note: There may be overlapping parts at the beginning and end of the fragments, "
        "and the overlapping content should be merged during assembly to ensure the restoration of the original complete sequence."
    )
    sequence_start = original_sequence[:75] if len(original_sequence) >= 75 else original_sequence
    start_info = f"The beginning of the complete sequence is: {sequence_start}"

    stats_info = []
    stats_info.append(f"Original sequence length: {len(original_sequence)}")
    stats_info.append(f"Total fragments: {total_fragments}")
    stats_info.append(f"Real fragments: {original_fragment_count}")
    if noise_fragments_added > 0:
        stats_info.append(f"Noise fragments added: {noise_fragments_added}")
        stats_info.append(f"Noise ratio: {actual_noise_ratio:.1%}")

    if tail_error_applied:
        stats_info.append(
            f"Terminal-error modification: {tail_error_ratio:.0%} of fragments had their last "
            f"{tail_error_length_range[0]}-{tail_error_length_range[1]} bp randomly substituted"
        )
    if modified_applied:
        stats_info.append(f"Modified noise fragments: ~{modification_ratio:.0%} base changes")
    if random_applied:
        stats_info.append(f"Random noise fragments: {noise_count if random_applied else 0} completely random fragments")

    stats_info.append(f"Average occurrence per character: {avg_occurrence:.2f}")

    # ===== Assemble question =====
    question_parts = [gene_id, "", task_description, ""]
    if tail_error_applied or modified_applied or random_applied:
        warnings_list = []
        if tail_error_applied:
            warnings_list.append(
                f"{tail_error_ratio:.0%} of fragments have their terminal "
                f"{tail_error_length_range[0]}-{tail_error_length_range[1]} bp randomly substituted"
            )
        if modified_applied:
            warnings_list.append(
                f"some fragments contain ~{modification_ratio:.0%} modified bases"
            )
        if random_applied:
            warnings_list.append(
                f"{noise_fragments_added if random_applied else 0} fragments are completely random and unrelated"
            )
        noise_warning = (
            f"Note: The fragment set contains noise. Specifically: "
            f"{'; '.join(warnings_list)}. "
            f"These noisy fragments should be identified and excluded during assembly."
        )
        question_parts.append(noise_warning)
        question_parts.append("")

    question_parts.append(start_info)
    question_parts.extend(stats_info)
    question = "\n".join(question_parts)

    seq_id = hashlib.md5(original_sequence.encode('utf-8')).hexdigest()   # 32-char lowercase

    return {
        "id": seq_id,
        "sequence_length": sequence_length,
        "copypaste_chunks": copypaste_chunks, 
        "copypaste_ops": copypaste_ops,
        "noise_modes": noise_modes,
        "original_fragment_count": original_fragment_count,
        "noise_fragments_added": noise_fragments_added,
        "total_fragments": total_fragments,
        "avg_occurrence":avg_occurrence,
        "tail_error_ratio": tail_error_ratio,
        "tail_error_length_range": tail_error_length_range,
        "modified_noise_ratio": modified_noise_ratio, 
        "modification_ratio": modification_ratio,
        "drop_original": drop_original,
        "question": question,
        "answer": original_sequence,
        "reasoning": "",
        "dna_sequences": current_fragments
    }


def generate_dataset(
    num_samples: int = 100,
    sequence_length_range: Tuple[int, int] = (2000, 2000),
    coverage_range: Tuple[int, int] = (30, 30),
    add_repeat_ratio: float = 0.3,
    repeat_length_min: int = 10,
    repeat_length_max: int = 30,
    noise_modes: List[str] = ["none"],
    drop_original: bool = False,
    modified_noise_ratio_range: Tuple[float, float] = (0.1, 0.3),
    random_noise_ratio_range: Tuple[float, float] = (0.1, 0.3),
    modification_ratio_range: Tuple[float, float] = (0.1, 0.3),
    tail_error_ratio: float = 0.3,
    tail_error_length_range: Tuple[int, int] = (3, 10),
    random_noise_length_range: Tuple[int, int] = (80, 120),
    read_length_range: Tuple[int, int] = (50, 150),
    output_file: str = "dataset.json",
    seed: int = None
) -> None:
    """
    Generate the full dataset.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    dataset = []
    
    print(f"Starting generation of {num_samples} samples...")
    
    for i in range(num_samples):
        # Randomly choose parameters
        seq_length = random.randint(*sequence_length_range)
        coverage = random.randint(*coverage_range)
        modified_noise_ratio = random.uniform(*modified_noise_ratio_range)
        random_noise_ratio = random.uniform(*random_noise_ratio_range)
        modification_ratio = random.uniform(*modification_ratio_range)
        random_noise_length = random.randint(*random_noise_length_range)
        read_length_min = read_length_range[0]
        read_length_max = read_length_range[1]
        
        # Use a different seed for each sample
        sample_seed = (seed + i) if seed is not None else None
        
        try:
            sample = generate_single_sample(
                sequence_length=seq_length,
                coverage=coverage,
                add_repeat_ratio=add_repeat_ratio,
                repeat_length_min=repeat_length_min,
                repeat_length_max=repeat_length_max,
                noise_modes=noise_modes,
                modified_noise_ratio=modified_noise_ratio,
                random_noise_ratio=random_noise_ratio,
                modification_ratio=modification_ratio,
                drop_original=drop_original,
                random_noise_length=random_noise_length,
                tail_error_ratio=tail_error_ratio,
                tail_error_length_range=tail_error_length_range,
                read_length_min=read_length_min,
                read_length_max=read_length_max,
                seed=sample_seed
            )
            
            dataset.append(sample)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
                
        except Exception as e:
            print(f"Error generating sample {i + 1}: {e}")
            continue
    
    # Save dataset
    # print(f"Saving dataset to {output_file}...")
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Saving dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"Dataset generation complete! Total samples: {len(dataset)}")
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"  Sequence length range: {sequence_length_range}")
    print(f"  Coverage range: {coverage_range}")
    print(f"  Noise modes: {noise_modes}")
    print(f"  Modified noise ratio range: {modified_noise_ratio_range}")
    print(f"  Random noise ratio range: {random_noise_ratio_range}")
    
    # Count fragments and sequence lengths
    noise_counts = {}
    for sample in dataset:
        fragments = sample['dna_sequences']
        answer_len = len(sample['answer'])
        noise_counts.setdefault('total_fragments', []).append(len(fragments))
        noise_counts.setdefault('sequence_lengths', []).append(answer_len)
    
    avg_fragments = sum(noise_counts['total_fragments']) / len(noise_counts['total_fragments'])
    avg_seq_len = sum(noise_counts['sequence_lengths']) / len(noise_counts['sequence_lengths'])
    
    print(f"  Average fragments: {avg_fragments:.1f}")
    print(f"  Average sequence length: {avg_seq_len:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Generate DNA sequence assembly dataset")
    
    parser.add_argument('--num-samples', '-n', type=int, default=100,
                       help='Number of samples to generate (default: 100)')
    parser.add_argument('--output', '-o', default='dataset.json',
                       help='Output filename (default: dataset.json)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Sequence params
    parser.add_argument('--seq-len-min', type=int, default=2000,
                       help='Minimum sequence length (default: 2000)')
    parser.add_argument('--seq-len-max', type=int, default=2000,
                       help='Maximum sequence length (default: 2000)')
    
    # Coverage params
    parser.add_argument('--coverage-min', type=int, default=30,
                       help='Minimum coverage (default: 30)')
    parser.add_argument('--coverage-max', type=int, default=30,
                       help='Maximum coverage (default: 30)')

    # Repeat insertion params
    parser.add_argument('--add-repeat-ratio', type=float, default=0.3,
                        help='Probability of inserting repeat sequences (default 0.3)')
    parser.add_argument('--repeat-length-min', type=int, default=10,
                        help='Minimum repeat length (default: 10)')
    parser.add_argument('--repeat-length-max', type=int, default=30,
                        help='Maximum repeat length (default: 30)')
    
    # Read length params
    parser.add_argument('--read-len-min', type=int, default=150,
                       help='Minimum read length (default: 150)')
    parser.add_argument('--read-len-max', type=int, default=150,
                       help='Maximum read length (default: 150)')
    
    # Noise mode params
    parser.add_argument('--noise-modes', nargs='+', 
                       choices=['tail_error','modified', 'random', 'none'],
                       default=['none'],
                       help='Noise mode (default: none)')

    parser.add_argument('--drop-original', action='store_true',
                        default=False,
                        help='Drop fragments before modification (default: keep)')

    # tail_error tail erosion ratio
    parser.add_argument('--tail-error-ratio', type=float, default=0.3,
                        help='Fraction of fragments eroded in tail_error mode (default 0.3)')
    parser.add_argument('--tail-error-len-min', type=int, default=3,
                        help='Minimum tail erosion length in tail_error (default 3)')
    parser.add_argument('--tail-error-len-max', type=int, default=10,
                        help='Maximum tail erosion length in tail_error (default 10)')

    # modified base change ratio
    parser.add_argument('--modified-noise-ratio-min', type=float, default=0.1,
                        help='Minimum noise ratio (default: 0.1)')
    parser.add_argument('--modified-noise-ratio-max', type=float, default=0.1,
                        help='Maximum noise ratio (default: 0.1)')
    parser.add_argument('--modification-ratio-min', type=float, default=0.1,
                       help='Minimum base modification ratio (default: 0.1)')
    parser.add_argument('--modification-ratio-max', type=float, default=0.3,
                       help='Maximum base modification ratio (default: 0.3)')

    # random noise: total length of added noisy fragments
    parser.add_argument('--random-noise-ratio-min', type=float, default=0.1,
                        help='Minimum noise ratio (default: 0.1)')
    parser.add_argument('--random-noise-ratio-max', type=float, default=0.1,
                        help='Maximum noise ratio (default: 0.1)')
    parser.add_argument('--random-noise-length', type=int, default=150,
                        help='Length of random fragments (default 150)')


    
    args = parser.parse_args()
    
    # Generate dataset
    generate_dataset(
        num_samples=args.num_samples,
        sequence_length_range=(args.seq_len_min, args.seq_len_max),
        coverage_range=(args.coverage_min, args.coverage_max),
        add_repeat_ratio=args.add_repeat_ratio,
        repeat_length_min=args.repeat_length_min,
        repeat_length_max=args.repeat_length_max,
        noise_modes=args.noise_modes,
        drop_original=args.drop_original,
        modified_noise_ratio_range=(args.modified_noise_ratio_min, args.modified_noise_ratio_max),
        random_noise_ratio_range=(args.random_noise_ratio_min, args.random_noise_ratio_max),
        modification_ratio_range=(args.modification_ratio_min, args.modification_ratio_max),
        tail_error_ratio=args.tail_error_ratio,
        tail_error_length_range=(args.tail_error_len_min, args.tail_error_len_max),
        random_noise_length_range=(args.random_noise_length, args.random_noise_length),
        read_length_range=(args.read_len_min, args.read_len_max),
        output_file=args.output,
        seed=args.seed
    )

if __name__ == "__main__":
    main()

    # num_samples = 1
    # seq_len_min = 1105
    # seq_len_max = 1105
    # coverage_min = 10
    # coverage_max = 10
    # add_repeat_ratio = 0.3
    # repeat_length_min = 10,
    # repeat_length_max = 30,
    # noise_modes = ['tail_error', 'modified', 'random']
    # drop_original = True
    # random_noise_ratio_min = 0.6
    # random_noise_ratio_max = 0.6
    # modified_noise_ratio_min = 0.6
    # modified_noise_ratio_max = 0.6
    # modification_ratio_min = 0.02
    # modification_ratio_max = 0.02
    # tail_error_ratio = 0.7
    # tail_error_len_min = 10
    # tail_error_len_max = 15
    # random_noise_length = 151
    # read_len_min = 151
    # read_len_max = 151
    # output = 'dataset.json'
    # seed = 91
    # generate_dataset(
    #     num_samples=num_samples,
    #     sequence_length_range=(seq_len_min, seq_len_max),
    #     coverage_range=(coverage_min, coverage_max),
    #     noise_modes=noise_modes,
    #     drop_original=drop_original,
    #     random_noise_ratio_range=(random_noise_ratio_min, random_noise_ratio_max),
    #     modified_noise_ratio_range=(modified_noise_ratio_min, modified_noise_ratio_max),
    #     modification_ratio_range=(modification_ratio_min, modification_ratio_max),
    #     tail_error_ratio=tail_error_ratio,
    #     tail_error_length_range=(tail_error_len_min, tail_error_len_max),
    #     random_noise_length_range=(random_noise_length, random_noise_length),
    #     read_length_range=(read_len_min, read_len_max),
    #     output_file=output,
    #     seed=seed
    # )
