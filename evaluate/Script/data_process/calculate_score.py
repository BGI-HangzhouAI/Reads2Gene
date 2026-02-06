# # import json
# # import dataclasses
# # import os
# # import dna_score

# # def process_dna_data(input_file, output_file, test_mode=False):
# #     """
# #     Read input_file (jsonl), compute scores, write to output_file (jsonl).
    
# #     Args:
# #         input_file: input file path
# #         output_file: output file path
# #         test_mode: if True, only process first 5 lines and print detailed timing info.
# #     """
    
# #     # 1. Initialize scorer
    
# #     print(f"Start processing: {input_file}")
# #     if test_mode:
# #         print("[Hint] Test mode: only process first 5 lines and print timing details.\n")
    
# #     success_count = 0
# #     error_count = 0
    
# #     # Check output directory and create if missing
# #     output_dir = os.path.dirname(output_file)
# #     if output_dir and not os.path.exists(output_dir):
# #         os.makedirs(output_dir)
    
# #     with open(input_file, 'r', encoding='utf-8') as fin, \
# #          open(output_file, 'w', encoding='utf-8') as fout:
        
# #         for line_num, line in enumerate(fin):
# #             # --- Test mode: stop after 5 lines ---
# #             if test_mode and line_num >= 5:
# #                 print("\n>>> Test mode: reached 5-line limit, stopping.")
# #                 break

# #             line = line.strip()
# #             if not line:
# #                 continue
            
# #             try:
# #                 # 2. Parse JSON data
# #                 data = json.loads(line)
                
# #                 pred_seq = data.get('extracted_answer', "")
# #                 gt_seq = data.get('answer', "")
# #                 reads = data.get('dna_sequences', [])
                
# #                 # Data cleaning
# #                 if pred_seq is None: pred_seq = ""
# #                 if gt_seq is None: gt_seq = ""
# #                 if reads is None: reads = []

# #                 # In test mode, print current index
# #                 if test_mode:
# #                     print(f"\n=== Processing record {line_num + 1} ===")

# #                 # 4. Score
# #                 if not gt_seq:
# #                     # If no ground truth, cannot score
# #                     metrics_dict = {
# #                         "error": "Ground truth sequence is empty",
# #                         "total_score": 0.0
# #                     }
# #                     if test_mode:
# #                         print("Skipping: Ground truth is empty.")
# #                 else:
# #                     # --- Core change: in test mode, set verbose=True ---
# #                     # This calls RewardMetrics.__str__ to print timing and scores
# #                     metrics: RewardMetrics = dna_score.reward_function_equal(
# #                         pred_seq_str=pred_seq,
# #                         gt_seq_str=gt_seq,
# #                         reads_list=reads
# #                     )
                    
# #                     # Convert to dict
# #                     metrics_dict = dataclasses.asdict(metrics)

# #                 # 5. Write scores back to data object
# #                 data['scoring_metrics'] = metrics_dict
# #                 data['final_score'] = metrics_dict.get('total_score', 0.0)

# #                 # 6. Write output file
# #                 fout.write(json.dumps(data, ensure_ascii=False) + "\n")
# #                 success_count += 1

# #                 # Progress logging in normal mode
# #                 if not test_mode and line_num % 10 == 0:
# #                     print(f"Processed {line_num + 1} records...")

# #             except Exception as e:
# #                 error_count += 1
# #                 print(f"Error on line {line_num + 1}: {e}")
# #                 continue

# #     print("-" * 30)
# #     print("Processing complete.")
# #     print(f"Success: {success_count}")
# #     print(f"Failed: {error_count}")
# #     print(f"Results saved to: {output_file}")

# # if __name__ == "__main__":
# #     # Configure input and output file paths
# #     INPUT_PATH = "/path/demo/test_500_2k_16k_5_11_wout_noise_wout_echo_ge3-pro-preview.jsonl"    
    
# #     # Use a different name for test output to avoid overwriting full run results
# #     OUTPUT_PATH = "/path/demo/test_500_2k_16k_5_11_wout_noise_wout_echo_ge3-pro-preview_score.jsonl"
    
# #     # --- Set to True to enable test mode ---
# #     process_dna_data(INPUT_PATH, OUTPUT_PATH, test_mode=False)

# import json
# import dataclasses
# import os
# import time  # New: import time module
# import dna_score

# def process_dna_data(input_file, output_file, test_mode=False):
#     """
#     Read input_file (jsonl), compute scores, write to output_file (jsonl).
#     """
    
#     print(f"Start processing: {input_file}")
#     if test_mode:
#         print("[Hint] Test mode: only process first 5 lines and print timing details.\n")
    
#     success_count = 0
#     error_count = 0
    
#     # Check output directory and create if missing
#     output_dir = os.path.dirname(output_file)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     with open(input_file, 'r', encoding='utf-8') as fin, \
#          open(output_file, 'w', encoding='utf-8') as fout:
        
#         for line_num, line in enumerate(fin):
#             # --- Test mode: stop after 5 lines ---
#             if test_mode and line_num >= 5:
#                 print("\n>>> Test mode: reached 5-line limit, stopping.")
#                 break

#             line = line.strip()
#             if not line:
#                 continue
            
#             try:
#                 # Parse JSON data
#                 data = json.loads(line)
                
#                 pred_seq = data.get('extracted_answer', "")
#                 gt_seq = data.get('answer', "")
#                 reads = data.get('dna_sequences', [])
                
#                 # Data cleaning
#                 if pred_seq is None: pred_seq = ""
#                 if gt_seq is None: gt_seq = ""
#                 if reads is None: reads = []

#                 if test_mode:
#                     print(f"\n=== Processing record {line_num + 1} ===")

#                 # Score
#                 if not gt_seq:
#                     metrics_dict = {
#                         "error": "Ground truth sequence is empty",
#                         "total_score": 0.0
#                     }
#                     if test_mode:
#                         print("Skipping: Ground truth is empty.")
#                 else:
#                     metrics = dna_score.reward_function_equal(
#                         pred_seq_str=pred_seq,
#                         gt_seq_str=gt_seq,
#                         reads_list=reads
#                     )
#                     metrics_dict = dataclasses.asdict(metrics)

#                 # Write scores back to data object
#                 data['scoring_metrics'] = metrics_dict
#                 data['final_score'] = metrics_dict.get('total_score', "null")

#                 # Write output file
#                 fout.write(json.dumps(data, ensure_ascii=False) + "\n")
#                 success_count += 1

#                 # Progress logging in normal mode
#                 if not test_mode and line_num % 10 == 0:
#                     print(f"Processed {line_num + 1} records...", flush=True) # flush to keep nohup logs fresh

#             except Exception as e:
#                 error_count += 1
#                 print(f"Error on line {line_num + 1}: {e}")
#                 continue

#     print("-" * 30)
#     print("Processing complete.")
#     print(f"Success: {success_count}")
#     print(f"Failed: {error_count}")
#     print(f"Results saved to: {output_file}")

# if __name__ == "__main__":
#     # Configure input and output file paths
#     INPUT_PATH = "/path/demo/test_500_2k_16k_5_11_with_noise_with_echo_ge3-pro-preview_extra_ans.jsonl"    
#     OUTPUT_PATH = "/path/demo/test_500_2k_16k_5_11_with_noise_with_echo_ge3-pro-preview_score.jsonl"
    
#     # --- Record start time ---
#     start_time = time.time()
    
#     # Run main logic (test_mode=False for full run)
#     process_dna_data(INPUT_PATH, OUTPUT_PATH, test_mode=False)
    
#     # --- Compute and print total runtime ---
#     end_time = time.time()
#     total_seconds = end_time - start_time
    
#     # Convert to h:m:s format
#     m, s = divmod(total_seconds, 60)
#     h, m = divmod(m, 60)
    
#     print("="*30)
#     print(f"Total runtime: {int(h)}h {int(m)}m {s:.2f}s")
#     print("="*30)

# import json
# import dataclasses
# import os
# import time
# import dna_score
# import sys 

# def process_dna_data(input_file, output_file, test_mode=False):
#     """
#     Read input_file (jsonl), compute scores, write to output_file (jsonl).
#     Supports resume: if output file exists, already processed data is skipped.
#     """
    
#     print(f"Start processing: {input_file}")
#     if test_mode:
#         print("[Hint] Test mode: only process first 5 lines and print timing details.\n")
    
#     success_count = 0
#     error_count = 0
#     skipped_count = 0 
    
#     # Check output directory
#     output_dir = os.path.dirname(output_file)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # --- 1. Scan processed data IDs (resume logic) ---
#     processed_ids = set()
#     write_mode = 'w'  # default overwrite mode

#     if not test_mode and os.path.exists(output_file):
#         print(f"Output file already exists: {output_file}")
#         print("Scanning processed data to resume...")
#         try:
#             with open(output_file, 'r', encoding='utf-8') as f_exist:
#                 for line in f_exist:
#                     line = line.strip()
#                     if not line: continue
#                     try:
#                         exist_data = json.loads(line)
#                         # Prefer id; fall back to answer
#                         uid = exist_data.get('id')
#                         if uid is None:
#                             uid = exist_data.get('answer')
                        
#                         if uid:
#                             processed_ids.add(str(uid))
#                     except:
#                         pass
#             print(f"Loaded {len(processed_ids)} historical records; these will be skipped.")
#             write_mode = 'a'  # switch to append mode
#         except Exception as e:
#             print(f"Failed to read old file; using overwrite mode: {e}")
#             write_mode = 'w'
    
#     # --- 2. Main processing logic ---
#     with open(input_file, 'r', encoding='utf-8') as fin, \
#          open(output_file, mode=write_mode, encoding='utf-8') as fout:
        
#         for line_num, line in enumerate(fin):
#             # Test mode limit
#             if test_mode and line_num >= 5:
#                 print("\n>>> Test mode: reached 5-line limit, stopping.")
#                 break

#             line = line.strip()
#             if not line:
#                 continue
            
#             try:
#                 # Parse JSON
#                 data = json.loads(line)
                
#                 # --- Check if already processed ---
#                 current_uid = data.get('id')
#                 if current_uid is None:
#                     current_uid = data.get('answer')
                
#                 # If exists, skip
#                 if str(current_uid) in processed_ids:
#                     skipped_count += 1
#                     # Skip logs can be throttled (every 1000 lines here)
#                     if skipped_count % 1000 == 0:
#                         print(f"Skipped {skipped_count} duplicate records...", flush=True)
#                     continue

#                 # === Scoring logic ===
#                 pred_seq = data.get('extracted_answer', "")
#                 gt_seq = data.get('answer', "")
#                 reads = data.get('dna_sequences', [])
                
#                 if pred_seq is None: pred_seq = ""
#                 if gt_seq is None: gt_seq = ""
#                 if reads is None: reads = []

#                 if test_mode:
#                     print(f"\n=== Processing record {line_num + 1} ===")

#                 if not gt_seq:
#                     metrics_dict = {
#                         "error": "Ground truth sequence is empty",
#                         "total_score": "null"
#                     }
#                     if test_mode:
#                         print("Skipping: Ground truth is empty.")
#                 else:
#                     metrics = dna_score.reward_function_equal(
#                         pred_seq_str=pred_seq,
#                         gt_seq_str=gt_seq,
#                         reads_list=reads
#                     )
#                     metrics_dict = dataclasses.asdict(metrics)

#                 data['scoring_metrics'] = metrics_dict
#                 data['final_score'] = metrics_dict.get('total_score', "null")

#                 # Write file
#                 fout.write(json.dumps(data, ensure_ascii=False) + "\n")
#                 success_count += 1

#                 # --- Change: log after each record ---
#                 # Only in non-test mode (test mode already logs)
#                 if not test_mode:
#                     print(f"Progress: line {line_num + 1} | Success: {success_count} | Skipped: {skipped_count}", flush=True)

#             except Exception as e:
#                 error_count += 1
#                 print(f"Error on line {line_num + 1}: {e}")
#                 continue

#     print("-" * 30)
#     print("Processing complete.")
#     print(f"Newly processed success: {success_count}")
#     print(f"Skipped existing: {skipped_count}")
#     print(f"Failed: {error_count}")
#     print(f"Results saved to: {output_file}")


# if __name__ == "__main__":
#     # Use command-line parameters
#     if len(sys.argv) < 3:
#         print("Usage: python calculate_score.py <input_file> <output_file>")
#         sys.exit(1)

#     INPUT_PATH = sys.argv[1]
#     OUTPUT_PATH = sys.argv[2]
    
#     start_time = time.time()
    
#     # Run main logic
#     process_dna_data(INPUT_PATH, OUTPUT_PATH, test_mode=False)
    
#     end_time = time.time()
#     total_seconds = end_time - start_time
#     m, s = divmod(total_seconds, 60)
#     h, m = divmod(m, 60)
    
#     print("="*30)
#     print(f"Total runtime: {int(h)}h {int(m)}m {s:.2f}s")
#     print("="*30)


import json
import dataclasses
import os
import time
import sys
import dna_score
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# Core compute function (must be at top-level for subprocesses)
# ==========================================
def worker_task(line_content):
    """
    Receive one JSON line, compute, and return the processed dict.
    If an error occurs, return a dict with error info.
    """
    try:
        data = json.loads(line_content)
        
        pred_seq = data.get('extracted_answer', "")
        gt_seq = data.get('answer', "")
        reads = data.get('dna_sequences', [])
        
        if pred_seq is None: pred_seq = ""
        if gt_seq is None: gt_seq = ""
        if reads is None: reads = []

        # Scoring logic
        if not gt_seq:
            metrics_dict = {
                "error": "Ground truth sequence is empty",
                "total_score": "null"
            }
        else:
            # Call dna_score (assumed CPU-bound)
            metrics = dna_score.reward_function_equal(
                pred_seq_str=pred_seq,
                gt_seq_str=gt_seq,
                reads_list=reads
            )
            metrics_dict = dataclasses.asdict(metrics)

        data['scoring_metrics'] = metrics_dict
        data['final_score'] = metrics_dict.get('total_score', "null")
        
        return {"status": "success", "data": data}

    except Exception as e:
        return {"status": "error", "msg": str(e), "line_content": line_content}

# ==========================================
# Main processing flow
# ==========================================
def process_dna_data_parallel(input_file, output_file, max_workers=100):
    """
    Parallel processing entry point.
    """
    print(f"Starting parallel processing: {input_file}")
    print(f"Worker processes: {max_workers}")
    
    success_count = 0
    error_count = 0
    skipped_count = 0 
    
    # Check output directory
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Scan processed data IDs (resume logic) ---
    processed_ids = set()
    write_mode = 'w'

    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        print("Scanning processed data to resume...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f_exist:
                for line in f_exist:
                    line = line.strip()
                    if not line: continue
                    try:
                        exist_data = json.loads(line)
                        uid = exist_data.get('id')
                        if uid is None:
                            uid = exist_data.get('answer')
                        if uid:
                            processed_ids.add(str(uid))
                    except:
                        pass
            print(f"Loaded {len(processed_ids)} historical records; these will be skipped.")
            write_mode = 'a'
        except Exception as e:
            print(f"Failed to read old file; using overwrite mode: {e}")
            write_mode = 'w'
    
    # --- 2. Read input and submit tasks ---
    tasks = []
    lines_to_process = []
    
    print("Reading input file...")
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            
            # Simple pre-check to skip (avoid sending skipped data to workers)
            # Parse ID here; small overhead is cheaper than cross-process transfer
            try:
                temp_data = json.loads(line)
                curr_id = temp_data.get('id')
                if curr_id is None:
                    curr_id = temp_data.get('answer')
                
                if str(curr_id) in processed_ids:
                    skipped_count += 1
                    continue
                
                # If not skipped, add to the processing list
                lines_to_process.append(line)
                
            except:
                # Send lines that fail parsing so worker can error out
                lines_to_process.append(line)

    print("Finished reading input file.")
    print(f"Skipped existing: {skipped_count}")
    print(f"Remaining to process: {len(lines_to_process)}")
    
    if not lines_to_process:
        print("All data already processed; nothing to run.")
        return

    # --- 3. Start process pool ---
    print("Starting process pool...")
    
    # Keep output file handle open
    with open(output_file, mode=write_mode, encoding='utf-8') as fout:
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            # future_to_line = {executor.submit(worker_task, line): line for line in lines_to_process}
            futures = [executor.submit(worker_task, line) for line in lines_to_process]
            
            total_tasks = len(futures)
            
            # Use as_completed to get results as they finish
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                
                if result['status'] == 'success':
                    # Write result
                    fout.write(json.dumps(result['data'], ensure_ascii=False) + "\n")
                    success_count += 1
                else:
                    error_count += 1
                    print(f"Processing error: {result['msg']}")

                # Progress logging (every 10 records or last one)
                current_idx = i + 1
                if current_idx % 10 == 0 or current_idx == total_tasks:
                    percentage = (current_idx / total_tasks) * 100
                    print(f"Progress: {current_idx}/{total_tasks} ({percentage:.1f}%) | Success: {success_count} | Failed: {error_count}", flush=True)

    print("-" * 30)
    print("Parallel processing complete.")
    print(f"Success this run: {success_count}")
    print(f"Skipped history: {skipped_count}")
    print(f"Failed: {error_count}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # Argument validation
    if len(sys.argv) < 3:
        print("Usage: python calculate_score.py <input_file> <output_file> [num_workers]")
        sys.exit(1)

    INPUT_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    
    # Default worker count, override if a third CLI arg is provided
    WORKERS = 10
    if len(sys.argv) >= 4:
        try:
            WORKERS = int(sys.argv[3])
        except ValueError:
            print("Warning: workers is not an integer; using default value")
    
    start_time = time.time()
    
    # Run parallel logic
    process_dna_data_parallel(INPUT_PATH, OUTPUT_PATH, max_workers=WORKERS)
    
    end_time = time.time()
    total_seconds = end_time - start_time
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    
    print("="*30)
    print(f"Total runtime: {int(h)}h {int(m)}m {s:.2f}s")
    print("="*30)
