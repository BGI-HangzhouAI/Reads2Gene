#!/bin/bash

TARGET_DIR="/data"

SCRIPT_EXTRA="/path/demo/reads2gene/Script/extra_ans.py"
SCRIPT_SCORE="/path/demo/reads2gene/Script/calculate_score.py"

# Iterate over all .jsonl files in target directory
for file in "$TARGET_DIR"/*.jsonl; do
    
    # Get filename without path
    filename=$(basename "$file")
    
    # --- 0. Filter logic ---
    # Skip generated result files (ending with _extra_ans.jsonl or _score.jsonl)
    if [[ "$filename" == *"_extra_ans.jsonl" ]] || [[ "$filename" == *"_score.jsonl" ]]; then
        continue
    fi

    echo "=========================================="
    echo "Processing source file: $filename"

    # Define full output paths
    base_path="${file%.jsonl}"
    extra_ans_output="${base_path}_extra_ans.jsonl"
    score_output="${base_path}_score.jsonl"

    # --- 1. Step 1: run extra_ans ---
    # Flag: whether step 1 output is ready (new or existing)
    step1_ready=false

    if [ -f "$extra_ans_output" ]; then
        echo "[Skip Step 1] File already exists: $(basename "$extra_ans_output")"
        step1_ready=true
    else
        echo "Step 1: running extra_ans.py ..."
        python "$SCRIPT_EXTRA" "$file" "$extra_ans_output"
        
        # Check Python script result
        if [ $? -eq 0 ]; then
            echo "Step 1 complete."
            step1_ready=true
        else
            echo ">>> Error: Step 1 failed; skipping remaining steps for this file."
        fi
    fi

    # --- 2. Step 2: run calculate_score ---
    # Only run step 2 if step 1 output is ready
    if [ "$step1_ready" = true ]; then
        
        if [ -f "$score_output" ]; then
            echo "[Skip Step 2] File already exists: $(basename "$score_output")"
        else
            echo "Step 2: running calculate_score.py ..."
            # Note: input file is extra_ans_output
            python "$SCRIPT_SCORE" "$extra_ans_output" "$score_output"
            
            if [ $? -eq 0 ]; then
                echo "Step 2 complete. Score file generated."
            else
                echo ">>> Error: Step 2 failed."
            fi
        fi
    else
        echo "Missing intermediate file ($(basename "$extra_ans_output")); cannot run Step 2."
    fi
    
    echo "=========================================="
    echo ""
done
