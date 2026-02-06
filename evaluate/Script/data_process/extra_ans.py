import json
import re
import os
import sys 

def process_and_save_jsonl(input_path, output_path):
    # Initialize total time
    total_generation_time = 0.0
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: input file not found {input_path}")
        return

    # Regex: match all non-letter characters
    non_letter_pattern = re.compile(r'[^a-zA-Z]')
    
    # Counter for processed lines
    processed_count = 0

    print(f"Processing file: {input_path} ...")

    try:
        # Open input (read) and output (write) simultaneously
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line_number, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: line {line_number} is not valid JSON; skipped.")
                    continue

                # -----------------------------
                # 1. Accumulate total time (do not store in fields)
                # -----------------------------
                g_time = data.get('generation_time')
                if g_time is not None:
                    try:
                        total_generation_time += float(g_time)
                    except ValueError:
                        pass # ignore non-numeric time formats

                # -----------------------------
                # 2. Handle extracted_answer == null
                # -----------------------------
                if data.get('extracted_answer') is None:
                    generation_text = data.get('generation', '')
                    
                    # Find <answer> tag
                    tag = "<answer>"
                    tag_index = generation_text.find(tag)
                    
                    if tag_index != -1:
                        # Extract content after the tag
                        raw_content = generation_text[tag_index + len(tag):]
                        
                        # Clean via regex: keep letters only, drop symbols/digits/newlines
                        clean_content = non_letter_pattern.sub('', raw_content)
                        
                        # Update the field
                        data['extracted_answer'] = clean_content
                    else:
                        # If no tag is found, keep None or leave as-is
                        pass

                # -----------------------------
                # 3. Write modified data to output file
                # -----------------------------
                # ensure_ascii=False keeps non-ASCII characters unescaped
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1

        print("-" * 30)
        print("Processing complete!")
        print(f"Total lines processed: {processed_count}")
        print(f"Saved to: {output_path}")
        print("-" * 30)
        # 4. Print total time (not stored)
        print(f"Sum of generation_time across all records: {total_generation_time}")
        print("-" * 30)

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Use command-line parameters
    if len(sys.argv) < 3:
        print("Usage: python extra_ans.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_and_save_jsonl(input_file, output_file)
