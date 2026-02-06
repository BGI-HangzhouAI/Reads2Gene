import json
import os
import glob
import tempfile
import shutil

# ================= Configuration =================
# Input folder path (update to your actual path)
folder_path = './score_result' 
# ===========================================

def process_and_overwrite():
    # Get all .jsonl files
    files = glob.glob(os.path.join(folder_path, '*.jsonl'))
    
    if not files:
        print(f"No .jsonl files found in {folder_path}.")
        return

    print(f"Found {len(files)} files; starting processing with overwrite...")

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename} ...", end='', flush=True)
        
        # Create a temporary file
        # delete=False keeps the file after close so we can move it
        temp_fd, temp_path = tempfile.mkstemp(dir=folder_path, text=True)
        
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as outfile, \
                 open(file_path, 'r', encoding='utf-8') as infile:
                
                for line in infile:
                    line = line.strip()
                    if not line: continue
                    
                    try:
                        data = json.loads(line)
                        
                        # === Remove field ===
                        data.pop('dna_sequences', None)
                        
                        # Write to temporary file
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                        
                    except json.JSONDecodeError:
                        # If data is bad, choose to skip; here we warn
                        print(f"\n[Warning] Parse failed, skipping line: {line[:30]}...")

            # Replace source with temp file (atomic in same directory)
            # shutil.move may not be atomic across filesystems; here it usually uses os.rename
            shutil.move(temp_path, file_path)
            print(" [Done]")

        except Exception as e:
            print(f"\n[Error] Failed to process {filename}: {e}")
            print("Keeping original file and cleaning up temp file.")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    print("All files processed.")

if __name__ == '__main__':
    process_and_overwrite()
