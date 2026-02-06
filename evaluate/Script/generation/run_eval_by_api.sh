# # API_URL="*"
# # API_KEY="*"
# # MODEL_ID="ge3-pro-preview"
# # MODEL_NAME="ge3-pro-preview"
# API_URL="*"
# API_KEY="*"
# MODEL_ID="step_250_ew_ew_tc_grpo_copypaste_v7_ew"
# MODEL_NAME="step_250_ew_ew_tc_grpo_copypaste_v7_ew"
# # API_URL="*"
# # API_KEY="*"
# # MODEL_ID="deepseek-chat"
# # MODEL_NAME="deepseek-v3p2"
# # MODEL_ID="qwen3_4b"
# # MODEL_NAME="qwen3_4b"
# # MODEL_ID="step_210_ew_grpo_copypaste_v6_ew"
# # MODEL_NAME="step_210_ew_grpo_copypaste_v6_ew"
# # API_URL="*"
# # API_KEY="*"
# # MODEL_ID='qwen3-max'
# # MODEL_NAME='qwen3-max'
# # API_URL="*"
# # API_KEY="*"
# # MODEL_ID="qwen3-30b-a3b"
# # MODEL_NAME="qwen3-30b-a3b"
# # API_URL="*"
# # API_KEY="*"
# # MODEL_ID="qwen3_32b"
# # MODEL_NAME="qwen3_32b"

# INPUT_FILES=(
#     "/path/demo/test_500_2k_10k_5_11_with_noise_with_echo.jsonl"
#     "/path/demo/test_500_2k_10k_5_11_wout_noise_wout_echo.jsonl"
#     "/path/demo/test_500_2k_10k_5_11_wout_noise_with_echo.jsonl"
#     "/path/demo/test_500_2k_10k_5_11_with_noise_wout_echo.jsonl"
# )
# OUTPUT_DIR="/data"
# LOG_DIR="/path/demo/reads2gene/logs"

# CONCURRENCY=1
# MAX_TOKEN=20000
# TOP_P=0.8
# TEMPERATURE=0.01
# STOP="</answer>"
# TEST_COUNT=5


# FILENAME=$(basename "$INPUT_FILE")
# BASENAME="${FILENAME%.*}"
# OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_${MODEL_NAME}.jsonl"
# LOG_FILE="${LOG_DIR}/${BASENAME}_${MODEL_NAME}.log"

# mkdir -p "$OUTPUT_DIR"
# mkdir -p "$LOG_DIR"

# echo "Starting Processing..."

# nohup python3 -u /path/demo/reads2gene/Script/Eval_by_api.py \
#     --url "$API_URL" \
#     --api_key "$API_KEY" \
#     --model_id "$MODEL_ID" \
#     --model_name "$MODEL_NAME" \
#     --temperature "$TEMPERATURE" \
#     --input_file "$INPUT_FILE" \
#     --output_file "$OUTPUT_FILE" \
#     --concurrency "$CONCURRENCY" \
#     --max_tokens "$MAX_TOKEN" \
#     --top_p "$TOP_P" \
#     --stop "$STOP" \
#     > "$LOG_FILE" 2>&1 &

# API_URL="*"
# API_KEY="*"
# MODEL_ID="ge3-pro-preview"
# MODEL_NAME="ge3-pro-preview"
API_URL="*"
API_KEY="*"
# MODEL_ID="step_250_ew_ew_tc_grpo_copypaste_v7_ew"
# MODEL_NAME="step_250_ew_ew_tc_grpo_copypaste_v7_ew"
# API_URL="*"
# API_KEY="*"
# MODEL_ID="deepseek-chat"
# MODEL_NAME="deepseek-v3p2"
MODEL_ID="4B-both"
MODEL_NAME="4B-both"
# MODEL_ID="step_210_ew_grpo_copypaste_v6_ew"
# MODEL_NAME="step_210_ew_grpo_copypaste_v6_ew"
# API_URL="*"
# API_KEY="*"
# MODEL_ID='qwen3-max'
# MODEL_NAME='qwen3-max'
# API_URL="*"
# API_KEY="*"
# MODEL_ID="qwen3-30b-a3b"
# MODEL_NAME="qwen3-30b-a3b"
# API_URL="*"
# API_KEY="*"
# MODEL_ID="qwen3_32b"
# MODEL_NAME="qwen3_32b"
#INPUT_FILE="/path/demo/test_2k_6k_5_11_with_noise_with_echo.jsonl"
#INPUT_FILE="/path/demo/test_2k_4k_5_11_with_noise_with_echo.jsonl"
# Define input files to process
INPUT_FILES=(
    "/path/demo/test_500_2k_10k_5_11_with_noise_with_echo.jsonl"
    "/path/demo/test_500_2k_10k_5_11_wout_noise_wout_echo.jsonl"
    "/path/demo/test_500_2k_10k_5_11_wout_noise_with_echo.jsonl"
    "/path/demo/test_500_2k_10k_5_11_with_noise_wout_echo.jsonl"
)

OUTPUT_DIR="/path/demo/ssx"
LOG_DIR="/path/demo/reads2gene/logs/ssx"

CONCURRENCY=8
MAX_TOKEN=20000
TOP_P=0.8
TEMPERATURE=0.01
STOP="</answer>"
TEST_COUNT=5

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Loop over each input file
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    echo "=========================================="
    echo "Processing: $INPUT_FILE"
    echo "=========================================="
    
    FILENAME=$(basename "$INPUT_FILE")
    BASENAME="${FILENAME%.*}"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_${MODEL_NAME}.jsonl"
    LOG_FILE="${LOG_DIR}/${BASENAME}_${MODEL_NAME}.log"
    
    echo "Output file: $OUTPUT_FILE"
    echo "Log file: $LOG_FILE"
    echo "Starting Processing..."
    
    python3 -u /path/demo/reads2gene/Script/Eval_by_api.py \
        --url "$API_URL" \
        --api_key "$API_KEY" \
        --model_id "$MODEL_ID" \
        --model_name "$MODEL_NAME" \
        --temperature "$TEMPERATURE" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --concurrency "$CONCURRENCY" \
        --max_tokens "$MAX_TOKEN" \
        --top_p "$TOP_P" \
        --stop "$STOP" 
    
    echo "Job started in background. PID: $!"
    echo ""
done

echo "All jobs have been submitted!"
