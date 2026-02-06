import asyncio
import json
import argparse
import os
import sys
import re
from datetime import datetime
import time 
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from openai import (
    APIError, 
    APITimeoutError, 
    APIConnectionError, 
    RateLimitError, 
    BadRequestError, 
    AuthenticationError, 
    InternalServerError,
    APIStatusError
)
import traceback

# ================= Configuration =================
SYSTEM_PROMPT_DEFAULT =  """
            Respond in the following format:  
            <answer>  
            ...  
            </answer>  
            """  

# ================= Helper: extract XML Answer content =================
def extract_xml_answer(text: str):
    """
    Extract content in <answer>...</answer> from generated text.
    """
    if not text:
        return None
    
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    if not matches:
        return None
    
    final_content = matches[-1].group(1).strip()
    return final_content

class LLMProcessor:
    def __init__(self, args):
        self.args = args
        self.client = AsyncOpenAI(
            api_key=args.api_key,
            base_url=args.url
        )
        self.semaphore = asyncio.Semaphore(args.concurrency) 
        self.processed_ids = set()
        self.lock = asyncio.Lock()
        
        self.extra_body_dict = None
        if args.extra_body:
            try:
                self.extra_body_dict = json.loads(args.extra_body)
                print(f"Loaded extra_body params: {self.extra_body_dict}")
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse extra_body JSON: {args.extra_body}")

    def load_processed_ids(self):
        """Load IDs already processed in the output file."""
        if not os.path.exists(self.args.output_file):
            return
        
        print(f"Checking existing progress in {self.args.output_file}...")
        with open(self.args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        self.processed_ids.add(str(data['id']))
                except json.JSONDecodeError:
                    continue
        print(f"Found {len(self.processed_ids)} processed items.")

    async def call_llm(self, data):
            """
            Core logic to call the LLM.
            Returns tuple: (result dict/original dict, success(bool), error message(str) or None)
            """
            from transformers import AutoTokenizer

            system_content = data.get('system_prompt', SYSTEM_PROMPT_DEFAULT) 
            query = data.get('question', 'N/A')
            seq = data.get('dna_sequences','N/A')
            # Build user prompt
            user_content = f'question:{query},seq:{seq}\n /no_think'
            if not user_content:
                return data, False, "User content is empty."

            # ================= [Change 1]: build api_kwargs before try =================
            api_kwargs = {}
            try:
                # --- 1. Build base params ---
                api_kwargs = {
                    "model": self.args.model_id,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": self.args.temperature
                }
                if self.args.model_id == "ge3-pro-preview":
                    api_kwargs['reasoning_effort'] = "minimal"
                # --- 2. Handle optional params ---
                if self.args.max_tokens is not None:
                        api_kwargs["max_tokens"] = self.args.max_tokens

                if self.args.top_p is not None:
                    api_kwargs["top_p"] = self.args.top_p

                # Handle stop param (list or str)
                if self.args.stop:
                    api_kwargs["stop"] = self.args.stop

                # Handle top_k param
                if self.args.top_k is not None:
                    api_kwargs["top_k"] = self.args.top_k

                # Handle extra_body
                if self.extra_body_dict:
                    api_kwargs["extra_body"] = self.extra_body_dict
            except Exception as e:
                # Rare: if param construction fails (e.g., JSON serialization), catch it
                print(f"Error constructing args: {e}")
                return data, False, f"Args construction failed: {e}"

            # ================= Start request =================
            async with self.semaphore:
                start_time = time.time()
                try:
                    # --- 3. Send request ---
                    response = await self.client.chat.completions.create(**api_kwargs)
                    
                    end_time = time.time()
                    
                    # --- 4. Defensive checks ---
                    if not response.choices:
                        return self._attach_args(data, api_kwargs), False, "API returned empty choices list"
                    
                    choice = response.choices[0]
                    
                    if not hasattr(choice, 'message') or choice.message is None:
                        debug_info = str(choice)[:100]
                        return self._attach_args(data, api_kwargs), False, f"API returned choice with no message object. Raw: {debug_info}"
                    if self.args.model_id == "qwen_235b":
                        content = choice.message.reasoning_content
                    else:
                        content = choice.message.content
                    
                    if content is None:
                        if hasattr(choice, 'finish_reason') and choice.finish_reason == 'content_filter':
                            return self._attach_args(data, api_kwargs), False, "Response blocked by content filter (content is None)"
                        content = ""

                    generation_time = end_time - start_time
                    # --- 5. Extract XML answer ---
                    extracted_answer = extract_xml_answer(content)
                    
                    # --- 6. Build success result ---
                    result = data.copy()
                    result['generation'] = content
                    result['generation_time'] = generation_time 
                    result['model_name'] = self.args.model_name
                    result['extracted_answer'] = extracted_answer
                    result['has_answer_tag'] = extracted_answer is not None
                    
                    # Save params
                    result['api_kwargs'] = api_kwargs # or use 'other': api_kwargs
                    print(generation_time,' ',result['sequence_length'],'\n')
                    return result, True, None

                # --- 7. Exception handling [Change 2]: include api_kwargs in error returns ---
                
                except APITimeoutError:
                    msg = "Request timed out (Client side)"
                    print(f"[Timeout] ID {data.get('id')} - {msg}", file=sys.stderr)
                    return self._attach_args(data, api_kwargs), False, msg

                except RateLimitError:
                    msg = "Rate limit exceeded (429)"
                    print(f"[RateLimit] ID {data.get('id')}: {msg}", file=sys.stderr)
                    return self._attach_args(data, api_kwargs), False, msg

                except BadRequestError as e:
                    # Safely extract error details
                    error_detail = ""
                    if hasattr(e, 'body') and isinstance(e.body, dict):
                        error_detail = e.body.get('message', str(e))
                    else:
                        error_detail = str(e)
                    
                    msg = f"Bad Request (400): {error_detail}"
                    print(f"[BadRequest] ID {data.get('id')}: {error_detail}", file=sys.stderr)
                    return self._attach_args(data, api_kwargs), False, msg

                except AuthenticationError as e:
                    msg = f"Auth Error (401): {e.message}"
                    print(f"[Auth] {msg}", file=sys.stderr)
                    return self._attach_args(data, api_kwargs), False, msg

                except InternalServerError as e:
                    msg = f"Server Error (5xx): {e.message}"
                    print(f"[ServerErr] ID {data.get('id')}: {msg}", file=sys.stderr)
                    return self._attach_args(data, api_kwargs), False, msg

                except APIStatusError as e:
                    msg = f"API Status Error {e.status_code}: {e.message}"
                    print(f"[StatusErr] ID {data.get('id')}: Code={e.status_code}", file=sys.stderr)
                    return self._attach_args(data, api_kwargs), False, msg

                except APIConnectionError as e:
                    msg = f"Connection Error: {str(e)}"
                    print(f"[ConnErr] ID {data.get('id')}: {msg}", file=sys.stderr)
                    return self._attach_args(data, api_kwargs), False, msg

                except Exception as e:
                    error_msg = f"Python Runtime Error: {type(e).__name__}: {str(e)}"
                    print(f"[RuntimeErr] ID {data.get('id')}: {error_msg}", file=sys.stderr)
                    traceback.print_exc() 
                    return self._attach_args(data, api_kwargs), False, error_msg

    def _attach_args(self, data, api_kwargs):
        """Helper: attach api_kwargs to a copy of the returned data."""
        new_data = data.copy()
        new_data['api_kwargs'] = api_kwargs
        return new_data

    async def save_result(self, result):
        """Thread-safe write of results."""
        async with self.lock:
            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            if 'id' in result:
                self.processed_ids.add(str(result['id']))

    async def worker(self, data):
        """
        Worker coroutine.
        Returns (success, result/original dict, error message or None).
        """
        item_id = str(data.get('id'))
        if item_id in self.processed_ids:
            return True, data, None

        result, success, error_msg = await self.call_llm(data)
        
        if success:
            await self.save_result(result)
            has_ans = "YES" if result.get('has_answer_tag') else "NO"
            sys.stdout.write(f"\rProcessed ID: {result.get('id')} | AnsFound: {has_ans} - Success")
            sys.stdout.flush()
            return True, result, None
        else:
            sys.stdout.write(f"\rProcessed ID: {item_id} | Failed: {error_msg[:50]}...") 
            sys.stdout.flush()
            return False, result, error_msg 

    async def process_dataset(self):
        # 1. Load processed IDs
        self.load_processed_ids()

        # 2. Read input file
        all_data = []
        count = 0
        limit = self.args.test_count if self.args.test_count is not None else float('inf') 

        with open(self.args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= limit: break
                if not line.strip(): continue
                try:
                    d = json.loads(line)
                    if 'id' not in d:
                        print("Warning: Data line missing 'id', skipping.")
                        continue
                    all_data.append(d)
                    count += 1
                except json.JSONDecodeError:
                    continue
        
        if self.args.test_count is not None:
            print(f"--- Test mode: loaded first {len(all_data)} records ---")
        
        # 3. Filter pending data
        pending_data_for_first_run = [d for d in all_data if str(d['id']) not in self.processed_ids]
        print(f"\nTotal tasks loaded: {len(all_data)}, Pending for processing: {len(pending_data_for_first_run)}")

        if not pending_data_for_first_run:
            print("All data processed (or none were loaded).")
            return

        # 4. Execute all tasks, no retries
        print(f"\n--- Starting processing of {len(pending_data_for_first_run)} items (no retries) ---")
        
        tasks = [self.worker(d) for d in pending_data_for_first_run]
        results = await asyncio.gather(*tasks)
        
        # Collect failed tasks and errors
        failed_tasks_for_output = []
        for success, item_data, error_msg in results:
            if not success:
                # Record failed task id and full error info
                # Note: item_data already processed via worker -> call_llm and includes 'api_kwargs'
                failed_item = item_data.copy()
                failed_item['error'] = error_msg
                failed_tasks_for_output.append(failed_item)
        
        print("\n--- Processing finished ---")

        # 5. Final check and handle failed data
        if failed_tasks_for_output:
            print(f"Finished with {len(failed_tasks_for_output)} failed items.")
            failed_output_file = self.args.output_file + '.failed'
            with open(failed_output_file, 'w', encoding='utf-8') as f:
                for d in failed_tasks_for_output:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f"Failed tasks (IDs, errors, and api_kwargs) written to: {failed_output_file}")
        else:
            print("SUCCESS: All pending data processed successfully!")
            failed_file_path = self.args.output_file + '.failed'
            if os.path.exists(failed_file_path):
                try:
                    os.remove(failed_file_path)
                    print(f"Removed empty failed file: {failed_file_path}")
                except Exception as e:
                    print(f"Warning: Could not remove failed file {failed_file_path}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Batch Processor")
    parser.add_argument("--url", type=str, required=True, help="API Base URL")
    parser.add_argument("--api_key", type=str, required=True, help="API Key")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID")
    parser.add_argument("--model_name", type=str, required=True, help="Custom Model Name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel requests")
    parser.add_argument("--test_count", type=int, default=None, help="Process limit")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens")
    parser.add_argument("--top_p", type=float, default=None, help="Top P")
    parser.add_argument("--stop", type=str, nargs='+', default=None, help="Stop sequences")
    parser.add_argument("--top_k", type=int, default=None, help="Top K sampling")
    parser.add_argument("--extra_body", type=str, default=None, help="JSON string for extra API parameters")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    processor = LLMProcessor(args)
    asyncio.run(processor.process_dataset())
