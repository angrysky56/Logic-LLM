import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import OpenAIModel
import argparse
import time
import re

class GPT3_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.demonstration_path = args.demonstration_path
        self.mode = args.mode
        self.current_tokens = 0  # Track tokens used in current minute

        # Create necessary directories
        os.makedirs(self.save_path, exist_ok=True)
        print(f"Results will be saved to: {self.save_path}")

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_completion_tokens)
        self.prompt_creator = self.prompt_LSAT
        self.label_phrase = 'The correct option is:'
    
    def prompt_LSAT(self, in_context_example, test_example):
        full_prompt = in_context_example
        context = test_example['context'].strip()
        question = test_example['question'].strip()
        options = '\n'.join([opt.strip() for opt in test_example['options']])
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[OPTIONS]]', options)
        return full_prompt

    def load_in_context_examples(self):
        example_path = os.path.join(self.demonstration_path, f'{self.dataset_name}_{self.mode}.txt')
        print(f"Loading examples from: {example_path}")
        with open(example_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_raw_dataset(self, split):
        dataset_path = os.path.join(self.data_path, self.dataset_name, f'{split}.json')
        print(f"Loading dataset from: {dataset_path}")
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def handle_rate_limit(self, error_message):
        # Extract tokens used and requested
        used_match = re.search(r"Used (\d+)", str(error_message))
        requested_match = re.search(r"Requested (\d+)", str(error_message))
        
        if used_match and requested_match:
            used_tokens = int(used_match.group(1))
            requested_tokens = int(requested_match.group(1))
            total_tokens = used_tokens + requested_tokens
            
            if total_tokens >= 25000:  # If approaching the 30k limit
                wait_time = 60  # Full minute wait
                print(f"Near token limit ({total_tokens}/30000). Waiting {wait_time} seconds for full token reset...")
            else:
                # Extract suggested wait time
                wait_match = re.search(r"Please try again in ([\d.]+)s", str(error_message))
                if wait_match:
                    wait_time = float(wait_match.group(1)) + 2  # Add 2s buffer
                else:
                    wait_time = 5  # Default wait
                print(f"Rate limit hit ({total_tokens} tokens). Waiting {wait_time} seconds...")
            
            time.sleep(wait_time)
            return True
        return False

    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()
        print("In-context examples loaded successfully.")

        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        print(f"Processing {len(dataset_chunks)} batches of size {batch_size}")
        
        for batch_idx, chunk in enumerate(tqdm(dataset_chunks, desc="Processing batches")):
            print(f"\nProcessing batch {batch_idx + 1}/{len(dataset_chunks)}")
            batch_start_time = time.time()
            
            # create prompt
            full_prompts = [self.prompt_creator(in_context_examples, example) for example in chunk]
            print(f"Created {len(full_prompts)} prompts for current batch")
            
            retry_count = 0
            while retry_count < 5:  # Maximum 5 retries per batch
                try:
                    print(f"Sending batch request to OpenAI API...")
                    batch_outputs = self.openai_api.batch_generate(full_prompts)
                    print(f"Received response from OpenAI API")
                    
                    # create output
                    for sample, output in zip(chunk, batch_outputs):
                        dict_output = self.update_answer(sample, output)
                        outputs.append(dict_output)
                    
                    batch_time = time.time() - batch_start_time
                    print(f"Batch {batch_idx + 1} completed in {batch_time:.2f} seconds")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f'Error in batch processing: {str(e)}')
                    if self.handle_rate_limit(str(e)):
                        retry_count += 1
                        continue
                    else:
                        print("Non-rate-limit error occurred. Waiting 60 seconds before next batch...")
                        time.sleep(60)
                        break
            
            # Save intermediate results every batch
            intermediate_path = os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_intermediate.json')
            with open(intermediate_path, 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
            print(f"Saved intermediate results to {intermediate_path}")
            
            # Add delay between batches
            time.sleep(5)  # 5-second base delay between batches

        # save final outputs        
        final_path = os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json')
        with open(final_path, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"\nSaved final results to {final_path}")
    
    def update_answer(self, sample, output):
        label_phrase = self.label_phrase
        generated_answer = output.split(label_phrase)[-1].strip()
        generated_reasoning = output.split(label_phrase)[0].strip()
        dict_output = {'id': sample['id'], 
                        'question': sample['question'], 
                        'answer': sample['answer'], 
                        'predicted_reasoning': generated_reasoning,
                        'predicted_answer': generated_answer}
        return dict_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_completion_tokens', type=int)
    parser.add_argument('--temperature', type=float, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    gpt3_problem_reduction.batch_reasoning_graph_generation(batch_size=5)  # Reduced batch size