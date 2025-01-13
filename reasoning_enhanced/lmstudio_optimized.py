import json
import os
import time
import requests
import re
from typing import List, Dict, Any, Optional

class LMStudioModel:
    def __init__(self, base_url: str = "http://localhost:1234/v1", max_tokens: int = None):
        self.base_url = base_url
        self.max_tokens = max_tokens
        self._verify_connection()

    def _verify_connection(self):
        """Verify LM Studio server is running and responsive"""
        try:
            response = requests.get(f"{self.base_url}/models")
            response.raise_for_status()
            print("Successfully connected to LM Studio server")
        except Exception as e:
            print(f"Warning: Could not connect to LM Studio server: {str(e)}")
            print("Please ensure LM Studio is running and the server is started")

    def generate(self, prompt: str, retries: int = 3, temperature: float = 0.1) -> Optional[str]:
        """Generate completion with improved error handling and retries"""
        messages = [
            {"role": "system", "content": (
                "You are a logical reasoning expert specialized in solving problems step by step. "
                "Your responses should be clear, systematic, and conclude with an explicit answer choice (A-E). "
                "Always verify your logic before stating the final answer."
            )},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "messages": messages,
                        "model": "nemomix-unleashed-12b",
                        "temperature": temperature,
                        "max_tokens": self.max_tokens,
                        "stream": False,
                        "top_p": 0.1,  # Added for more focused responses
                        "frequency_penalty": 0.3,  # Reduce repetition
                        "presence_penalty": 0.3  # Encourage focused completion
                    },
                    timeout=30  # Reduced timeout to fail faster
                )
                
                response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"]
                
                # Verify we have a complete response with an answer
                if not self._is_complete_response(result):
                    print(f"Incomplete response detected (attempt {attempt + 1})")
                    continue
                    
                return result
                
            except requests.exceptions.Timeout:
                print(f"Request timed out (attempt {attempt + 1})")
                time.sleep(2)  # Short wait between retries
            except requests.exceptions.ConnectionError:
                print(f"Connection error (attempt {attempt + 1})")
                time.sleep(5)  # Longer wait for connection issues
            except Exception as e:
                print(f"Error in generation (attempt {attempt + 1}): {str(e)}")
                time.sleep(2)
        
        return None

    def _is_complete_response(self, response: str) -> bool:
        """Check if response is complete with an answer"""
        # Check for truncation indicators
        if response.endswith('...') or response.endswith('…'):
            return False
            
        # Check for answer presence
        answer_patterns = [
            r"(?i)answer\s*(?:is|:)\s*[^A-E]*([A-E])",
            r"(?i)(?:therefore|thus|hence|so)\s*[^A-E]*([A-E])",
            r"(?i)option\s*([A-E])\s*(?:is correct|must be true)",
            r"(?i)([A-E])\s*is\s*(?:the|must be the)?\s*(?:correct|true)\s*answer"
        ]
        
        for pattern in answer_patterns:
            if re.search(pattern, response):
                return True
                
        return False

class LogicalDeductionBaseline:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.save_path = args.save_path
        
        print(f"Initialized with paths:")
        print(f"- Data path: {self.data_path}")
        print(f"- Save path: {self.save_path}")
        
        self.model = LMStudioModel(max_tokens=args.max_completion_tokens)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        context = example['context'].replace("\\n", "\n")
        options_text = "\n".join(f"{opt}" for opt in example['options'])
        
        # More concise, focused prompt
        prompt = f"""# Logic Problem

{context}

Question: {example['question']}

Options:
{options_text}

Solve this step by step:
1) List each given fact
2) Analyze relationships and draw logical connections
3) Eliminate impossible options
4) Verify the only possible answer

Your solution:"""

        return prompt

    def process_dataset(self):
        dataset_path = os.path.join(self.data_path, self.dataset_name, f'{self.split}.json')
        print(f"Loading dataset from: {dataset_path}")
        
        with open(dataset_path) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples")

        outputs = []
        for i, example in enumerate(dataset):
            print(f"\nProcessing example {i+1}/{len(dataset)} - ID: {example['id']}")
            
            try:
                start_time = time.time()
                response = self.model.generate(
                    self.format_prompt(example),
                    retries=3
                )
                
                if response:
                    predicted_answer = self._extract_answer(response)
                    elapsed = time.time() - start_time
                    
                    output = {
                        'id': example['id'],
                        'answer': example['answer'],
                        'predicted_answer': predicted_answer,
                        'predicted_reasoning': response,
                        'correct': predicted_answer == example['answer'] if predicted_answer else False,
                        'time_taken': elapsed
                    }
                    outputs.append(output)
                    
                    # Status update
                    print(f"Time: {elapsed:.2f}s | Predicted: {predicted_answer} | Actual: {example['answer']} | {'✓' if output['correct'] else '✗'}")
                    
                    # Save progress
                    self._save_outputs(outputs)
                    
                    # If taking too long, adjust temperature
                    if elapsed > 45:  # If taking more than 45 seconds
                        print("Response time high, adjusting parameters for next iteration...")
                        
            except Exception as e:
                print(f"Error processing example {example['id']}: {str(e)}")
                continue

        # Final save with statistics
        self._save_final_results(outputs)
        return outputs

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract answer with improved pattern matching"""
        # First look for clear conclusion statements
        conclusion_patterns = [
            r"(?i)(?:therefore|thus|hence|so|conclude that|answer is)[^A-E]*([A-E])",
            r"(?i)option\s*([A-E])\s*(?:is|must be)\s*(?:the|correct|true)",
            r"(?i)([A-E])\s*is\s*(?:the only possible|the correct|must be the)\s*answer"
        ]
        
        for pattern in conclusion_patterns:
            if match := re.search(pattern, response):
                return match.group(1).upper()
        
        # Fallback to simpler patterns
        simple_patterns = [
            r"(?i)answer:\s*([A-E])",
            r"(?i)([A-E])\s*is\s*correct",
            r"(?i)select\s*([A-E])"
        ]
        
        for pattern in simple_patterns:
            if match := re.search(pattern, response):
                return match.group(1).upper()
        
        return None

    def _save_outputs(self, outputs: List[Dict]):
        """Save intermediate results with basic stats"""
        save_path = os.path.join(self.save_path, f'results_intermediate.json')
        
        output_data = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_processed': len(outputs),
                'correct_count': sum(1 for o in outputs if o.get('correct', False))
            },
            'results': outputs
        }
        
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
    def _save_final_results(self, outputs: List[Dict]):
        """Save final results with detailed statistics"""
        if not outputs:
            print("No results to save")
            return
            
        total = len(outputs)
        correct = sum(1 for o in outputs if o.get('correct', False))
        accuracy = correct / total
        avg_time = sum(o.get('time_taken', 0) for o in outputs) / total
        
        stats = {
            'total_examples': total,
            'correct_answers': correct,
            'accuracy': accuracy,
            'average_time': avg_time,
            'completion_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        save_path = os.path.join(self.save_path, f'results_final.json')
        with open(save_path, 'w') as f:
            json.dump({
                'statistics': stats,
                'results': outputs
            }, f, indent=2)
            
        print(f"\nFinal Statistics:")
        print(f"Total examples: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average time per example: {avg_time:.2f}s")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="LogicalDeduction", help="Name of the dataset")
    parser.add_argument("--split", default="dev", help="Dataset split to use")
    parser.add_argument("--model_name", default="nemomix-unleashed-12b", help="Model name/identifier")
    parser.add_argument("--max_completion_tokens", type=int, default=2048, help="Maximum tokens for completion")
    parser.add_argument("--data_path", default="data", help="Path to the data directory")
    parser.add_argument("--save_path", default="results", help="Path to save results")
    
    args = parser.parse_args()
    
    baseline = LogicalDeductionBaseline(args)
    baseline.process_dataset()