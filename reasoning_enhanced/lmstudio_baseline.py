import json
import os
import time
import requests
import re
from typing import List, Dict, Any, Optional
from lmstudio_config import MODEL_SETTINGS

class LMStudioModel:
    def __init__(self, max_tokens: int = None):
        self.base_url = "http://localhost:1234/v1"
        self.settings = MODEL_SETTINGS.copy()
        if max_tokens is not None:
            self.settings["max_tokens"] = max_tokens
        self._verify_connection()

    def _verify_connection(self):
        try:
            response = requests.get(f"{self.base_url}/models")
            print("Successfully connected to LM Studio server")
        except Exception as e:
            print(f"Warning: Could not verify LM Studio server: {e}")
            print("Please ensure LM Studio is running with the API server enabled")

    def generate(self, prompt: str) -> Optional[Dict]:
        messages = [
            {
                "role": "system", 
                "content": """You are a logical reasoning expert. Respond in this exact JSON format:
{
    "steps": [
        {"step": 1, "type": "facts", "content": "List all given facts"},
        {"step": 2, "type": "analysis", "content": "Analyze relationships"},
        {"step": 3, "type": "deduction", "content": "Apply logical deduction"},
        {"step": 4, "type": "conclusion", "content": "State which option (A-E) must be correct and why"}
    ],
    "answer": "A",
    "confidence": "high"
}"""
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]

        try:
            # Prepare the request with all available parameters
            data = {
                "messages": messages,
                **self.settings  # Include all model settings
            }
            
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print("Response was not valid JSON, attempting to parse...")
                    return self._parse_non_json_response(content)
            else:
                print(f"Error {response.status_code}: {response.text}")
                    
        except Exception as e:
            print(f"Error in generation: {str(e)}")
        
        return None

    def _parse_non_json_response(self, text: str) -> Dict:
        """Extract structured information from non-JSON response"""
        steps = []
        answer = None
        confidence = "low"

        # Try to find steps
        step_pattern = r"(?:^|\n)(\d+)[.).]\s*(.+?)(?=(?:\n\d+[.).])|$)"
        for match in re.finditer(step_pattern, text, re.DOTALL):
            step_num = int(match.group(1))
            content = match.group(2).strip()
            step_type = ["facts", "analysis", "deduction", "conclusion"][min(step_num-1, 3)]
            steps.append({
                "step": step_num,
                "type": step_type,
                "content": content
            })

        # Try to find answer
        answer_pattern = r"(?i)(?:answer|conclusion).*?([A-E])"
        if match := re.search(answer_pattern, text):
            answer = match.group(1).upper()
            confidence = "medium"  # Since we had to extract it

        return {
            "steps": steps,
            "answer": answer,
            "confidence": confidence
        }

class LogicalDeductionBaseline:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.model = LMStudioModel(max_tokens=args.max_completion_tokens)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        context = example['context'].replace("\\n", "\n")
        options_text = "\n".join(f"{opt}" for opt in example['options'])
        
        prompt = f"""Solve this logical reasoning problem. Provide your solution in JSON format.

Context:
{context}

Question:
{example['question']}

Options:
{options_text}"""

        return prompt

    def process_dataset(self):
        dataset_path = os.path.join(self.data_path, self.dataset_name, f'{self.split}.json')
        print(f"Loading dataset from: {dataset_path}")
        
        with open(dataset_path) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples")

        outputs = []
        total_start_time = time.time()

        for i, example in enumerate(dataset):
            print(f"\nProcessing example {i+1}/{len(dataset)} - ID: {example['id']}")
            
            try:
                start_time = time.time()
                print("Generating response (this may take a while)...")
                
                response = self.model.generate(self.format_prompt(example))
                if response:
                    elapsed = time.time() - start_time
                    
                    output = {
                        'id': example['id'],
                        'answer': example['answer'],
                        'predicted_answer': response.get('answer'),
                        'reasoning_steps': response.get('steps', []),
                        'confidence': response.get('confidence', 'unknown'),
                        'correct': response.get('answer') == example['answer'],
                        'time_taken': elapsed
                    }
                    outputs.append(output)
                    
                    # Progress update
                    print(f"Time taken: {elapsed:.1f}s")
                    print(f"Predicted: {output['predicted_answer']} | Actual: {example['answer']} | {'✓' if output['correct'] else '✗'}")
                    print(f"Confidence: {output['confidence']}")
                    print("\nReasoning steps:")
                    for step in output['reasoning_steps']:
                        print(f"{step['step']}. {step['type'].title()}: {step['content'][:100]}...")
                    
                    # Save progress
                    self._save_outputs(outputs)
                    
            except Exception as e:
                print(f"Error processing example {example['id']}: {str(e)}")
                continue

        # Save final results
        total_time = time.time() - total_start_time
        self._save_final_results(outputs, total_time)
        return outputs

    def _save_outputs(self, outputs: List[Dict]):
        save_path = os.path.join(self.save_path, f'results_intermediate.json')
        with open(save_path, 'w') as f:
            json.dump({'results': outputs}, f, indent=2)

    def _save_final_results(self, outputs: List[Dict], total_time: float):
        if not outputs:
            return
            
        total = len(outputs)
        correct = sum(1 for o in outputs if o.get('correct', False))
        accuracy = correct / total
        avg_time = sum(o.get('time_taken', 0) for o in outputs) / total
        
        stats = {
            'total_examples': total,
            'correct_answers': correct,
            'accuracy': accuracy,
            'average_time_per_example': avg_time,
            'total_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_settings': MODEL_SETTINGS  # Include settings in results
        }
        
        print(f"\nFinal Statistics:")
        print(f"Total examples: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average time per example: {avg_time:.1f}s")
        print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f} hours)")
        
        save_path = os.path.join(self.save_path, f'results_final.json')
        with open(save_path, 'w') as f:
            json.dump({
                'statistics': stats,
                'results': outputs
            }, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="LogicalDeduction", help="Name of the dataset")
    parser.add_argument("--split", default="dev", help="Dataset split to use")
    parser.add_argument("--model_name", default="nemomix-unleashed-12b", help="Model name/identifier")
    parser.add_argument("--max_completion_tokens", type=int, default=-1, help="Maximum tokens for completion")
    parser.add_argument("--data_path", required=True, help="Path to the data directory")
    parser.add_argument("--save_path", required=True, help="Path to save results")
    
    args = parser.parse_args()
    
    baseline = LogicalDeductionBaseline(args)
    baseline.process_dataset()