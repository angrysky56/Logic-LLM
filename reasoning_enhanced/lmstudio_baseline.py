import json
import os
import time
import requests
import re
from typing import List, Dict, Any, Optional
from lmstudio_config import MODEL_SETTINGS

class LMStudioModel:
    """
    A client for sending requests to an LM Studio server (running locally on http://localhost:1234/v1).
    It initializes with certain model settings (e.g., max_tokens), verifies the connection to the server,
    and provides a method (generate) to get completions from the model.
    """
    def __init__(self, max_tokens: Optional[int] = None):
        """
        Initialize an LMStudioModel instance.

        Args:
            max_tokens (Optional[int]): If provided, overrides the default number of tokens for generation.
        """
        self.base_url = "http://localhost:1234/v1"
        self.settings = MODEL_SETTINGS.copy()
        if max_tokens is not None:
            self.settings["max_tokens"] = max_tokens
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Attempt to verify the connection to the LM Studio server by calling the /models endpoint."""
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code == 200:
                print("Successfully connected to LM Studio server.")
            else:
                print(f"Warning: Could not verify LM Studio server. Status code: {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not verify LM Studio server: {e}")
            print("Please ensure LM Studio is running with the API server enabled.")

    def generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generate a response from the model based on the given prompt.

        Args:
            prompt (str): The user query or request to be passed to the model.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the JSON-formatted model output,
            or None if generation fails.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a logical reasoning expert. Respond in this exact JSON format:\n"
                    "{\n"
                    '    "steps": [\n'
                    '        {"step": 1, "type": "facts", "content": "List all given facts"},\n'
                    '        {"step": 2, "type": "analysis", "content": "Analyze relationships"},\n'
                    '        {"step": 3, "type": "deduction", "content": "Apply logical deduction"},\n'
                    '        {"step": 4, "type": "conclusion", "content": "State which option (A-E) must be correct and why"}\n'
                    "    ],\n"
                    '    "answer": "A",\n'
                    '    "confidence": "high"\n'
                    "}"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Prepare the request data
        data = {
            "messages": messages,
            **self.settings  # Include all model settings
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=data
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                try:
                    # Attempt direct JSON parse
                    return json.loads(content)
                except json.JSONDecodeError:
                    print("Response was not valid JSON, attempting to parse with regex...")
                    return self._parse_non_json_response(content)
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Error in generation: {str(e)}")
        
        return None

    def _parse_non_json_response(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information from a response that failed direct JSON parsing,
        using regular expressions to approximate steps, answer, and confidence.

        Args:
            text (str): The raw text response from the model.

        Returns:
            Dict[str, Any]: The best guess of a structured result, containing steps, answer, and confidence.
        """
        steps = []
        answer = None
        confidence = "low"

        # Regex pattern to capture step-like blocks
        step_pattern = r"(?:^|\n)(\d+)[.).]\s*(.+?)(?=(?:\n\d+[.).])|$)"
        for match in re.finditer(step_pattern, text, re.DOTALL):
            step_num = int(match.group(1))
            content = match.group(2).strip()
            # Map step number to a type label: 1->facts, 2->analysis, 3->deduction, 4->conclusion
            step_type_index = min(step_num - 1, 3)
            step_type = ["facts", "analysis", "deduction", "conclusion"][step_type_index]
            steps.append({
                "step": step_num,
                "type": step_type,
                "content": content
            })

        # Regex to detect an answer (A-E) in text
        answer_pattern = r"(?i)(?:answer|conclusion).*?([A-E])"
        match_answer = re.search(answer_pattern, text)
        if match_answer:
            answer = match_answer.group(1).upper()
            confidence = "medium"  # Adjust confidence based on the uncertain extraction

        return {
            "steps": steps,
            "answer": answer,
            "confidence": confidence
        }

class LogicalDeductionBaseline:
    """
    A baseline tool for processing a logical deduction dataset. It uses an LMStudioModel
    to generate reasoning steps and final answers. The results are then stored in JSON.
    """
    def __init__(self, args):
        """
        Args:
            args: Command-line arguments containing dataset info, model settings, and paths.
        """
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.save_path = args.save_path
        
        # Initialize model
        self.model = LMStudioModel(max_tokens=args.max_completion_tokens)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Create a formatted prompt string for the given example.

        Args:
            example (Dict[str, Any]): An item from the dataset containing context, question, options, etc.

        Returns:
            str: A formatted prompt to be sent to the model.
        """
        context = example['context'].replace("\\n", "\n")
        options_text = "\n".join(example['options'])

        prompt = (
            "Solve this logical reasoning problem. Provide your solution in JSON format.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{example['question']}\n\n"
            f"Options:\n{options_text}"
        )
        return prompt

    def process_dataset(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset from a JSON file, iterates over each example, requests a model response,
        and collects the outputs (including correctness, confidence, and reasoning steps).

        Returns:
            List[Dict[str, Any]]: A list of result dictionaries for each processed example.
        """
        dataset_path = os.path.join(self.data_path, self.dataset_name, f'{self.split}.json')
        print(f"Loading dataset from: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        print(f"Loaded {len(dataset)} examples.")
        outputs = []
        total_start_time = time.time()

        for i, example in enumerate(dataset):
            print(f"\nProcessing example {i+1}/{len(dataset)} - ID: {example['id']}")
            print("Generating response (this may take a while)...")

            try:
                start_time = time.time()
                prompt = self.format_prompt(example)
                response = self.model.generate(prompt)
                elapsed = time.time() - start_time

                if response:
                    predicted_answer = response.get('answer')
                    is_correct = (predicted_answer == example['answer'])
                    
                    output = {
                        'id': example['id'],
                        'answer': example['answer'],
                        'predicted_answer': predicted_answer,
                        'reasoning_steps': response.get('steps', []),
                        'confidence': response.get('confidence', 'unknown'),
                        'correct': is_correct,
                        'time_taken': elapsed
                    }
                    outputs.append(output)

                    # Print a quick status of the result
                    print(f"Time taken: {elapsed:.1f}s")
                    print(f"Predicted: {predicted_answer} | Actual: {example['answer']} | {'✓' if is_correct else '✗'}")
                    print(f"Confidence: {output['confidence']}")

                    # Optionally, show some snippet of the reasoning steps
                    if output['reasoning_steps']:
                        print("\nReasoning steps:")
                        for step in output['reasoning_steps']:
                            snippet = step['content'][:100] + ("..." if len(step['content']) > 100 else "")
                            print(f"  {step['step']}. {step['type'].capitalize()}: {snippet}")

                    # Save partial results to disk (progress save)
                    self._save_outputs(outputs)

            except Exception as e:
                print(f"Error processing example {example['id']}: {e}")
                continue

        total_time = time.time() - total_start_time
        self._save_final_results(outputs, total_time)
        return outputs

    def _save_outputs(self, outputs: List[Dict[str, Any]]) -> None:
        """
        Save current results in a JSON file (incremental or intermediate results).

        Args:
            outputs (List[Dict[str, Any]]): The current accumulated results.
        """
        save_path = os.path.join(self.save_path, 'results_intermediate.json')
        with open(save_path, 'w') as f:
            json.dump({'results': outputs}, f, indent=2)

    def _save_final_results(self, outputs: List[Dict[str, Any]], total_time: float) -> None:
        """
        After processing all examples, compute summary statistics and save final results.

        Args:
            outputs (List[Dict[str, Any]]): The list of processed results.
            total_time (float): Elapsed time for the entire run, in seconds.
        """
        if not outputs:
            print("No outputs to save.")
            return

        total = len(outputs)
        correct = sum(o.get('correct', False) for o in outputs)
        accuracy = correct / total if total > 0 else 0.0
        avg_time = sum(o.get('time_taken', 0) for o in outputs) / total if total > 0 else 0.0
        
        stats = {
            'total_examples': total,
            'correct_answers': correct,
            'accuracy': accuracy,
            'average_time_per_example': avg_time,
            'total_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_settings': MODEL_SETTINGS
        }

        print("\nFinal Statistics:")
        print(f"Total examples: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average time per example: {avg_time:.1f}s")
        print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f} hours)")

        save_path = os.path.join(self.save_path, 'results_final.json')
        with open(save_path, 'w') as f:
            json.dump({'statistics': stats, 'results': outputs}, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="LogicalDeduction", help="Name of the dataset.")
    parser.add_argument("--split", default="dev", help="Which dataset split to use.")
    parser.add_argument("--model_name", default="nemomix-unleashed-12b", help="Model name/identifier.")
    parser.add_argument("--max_completion_tokens", type=int, default=-1, help="Maximum tokens for completion.")
    parser.add_argument("--data_path", required=True, help="Path to the data directory.")
    parser.add_argument("--save_path", required=True, help="Path to save results.")
    
    args = parser.parse_args()
    baseline = LogicalDeductionBaseline(args)
    baseline.process_dataset()
