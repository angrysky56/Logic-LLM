import json
import requests
from typing import Dict, List, Optional
from base_validator import LogicalValidator

class MultiPathReasoner:
    def __init__(self, model_path="nemomix-unleashed-12b", base_url="http://localhost:1234/v1"):
        self.validator = LogicalValidator()
        self.model_path = model_path
        self.base_url = base_url
        
        # Different reasoning prompts for diversity
        self.prompts = [
            # Direct logical deduction
            """Solve this logical deduction step by step:
            1. First, write down all given facts
            2. Then, make logical deductions one at a time
            3. Finally, state which option must be true
            
            {context}
            {question}
            Options:
            {options}
            """,
            
            # Elimination approach
            """Let's solve this by eliminating incorrect options:
            1. Start with all options: {options}
            2. Use each fact to eliminate impossible options
            3. The remaining option must be true
            
            {context}
            {question}
            """,
            
            # Forward chaining
            """Let's solve this by building up from what we know:
            1. Start with the clearest fact
            2. See what must logically follow
            3. Continue until we can prove one option
            
            {context}
            {question}
            Options:
            {options}
            """
        ]

    def get_prediction(self, prompt: str) -> str:
        """Get a single prediction from the model"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_path,
            "messages": [
                {"role": "system", "content": "You are a logical reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return None

    def get_multi_path_prediction(self, example: Dict) -> Dict:
        """Get predictions using multiple reasoning approaches"""
        context = example.get('context', '')
        question = example.get('question', '')
        options = example.get('options', [])
        
        # Format options text
        options_text = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        
        # Get predictions using different prompts
        predictions = []
        for prompt_template in self.prompts:
            prompt = prompt_template.format(
                context=context,
                question=question,
                options=options_text
            )
            
            prediction = self.get_prediction(prompt)
            if prediction:
                validated = self.validator.validate_reasoning(prediction, None)
                if validated[0]:  # if valid
                    predictions.append({
                        'reasoning': prediction,
                        'confidence': validated[1],
                        'error': validated[2]
                    })
        
        # No valid predictions
        if not predictions:
            return {
                'predicted_reasoning': '',
                'predicted_answer': '',
                'confidence': 0.0,
                'error': 'No valid predictions'
            }
            
        # Sort by confidence and return best
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        best_pred = predictions[0]
        
        return {
            'predicted_reasoning': best_pred['reasoning'],
            'predicted_answer': self.validator._extract_answer(best_pred['reasoning']),
            'confidence': best_pred['confidence'],
            'error': ''
        }

    def process_batch(self, examples: List[Dict]) -> List[Dict]:
        """Process examples using multiple reasoning paths"""
        results = []
        for example in examples:
            prediction = self.get_multi_path_prediction(example)
            results.append({
                **example,
                'predicted_reasoning': prediction['predicted_reasoning'],
                'predicted_answer': prediction['predicted_answer'],
                'confidence': prediction['confidence'],
                'error': prediction['error']
            })
        return results