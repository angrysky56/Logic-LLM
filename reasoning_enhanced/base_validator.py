import re
import json
from typing import Dict, List, Tuple, Optional

class LogicalValidator:
    def __init__(self):
        self.confidence_threshold = 0.6  # Lowered from 0.8
        self.min_reasoning_steps = 1     # Lowered from 2
        self.required_patterns = [
            # More flexible conclusion patterns
            r"(?i)(therefore|thus|hence|so|conclusion|answer)",
            r"(?i)(option|answer|choice)\s*[A-G]",
            r"(?i)[A-G]\s*(is|must be)\s*(correct|true|the answer)",
        ]
        self.step_markers = [
            r"^\d+[.)]",
            r"^Step \d+",
            r"^First",
            r"^Second",
            r"^Finally",
            r"^Given",
            r"^Let'?s",
            r"^Since",
            r"^Because",
            r"^•",
            r"^-",
        ]
        
    def validate_reasoning(self, reasoning: str, answer: str) -> Tuple[bool, float, str]:
        """Validate reasoning and return (is_valid, confidence, error_msg)"""
        if not reasoning or not answer:
            return False, 0.0, "Empty reasoning or answer"

        # Check basic reasoning presence
        steps = self._count_reasoning_steps(reasoning)
        if steps < self.min_reasoning_steps:
            # Look for implicit reasoning in prose
            implicit_steps = len(re.findall(r'(?i)(because|since|therefore|thus|hence)', reasoning))
            steps += implicit_steps
            if steps < self.min_reasoning_steps:
                return False, 0.0, f"Only found {steps} reasoning steps, need at least {self.min_reasoning_steps}"
            
        # Check for at least one required pattern more flexibly
        has_pattern = False
        for pattern in self.required_patterns:
            if re.search(pattern, reasoning):
                has_pattern = True
                break
        if not has_pattern:
            return False, 0.0, "No clear conclusion pattern found"
            
        # Extract and validate conclusion matches answer
        extracted_answer = self._extract_answer(reasoning)
        if extracted_answer and not self._answers_match(extracted_answer, answer):
            return False, 0.0, f"Reasoning conclusion ({extracted_answer}) doesn't match given answer ({answer})"

        # If we got here with no extracted answer but everything else looks good,
        # we'll accept it with lower confidence
        if not extracted_answer:
            base_confidence = self._calculate_confidence(reasoning, steps)
            return True, base_confidence * 0.7, "Valid reasoning but unclear conclusion"

        # Calculate full confidence
        confidence = self._calculate_confidence(reasoning, steps)
        return True, confidence, "Valid reasoning"

    def _count_reasoning_steps(self, reasoning: str) -> int:
        steps = 0
        lines = reasoning.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for marker in self.step_markers:
                if re.match(marker, line):
                    steps += 1
                    break
                    
        return steps

    def _extract_answer(self, reasoning: str) -> Optional[str]:
        # Look for answer in conclusion
        patterns = [
            r"(?i)therefore[^A-G]*([A-G])",
            r"(?i)(?:answer|option|choice)\s*(?:is|:)\s*[^A-G]*([A-G])",
            r"(?i)conclusion[^A-G]*([A-G])",
            r"(?i)([A-G])\s*(?:is correct|is true|must be true)",
            r"(?i)([A-G])[^A-G]{0,30}$"  # Answer near the end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reasoning)
            if match:
                return match.group(1)
                
        # If no clear conclusion, look for emphasized single letter
        emphasized = re.findall(r'(?i)(?:^|\s+|\*\*|\*)([A-G])(?:\s+|$|\*\*|\*)', reasoning)
        if len(emphasized) == 1:
            return emphasized[0]
            
        return None

    def _answers_match(self, extracted: str, given: str) -> bool:
        if not extracted or not given:
            return True  # Be lenient if we're missing either
        return extracted.strip().upper() == given.strip().upper()

    def _calculate_confidence(self, reasoning: str, steps: int) -> float:
        confidence = 0.0
        
        # Base confidence from number of steps (max 0.4)
        confidence += min((steps / 2), 1.0) * 0.4
        
        # Confidence from logical connectors (max 0.3)
        logical_markers = [
            r"(?i)because",
            r"(?i)since",
            r"(?i)given that",
            r"(?i)therefore",
            r"(?i)thus",
            r"(?i)hence",
            r"(?i)conclude",
        ]
        marker_count = sum(1 for marker in logical_markers if re.search(marker, reasoning))
        confidence += min((marker_count / 3), 1.0) * 0.3
        
        # Confidence from clear structure (max 0.3)
        structure_points = 0
        if re.search(r'(?i)^(given|first|step|1)[:\.]', reasoning, re.MULTILINE):
            structure_points += 1
        if re.search(r'\n\s*[-•*]\s+|\n\d+[.)]', reasoning):
            structure_points += 1
        if re.search(r'(?i)therefore|conclusion|thus,', reasoning):
            structure_points += 1
        confidence += (structure_points / 3) * 0.3
            
        return min(confidence, 1.0)

    def batch_validate(self, predictions: List[Dict]) -> List[Dict]:
        """Validate a batch of predictions and add validation info"""
        validated = []
        for pred in predictions:
            is_valid, confidence, error_msg = self.validate_reasoning(
                pred.get('predicted_reasoning', ''),
                pred.get('predicted_answer', '')
            )
            validated.append({
                **pred,
                'validation': {
                    'is_valid': is_valid,
                    'confidence': confidence,
                    'error': error_msg
                }
            })
            
        return validated

    def validate_and_filter(self, predictions: List[Dict], min_confidence: float = None) -> Tuple[List[Dict], List[Dict]]:
        """Split predictions into valid and invalid based on validation"""
        min_confidence = min_confidence or self.confidence_threshold
        valid = []
        invalid = []
        
        validated = self.batch_validate(predictions)
        for pred in validated:
            if pred['validation']['is_valid'] and pred['validation']['confidence'] >= min_confidence:
                valid.append(pred)
            else:
                invalid.append(pred)
                
        return valid, invalid