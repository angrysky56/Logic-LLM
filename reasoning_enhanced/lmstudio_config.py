"""Configuration for LM Studio API"""

MODEL_SETTINGS = {
    "model": "nemomix-unleashed-12b",
    "temperature": 0.1,       # Lower for more deterministic logical reasoning
    "top_p": 0.1,            # Lower for more focused sampling
    "top_k": 40,             # Reasonable default for focused outputs
    "max_tokens": -1,        # Let model complete thoughts
    "presence_penalty": 0.0,  # Default
    "frequency_penalty": 0.0, # Default
    "repeat_penalty": 1.1,    # Slight penalty for repetition
    "seed": 42,              # Fixed seed for reproducibility
    "stream": False,         # Complete response needed
    "stop": None,            # No early stopping
    "logit_bias": {}         # No bias adjustment needed
}