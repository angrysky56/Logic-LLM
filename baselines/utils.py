import os
import openai
from typing import Any

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens, base_url="http://localhost:1234/v1") -> None:
        openai.api_key = API_KEY  # Can be any string for LM Studio
        openai.api_base = base_url
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    def chat_generate(self, input_string, temperature=0.7):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves logical deduction problems step by step."},
                    {"role": "user", "content": input_string}
                ],
                temperature=temperature,
                max_tokens=self.max_new_tokens,
                stop=self.stop_words
            )
            generated_text = response['choices'][0]['message']['content'].strip()
            return generated_text
        except Exception as e:
            print(f"Error in chat_generate: {str(e)}")
            return ""

    def generate(self, input_string, temperature=0.7):
        return self.chat_generate(input_string, temperature)

    def batch_generate(self, messages_list, temperature=0.7):
        outputs = []
        for message in messages_list:
            output = self.generate(message, temperature)
            outputs.append(output)
        return outputs