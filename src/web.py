import requests
from eval import QAEvaluator

class WebAPIQAPipeline:
    def __init__(self, model_name="deepset/bert-large-uncased-whole-word-masking-squad2"):
        self.API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": "Bearer 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"}  # Replace with your token
        
    def __call__(self, question: str, context: str) -> str:
        try:
            payload = {
                "inputs": {
                    "question": question,
                    "context": context
                }
            }
            response = requests.post(self.API_URL, headers=self.headers, json=payload)
            output = response.json()
            #print(output)
            
            return output.get('answer', '')
            
        except Exception as e:
            print(f"Error in API call: {e}")
            return ""

# Usage example:
web_pipeline = WebAPIQAPipeline()
web_evaluator = QAEvaluator('out/api_results.json')

# Evaluate pipeline with context.txt and questions.json
web_evaluator.evaluate_pipeline(
    pipeline=web_pipeline,
    questions_file='data/questions.json',
    context_file='data/context.txt'
)