from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import List, Dict, Any
import torch
from eval import QAEvaluator

class FineTunedQAPipeline:
    def __init__(self, model_name="deepset/bert-large-uncased-whole-word-masking-squad2"):
        self.device = torch.device('cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch.cuda.empty_cache()
        
        # Load SQuAD fine-tuned model with optimizations
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.max_length = 384
        self.model.eval()

    def __call__(self, question: str, context: str) -> str:
        try:
            inputs = self.tokenizer(
                question,
                context,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_offsets_mapping=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
                
                start_logits = outputs.start_logits[0]
                end_logits = outputs.end_logits[0]
                
                # Get the most likely beginning and end of answer
                start_idx = torch.argmax(start_logits)
                end_idx = torch.argmax(end_logits)
                
                answer_tokens = inputs.input_ids[0][start_idx:end_idx + 1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                del outputs
                
                return answer.strip()
                
        except Exception as e:
            print(f"Error in question answering: {e}")
            return ""
        
    def __del__(self):
        del self.model
        del self.tokenizer

ft_pipeline = FineTunedQAPipeline()
ft_evaluator = QAEvaluator('out/finetuned_results.json')

# Evaluate pipeline with context.txt and questions.json
ft_evaluator.evaluate_pipeline(
    pipeline=ft_pipeline,
    questions_file='data/questions.json',
    context_file='data/context.txt'
)