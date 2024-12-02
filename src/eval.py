import json
import time
import psutil
import numpy as np
from typing import Dict, Any

from transformers import AutoTokenizer, AutoModel

# Initialize the LLM
model = AutoModel.from_pretrained("GEB-AGI/geb-1.3b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("GEB-AGI/geb-1.3b", trust_remote_code=True)

# Create LLM wrapper for semantic scoring
def llm_wrapper(prompt):
    response, _ = model.chat(tokenizer, prompt, history=[])
    return response


prompts = {
    'single_choice': "Respond only with the exact text of the chosen option, nothing else.",
    'multiple_choice': "Format your response as a comma-separated list of the exact option texts, without any additional words or punctuation.",
    'fill_blank': "Provide only the missing word/phrase without any additional context or punctuation.",
    'true_false': "Respond with only the word 'true' or 'false', nothing else"
}


class QAEvaluator:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.results = []
        self.llm = llm_wrapper

    def calculate_em_score(self, prediction: str, ground_truth: str) -> float:
        return float(prediction.lower().strip() == ground_truth.lower().strip())

    def calculate_f1_score(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def semantic_match(self, prediction: str, ground_truth: str) -> float:
        prompt = f"""Compare these answers semantically and respond with a score between 0 or 1:
Predicted: {prediction}
Actual: {ground_truth}
Score (0 or 1):"""
        
        try:
            score = float(self.llm(prompt))
            return min(max(score, 0), 1)
        except:
            return 0.0

    def measure_response_time(self, start_time: float) -> float:
        return time.time() - start_time

    def track_resource_utilization(self) -> Dict[str, float]:
        process = psutil.Process()
        return {
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent()
        }

    def evaluate_question(self, pipeline: Any, question: Dict, context: str = None) -> Dict:
        start_time = time.time()
        prompt = prompts[question['type']]
        
        if question['type'] in ['single_choice', 'multiple_choice']:
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in question['options']])
            augmented_question = f"{prompt}\n\nQuestion: {question['question']}{options_text}"
        else:
            augmented_question = f"{prompt}\n\nQuestion: {question['question']}"

        if context:
            prediction = pipeline(question=augmented_question, context=context)
        else:
            prediction = pipeline(augmented_question)
        
        print(prediction)
        
        metrics = {
            'question_id': question['id'],
            'question_type': question['type'],
            'question': augmented_question,
            'predicted_answer': prediction,
            'actual_answer': question['answer'],
            'exact_match': self.calculate_em_score(prediction, str(question['answer'])),
            'f1_score': self.calculate_f1_score(prediction, str(question['answer'])),
            'semantic_score': self.semantic_match(prediction, str(question['answer'])),
            'response_time': self.measure_response_time(start_time),
            'resource_usage': self.track_resource_utilization()
        }
        return metrics

    def evaluate_pipeline(self, pipeline: Any, questions_file: str, context_file: str = None) -> None:
        # Load questions
        with open(questions_file, 'r') as f:
            questions = json.load(f)['questions']
        
        # Load context if provided
        context = None
        if context_file:
            with open(context_file, 'r') as f:
                context = f.read()

        # Evaluate each question
        for question in questions:
            result = self.evaluate_question(pipeline, question, context)
            self.results.append(result)

        # Calculate aggregate metrics
        aggregate_metrics = {
            'average_exact_match': np.mean([r['exact_match'] for r in self.results]),
            'average_f1_score': np.mean([r['f1_score'] for r in self.results]),
            'average_semantic_score': np.mean([r['semantic_score'] for r in self.results]),
            'average_response_time': np.mean([r['response_time'] for r in self.results]),
            'total_questions': len(self.results)
        }

        # Write results to output file
        with open(self.output_file, 'w') as f:
            json.dump({
                'individual_results': self.results,
                'aggregate_metrics': aggregate_metrics
            }, f, indent=2)