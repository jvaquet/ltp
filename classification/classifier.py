from transformers import pipeline, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
import pickle
import numpy as np

class SemanticTypeAnalyzer:
    
    def __init__(self, data_train):
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2.config.max_length = 1024
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.generator = pipeline("text-generation", model=self.gpt2, tokenizer=self.tokenizer)
        
        self.data_train = data_train
        self.classes = list(data_train.keys())
        self.classes_text = list(map(lambda x: x.split('_')[-1], self.classes))
        
    def generate_prompt(self, phrase, n_examples=15):
        prompt = ''
        for i in range(n_examples):
            cls_idx = np.random.randint(0, len(self.classes))
            cls_name_prompt = self.classes_text[cls_idx]
            cls_name = self.classes[cls_idx]
            sample_idx = np.random.randint(0, len(self.data_train[cls_name]))
            sample = self.data_train[cls_name][sample_idx]
            prompt += f'Assign this phrase one of the five types {", ".join(self.classes_text)}:\n'
            prompt += f'"{sample}"\nType: {cls_name_prompt}\n\n'
        prompt += f'"{phrase}"\nType:'
        return prompt

    def evaluate_prompt(self, prompt, n_samples=5):
        len_prompt = len(prompt)
        len_answer = len(self.tokenizer(prompt).input_ids) + 5
        res = self.generator(prompt, max_length=len_answer, num_return_sequences=n_samples)
        return list(map(lambda x: x['generated_text'][len_prompt:].split('\n')[0].replace(' ', ''), res))
    
    def predict_claim(self, claim, n_prompts=10, n_samples=5):
        preds = []
        for _ in range(n_prompts):
            prompt = self.generate_prompt(claim)
            preds += self.evaluate_prompt(prompt, n_samples=n_samples)

        results = {}
        for cls in self.classes_text:
            results[cls] = preds.count(cls)
        results['other_text'] = list(filter(lambda x: x not in self.classes_text, preds))
        results['other'] = len(results['other_text'])
        return results


with open('data_train.p', 'rb') as f:
    data = pickle.load(f)

analyzer = SemanticTypeAnalyzer(data)
pred = analyzer.predict_claim('Yes, I agree!')
print(pred)