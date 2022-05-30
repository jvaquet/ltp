from transformers import pipeline
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
#from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pickle

class SemanticTypeAnalyzer:
    
    def __init__(self, data_train):
        #self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        #self.gpt2.config.max_length = 1024
        #self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #self.generator = pipeline("text-generation", model=self.gpt2, tokenizer=self.tokenizer)
        
        # We saw that GPT-neo worked best
        self.gpt2 = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
        self.gpt2.config.max_length = 1024
        self.tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', device=0)
        self.generator = pipeline("text-generation", model=self.gpt2, tokenizer=self.tokenizer, device=0)

        #self.gpt2 = AutoModelForCausalLM.from_pretrained('facebook/opt-6.7b')
        #self.gpt2.config.max_length = 1024
        #self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-6.7b', device=0)
        #self.generator = pipeline("text-generation", model=self.gpt2, tokenizer=self.tokenizer, device=0)


        self.data_train = data_train
        self.classes = list(data_train.keys())
        self.classes_text = list(map(lambda x: x.split('_')[-1], self.classes))
        self.classes_prompt = self.classes_text.copy()
        self.classes_prompt[0] = 'perception'
        self.all_res = []
        
    def generate_prompt(self, phrase, n_examples=15):
        prompt = ''
        for i in range(n_examples):
            cls_idx = np.random.randint(0, len(self.classes))
            cls_name_prompt = self.classes_prompt[cls_idx]
            cls_name = self.classes[cls_idx]
            sample_idx = np.random.randint(0, len(self.data_train[cls_name]))
            sample = self.data_train[cls_name][sample_idx]
            prompt += f'Assign this phrase one of the five types {", ".join(self.classes_prompt)}:\n'
            prompt += f'"{sample}"\nType: {cls_name_prompt}\n\n'
        prompt += f'"{phrase}"\nType:'
        return prompt

    def evaluate_prompt(self, prompt, n_samples=5):
        len_prompt = len(prompt)
        len_answer = len(self.tokenizer(prompt).input_ids) + 5
        res = self.generator(prompt, max_length=len_answer, num_return_sequences=n_samples)
        return list(map(lambda x: x['generated_text'][len_prompt:].split('\n')[0].replace(' ', ''), res))
    
    def predict_claim(self, claim, n_prompts=10, n_samples=5):
        isInterpretation = lambda x: 'percept' in x
        isRational = lambda x: 'ration' in x
        isEmotional = lambda x: 'emotion' in x
        isAgreement = lambda x: 'agree' in x and not 'disagree' in x
        isDisagreement = lambda x: 'disagree' in x

        results = {
            'interpretation': 0,
            'rational': 0,
            'emotional': 0,
            'agreement': 0,
            'disagreement': 0
        }

        for _ in range(n_prompts):
            prompt = self.generate_prompt(claim)
            preds = self.evaluate_prompt(prompt, n_samples=n_samples)
            
            for pred in preds:
                if isInterpretation(pred):
                    results['interpretation'] += 1
                if isRational(pred):
                    results['rational'] += 1
                if isEmotional(pred):
                    results['emotional'] += 1
                if isAgreement(pred):
                    results['agreement'] += 1
                if isDisagreement(pred):
                    results['disagreement'] += 1

        return results
    
    def predict_probas(self, claim, n_prompts=10, n_samples=5):
        res = self.predict_claim(claim, n_prompts=n_prompts, n_samples=n_samples)
        self.all_res.append(res)
        probas = np.zeros(len(self.classes_text))
        for i, cls in enumerate(self.classes_text):
            probas[i] = res[cls]
        probas /= np.sum(probas)
        return probas
        
    def predict(self, X, n_prompts=10, n_samples=5):
        probas = np.array(list(map(lambda x: self.predict_probas(x, n_prompts=n_prompts, n_samples=n_samples), X)))
        with open("outputs_analysis.p", "wb") as f:
            pickle.dump(self.all_res, f)
        return probas
