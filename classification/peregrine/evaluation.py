print("Importing...")
from transformers import pipeline, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
import pickle
import numpy as np
from SemanticTypeAnalyzer import SemanticTypeAnalyzer

print("Load Training data...")

with open('data_train.p', 'rb') as f:
    data_train = pickle.load(f)

print("Intializing...")

data = np.load('dataset.npz')
X = data['X']
y = data['y']

analyzer = SemanticTypeAnalyzer(data_train)

print("Analyzing...")

preds = analyzer.predict(X)

np.savez('results_interp_judge.npz', preds=preds, y=y)


