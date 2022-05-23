print("Importing...")
from transformers import pipeline, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
import pickle
import numpy as np
from SemanticTypeAnalyzer import SemanticTypeAnalyzer

print("Done Importing")

with open('data_train.p', 'rb') as f:
    data_train = pickle.load(f)

print("Done Loading training data")

data = np.load('dataset.npz')
X = data['X'][:1]
y = data['y'][:1]

analyzer = SemanticTypeAnalyzer(data_train)

print("Initialized analyzer")

preds = analyzer.predict(X)

np.savez('results_opt_test.npz', preds=preds, y=y)


