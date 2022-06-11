# ML_Algorithms_For_Digit_Classification
Comparison of various algorithms for Digit Classification

### Table of Contents

1. [Installation](#installation)
2. [File Descriptions](#files)

## Installation <a name="installation"></a>

The libraries required for the successful execution of this code are mentioned in requirements.txt. In order to install all the libraries:
`pip install -r requirements.txt`

## File Descriptions <a name="files"></a>

- `utils.py` contains various helper functions to read, pre-process, and store the data. Furthermore, functions for training and testing various prompts are also included.
- `preprocess_claim_ident.py` contains the functions to pre-process the data and store it in json and pickle format
- `baseline.py` contains the code to train and evaluate the baseline model on the binary and 3-label claim identification
- `base_prompt_engg.py` contains the code to test the different prompts for the identification without any fine-tuning
- `prompt_fine_tuning.py` contains the code to fine-tune the model using the prompt for only binary claim identificaiton

- `main.py` uses the code from above files to pre-process the data, run the baselines, and the prompt engineering task. 

Before doing the fine-tuning, you need to login to your huggingface account using the command line. You can check how to do it [here](!https://huggingface.co/docs/transformers/model_sharing)

After that you can run it with command line argument of your account id, for instance

```python main.py -a nouman-10```


