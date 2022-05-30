# Claim Classification
This folder contains all the code and data for the classifciation task

## Ipython Notebooks
In this folder, there are several Ipython notebooks:
* `Analysis.ipynb`: Analyse and compare results of three classification results of the same pipeline.
* `AnalysisSingle.ipynb`: Analyze the results of a pipeline.
* `CreateDataset.ipynb`: Creates the various dataset files used. Requires the original data to be located in `data/p` and `data/n` respectively.
* `EnhancedExtraction.ipynb`: Convert an output pickle generated by the naive class extractor into class probabilities using the more advanced extractor. The advanced extractor was developed in this notebook.
* `Eval.ipynb`: Early, local evaluation for the developed pipeline. Once a certain level of sophistication was reached, the inference step was moved to perigrine.
* `ExplicitPrompts.ipynb`: Local development and testing of the classification pipeline. Later refactored into python files to be used on perigrine.
* `krippendorff_alpha.py` Helper function file to calculate the krippendorff alpha.

## Peregrine-Folder
This folder contains all the files used on perigrine, for different experiments they were modified accordingly.
* `evaluation.py`: Loads data and intializes a SemanticTypeAnalyzer, performs the prediction step and then saves the results
* `SemanticTypeAnalyzer.py`: Python class that takes claims as an input and returns their class likelihoods as an output.
* `job_eval.sh`: Jobscript to start the experiment in batch mode.

## Results
The results folder contains the outputs from various runs of the experiment on perigrine. The `results_*.npz` files contain numpy arrays with the class likelihoods:

|File           | Class Extractor   | LM        | Comments |
|---------------|-------------------|-----------|----------|
|`results_#.npz`| Naive             | GPT-2     | -        |
|`results_neo_#.npz`| Naive         | GPT-Neo   | -        |
|`results_neo_#_enh.npz`| Advanced  | GPT-Neo   | -        |
|`results_interp_analysis.npz`|Advanced| GPT-Neo| The `inerpratation` class is prompted as `analysis`|
|`results_interp_judgement.npz`|Advanced| GPT-Neo| The `inerpratation` class is prompted as `judgement`|
|`results_interp_perception.npz`|Advanced| GPT-Neo| The `inerpratation` class is prompted as `perception`|

The `#` is a placeholder, either for the number of the pass on validation data, or `test` if the experimetn was performed on the test data.


The `outputs_*.p` are pickles with all generated output:
|File           | Class Extractor   | LM        | Comments |
|---------------|-------------------|-----------|----------|
|`outputs_neo_#.p`| Naive           | GPT-Neo   | -        |
|`outputs_analysis.p`| Advanced     | GPT-Neo   |The `inerpratation` class is prompted as `analysis`|
|`outputs_judgement.p`| Advanced    | GPT-Neo   |The `inerpratation` class is prompted as `judgement`|
|`outputs_perception.p`| Advanced   | GPT-Neo   |The `inerpratation` class is prompted as `perception`|

The `#` is a placeholder, either for the number of the pass on validation data, or `test` if the experimetn was performed on the test data.