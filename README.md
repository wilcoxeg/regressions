# Regressions

Data and scripts for running experiments on information theory and regressions in reading

## Install Dependencies

Create a conda environment with
```bash
$ conda env create -f environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers
```

## Analysis

* `clean_data` takes the results and outputs `regression_df.csv` in the data directory, which can be used for analysis.
* `analysis` runs the regressions

### Scripts

* `get_predictors.py` iterates through `provo_sents.tsv` and gets pmi, kld and top_k for each word/word pair in the sentence.

### Data

* You'll need to download `Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv` from https://osf.io/sjefs/

* `regression_df.csv` --> you should be able to use this for analysis
* `provo_refs.tsv` is a reference dataframe with text id, word id, and word in sentence idx for provo
* `provo_sents.csv` is lowercased and tokenized sentences from the provo corpus. There are problems with sentences in four texts: Story 21 (about neuroscience and music), story 34 (lady Gaga's youtube account), story 45 (Susan B. Anthony) and Story 55 (Voltaire)
