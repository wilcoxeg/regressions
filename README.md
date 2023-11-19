# Regressions

Data and scripts for running experiments on information theory and regressions in reading. 

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

### Data

* You'll need to download `Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv` from https://osf.io/sjefs/

* `regression_df.csv` --> you should be able to use this for analysis
* `provo_refs.tsv` is a reference dataframe with text id, word id, and word in sentence idx for provo
* `provo_sents.csv` is lowercased and tokenized sentences from the provo corpus. There are problems with sentences in four texts: Story 21 (about neuroscience and music), story 34 (lady Gaga's youtube account), story 45 (Susan B. Anthony) and Story 55 (Voltaire)

### Scripts

* `get_predictors.py` iterates through `provo_sents.tsv` and gets PMI, E[PMI], E[PPMI] and E[NPMI] for each word pair in the sentence. A sample call is below:

`python get_predictors.py --input_path ../data/test/test_sents.csv --dataset test --mlm_model bert-en --ar_model gpt2 --language en --mask_type none`

Where `dataset` is in `{test, provo, dundee, ucl, meco}` , `mlm_model` is in `bert-en, mbert, bert-tr, bert-it, bert-de, bert-sp, bert-fi, bert-nl, bert-ru`, `ar-model`, and mask type is either `all, none, mask, truncate` (corresponding to the masking settings)

* `merge_results.Rmd` merges the results of the above script with the eyetracking data, to obtain the number of regressions between each pair of words in the corpus.

## Analysis

* `analysis_scripts_en` contains the analysis scripts for the first experiment. The raw by-corpus data are read in and merged with `clean_data.Rmd`

The main analysis is computed in `en_analysis.Rmd` with the other two scripts and the other two scripts computing secondary analysis, i.e., the correlations between variables and the ZIP/Poisson model comparison

* `analysis_scripts_meco` contains the analysis for the second experiment, this time in a single file.

* `images` contains images, as well as some powerpoint files used for combining PDFs into a single image








