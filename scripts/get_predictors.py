
import os
import re
import torch
import argparse
import pandas as pd
import numpy as np
from string import punctuation
import sentencepiece
from torch.nn import functional as F
from wordfreq import zipf_frequency

from transformers import BertTokenizer, BertForMaskedLM
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from transformers import logging
logging.set_verbosity_error()


MODEL_NAME = {
	"bert-en" : "bert-base-uncased",
    "mbert" : "bert-base-multilingual-cased",
    "bert-tr" : "dbmdz/bert-base-turkish-cased",
    "bert-it" : "dbmdz/bert-base-italian-cased",
    "bert-de" : "bert-base-german-cased",
    "bert-sp" : "dccuchile/bert-base-spanish-wwm-cased",
    "bert-fi" : "TurkuNLP/bert-base-finnish-cased-v1",
    "bert-nl" : "GroNLP/bert-base-dutch-cased"
}

MODEL_TO_SPECIAL_TOKENS = {
    "gpt2" : "Ġ",
    "sberbank-ai/mGPT" : "Ġ"}

MASK_TYPE_DICT = {
    "all":["none", "mask", "truncate"],
    "none":["none"],
    "mask":["mask"],
    "truncate":["truncate"]
}

# Helper functions for getting pairwise stats

def get_output(model, tokenizer, sent):
    
    inputs = tokenizer(" ".join(sent), return_tensors="pt")
    
    with torch.no_grad(): logits = model(**inputs).logits
    probs = F.softmax(logits[0], dim=-1)
        
    return(probs, inputs)

# Converts the target word to its ID and returns whether it's multiple tokens
def get_target_id(target, tokenizer):
    
    target_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target))

    if(len(target_id) > 1):
        return target_id[0], True
    else:
        return target_id, False


# Helper functions for getting non-pairwise stats

def get_ar_predictions(sentence, ar_model):
    model = GPT2LMHeadModel.from_pretrained(ar_model)
    tokenizer = GPT2TokenizerFast.from_pretrained(ar_model)

    sent_tokens = tokenizer.tokenize(sentence, add_special_tokens=True)

    indexed_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)

    tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)

    with torch.no_grad():
        probs = model(tokens_tensor)[0].softmax(dim=2).squeeze()

    return list(zip(sent_tokens, indexed_tokens, (None,) + probs.unbind()))


def get_ar_surprisals(predictions):
    result = []
    for j, (word, word_idx, preds) in enumerate(predictions):
        if preds is None:
            surprisal = 0.0
        else:
            surprisal = -1 * np.log2(preds[word_idx].item())
        result.append( (j+1, word, surprisal) )
    return result

def retokenize(tokens, reference, ar_model):

    token_symbol = MODEL_TO_SPECIAL_TOKENS[ar_model]
    sent = [x[1] for x in tokens]
    stats = [x[2] for x in tokens]
    retokenized_sent, retokenized_stat = [], []

    i, j, = 0, 0
    while True:
        if i >= len(sent) - 1:
            break
        while True:
            j += 1
            if j >= len(sent):
                break
            if sent[j].startswith(token_symbol):
                break
        retokenized_sent.append("".join(sent[i:j]))
        retokenized_stat.append(sum(stats[i:j]))
        i = j

    if len(reference) == len(retokenized_sent):
        return retokenized_stat
    else:
        print("Tokenization error:")
        print(retokenized_sent)
        print(reference)
        print(retokenized_stat)
        print("\n")
        return [0.0 for _ in range(len(reference))]

def get_autoregressive_stats(sent, ar_model, language):
    tokenized_sent = sent.split(" ")
    predictions = get_ar_predictions(sent + ".", ar_model)

    frequencies = [zipf_frequency(w.strip().strip(punctuation), language, wordlist='best', minimum=0.0) for w in tokenized_sent]
    surprisals = retokenize(get_ar_surprisals(predictions), tokenized_sent, ar_model)

    return surprisals, frequencies

def get_mask(split_sent, mt, j, i, mask_idxs):
    # mt = mask type

    if mask_idxs == "mask_first":

        if mt == "none":
            return split_sent[0:j] + ["[MASK]"] + split_sent[j+1:i] + [split_sent[i]] + split_sent[i+1:len(split_sent)]
        if mt == "mask":
            return split_sent[0:j] + ["[MASK]"] + split_sent[j+1:i] + [split_sent[i]] + ["[MASK]"] * (len(split_sent) - i - 1)
        if mt == "truncate":
            return split_sent[0:j] + ["[MASK]"] + split_sent[j+1:i] + [split_sent[i]]

    if mask_idxs == "mask_both":

        if mt == "none":
            return split_sent[0:j] + ["[MASK]"] + split_sent[j+1:i] + ["[MASK]"] + split_sent[i+1:len(split_sent)]
        if mt == "mask":
            return split_sent[0:j] + ["[MASK]"] + split_sent[j+1:i] + ["[MASK]"] + ["[MASK]"] * (len(split_sent) - i - 1)
        if mt == "truncate":
            return split_sent[0:j] + ["[MASK]"] + split_sent[j+1:i] + ["[MASK]"]

    if mask_idxs == "mask_second":

        if mt == "none":
            return split_sent[0:j] + [split_sent[j]] + split_sent[j+1:i] + ["[MASK]"] + split_sent[i+1:len(split_sent)]
        if mt == "mask":
            return split_sent[0:j] + [split_sent[j]] + split_sent[j+1:i] + ["[MASK]"] + ["[MASK]"] * (len(split_sent) - i - 1)
        if mt == "truncate":
            return split_sent[0:j] + [split_sent[j]] + split_sent[j+1:i] + ["[MASK]"]


def get_pairwise_stats(split_sent, text_id, sent_id, mlm_model, ar_surps, freqs, mask_type):
    
    #tokenizer = BertTokenizer.from_pretrained(MODEL_NAME[mlm_model], cache_dir = "/cluster/scratch/ewilcox/transformer_models")
    #model = BertForMaskedLM.from_pretrained(MODEL_NAME[mlm_model], cache_dir = "/cluster/scratch/ewilcox/transformer_models")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME[mlm_model])
    model = BertForMaskedLM.from_pretrained(MODEL_NAME[mlm_model])
    
    model.eval()

    mask_types = MASK_TYPE_DICT[mask_type]

    pairwise_stats = []

    for mt in mask_types:
    # "i" is the index of the trigger and "j" is the index of the target
    # "j" = instances when just j is masked. "ji" = instances when j and i are masked
        for i in range(len(split_sent)):

            for j in range(i):
                
                w_i = split_sent[i]
                w_j = split_sent[j]
              	# Get the ID of the target word, and determine if this is multiple tokens
                target_id, is_multitok = get_target_id(w_j, tokenizer)

                # === Get model outputs ===
                j_mask = get_mask(split_sent, mt, j, i, "mask_first")
                #j_mask = split_sent[0:j] + ["[MASK]"] + split_sent[j+1:len(split_sent)]
                j_probs, j_inputs = get_output(model, tokenizer, j_mask)
                j_mask_token_index = (j_inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][0]

                ji_mask = get_mask(split_sent, mt, j, i, "mask_both")
                #ji_mask = split_sent[0:j] + ["[MASK]"] + split_sent[j+1:i] + ["[MASK]"] + split_sent[i+1:len(split_sent)]
                ji_probs, ji_inputs = get_output(model, tokenizer, ji_mask)
                ji_mask_token_index = (ji_inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][0]

                i_mask = get_mask(split_sent, mt, j, i, "mask_second")
                #i_mask = split_sent[0:i] + ["[MASK]"] + split_sent[i+1:len(split_sent)]
                i_probs, i_inputs = get_output(model, tokenizer, i_mask)
                i_mask_token_index = (i_inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][0]
                i_probs = np.asarray(i_probs[i_mask_token_index], dtype=float)

                # === Top 5 Predictions ===
                k = 5
                top_j_probs, top_j_idxs = torch.topk(j_probs[j_mask_token_index], k, sorted=True)
                top_j_preds = [tokenizer.convert_ids_to_tokens(x) for x in top_j_idxs.tolist()]

                top_ji_probs, top_ji_idxs = torch.topk(ji_probs[j_mask_token_index], k, sorted=True)
                top_ji_preds = [tokenizer.convert_ids_to_tokens(x) for x in top_ji_idxs.tolist()]

                # === Get surps === 
                j_surps = np.asarray(- np.log2(j_probs[j_mask_token_index]), dtype=float)
                ji_surps = np.asarray(-np.log2(ji_probs[ji_mask_token_index]), dtype=float)

                j_probs = np.asarray(j_probs[j_mask_token_index], dtype=float)
                ji_probs = np.asarray(ji_probs[ji_mask_token_index], dtype=float)

                # j_surp = only j is masked = p(w_t | w_s, w_not-t-s)
                # ji_surp = both j and i are masked = p(w_t | w_not-t-s )
                # the PMI is p(w_t | w_not) - p(w_t | w_s, w_not)
                # the PMI is ji_surp - j_surp

                # === PMI ===
                ji_surp = ji_surps[target_id]
                j_surp = j_surps[target_id]
                pmi = ji_surp - j_surp

                # === Expected PMI ===
                # NB We use the probs for j when i is known, which is the "trigger aware" expectation, in the paper
                e_pmi = np.sum(j_probs * (ji_surps - j_surps))
                e_ppmi = np.sum(j_probs * np.maximum((ji_surps - j_surps), 0))
                e_npmi = np.sum(j_probs * - np.minimum((ji_surps - j_surps), 0))

                e_pmi_i = np.sum(i_probs * (ji_surps - j_surps))
                e_ppmi_i = np.sum(i_probs * np.maximum((ji_surps - j_surps), 0))
                e_npmi_i = np.sum(i_probs * - np.minimum((ji_surps - j_surps), 0))

                # === Expected PMI, trigger blind ===
                #e_pmi_tb = np.sum(ji_probs * (ji_surps - j_surps))

                result = (text_id, sent_id, i, w_i, len(w_i), freqs[i], ar_surps[i], j, w_j, len(w_j), freqs[j], ar_surps[j], ji_surp.item(), j_surp.item(), pmi.item(), e_pmi.item(), e_ppmi.item(), e_npmi.item(), e_pmi_i.item(), e_ppmi_i.item(), e_npmi_i.item(), is_multitok, top_j_preds, top_ji_preds, mask_type)
                pairwise_stats.append(result)
    
    
    return pairwise_stats


def get_stats(dataset, input_path, mlm_model, ar_model, language, mask_type):
    
    #sents = pd.read_csv(input_path)
    sents = pd.read_csv(input_path)[315:]
    #sents = pd.read_csv(DATA_PATH)[50:101]
    #sents = pd.read_csv(DATA_PATH)[100:]

    for index, sent in sents.iterrows():

    	split_sent = sent["text"].split(" ")
    	text_id = sent["Text_ID"]
    	sent_id = sent["Sentence_Number"]

    	ar_surps, freqs = get_autoregressive_stats(sent["text"], ar_model, language)
    	pairwise_stats = get_pairwise_stats(split_sent, text_id, sent_id, mlm_model, ar_surps, freqs, mask_type)

    	df_export = pd.DataFrame(pairwise_stats, columns = ["text_id", "sent_id", "trigger_idx", "trigger", "trigger_len", "trigger_freq", "trigger_surp",  "target_idx", "target", "target_len", "target_freq", "target_surp", "ji_surp", "j_surp", "pmi", "e_pmi", "e_ppmi", "e_npmi", "e_pmi_i", "e_ppmi_i", "e_npmi_i", "is_multitok", "top_j_preds", "top_ji_preds", "mask_type"])
    	df_export.to_csv("./results/" + dataset + "_" + language + "_results_"+str(index)+".csv")


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--input_path", type=str, required=True)
	parser.add_argument("--dataset", type=str, required=True)
	parser.add_argument("--mlm_model", type=str, required=True)
	parser.add_argument("--ar_model", type=str, required=True)
	parser.add_argument("--language", type=str, required=True)
	parser.add_argument("--mask_type", type=str, required=True)
	args = parser.parse_args()

	input_path = args.input_path
	dataset = args.dataset
	if dataset not in ["provo", "dundee", "ucl", "test", "meco"]:
		raise ValueError('Invalid dataset name: %s' % args.dataset)

	get_stats(args.dataset, args.input_path, args.mlm_model, args.ar_model, args.language, args.mask_type)

    # Call to get multilingual results from the MECO corpus for just the "mask" mask-type
    #python get_predictors.py --input_path ./test_sents.csv --dataset test --mlm_model mbert --ar_model sberbank-ai/mGPT --language ru --mask_type mask

    # Call to get all mask-type results from a test corpus
    #python get_predictors.py --input_path ./test_sents.csv --dataset test --mlm_model bert --ar_model gpt2 --language en --mask_type all


if __name__ == '__main__':
	main()

