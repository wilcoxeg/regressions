
import os
import re
import torch
import pandas as pd
import numpy as np
from string import punctuation
import sentencepiece
from torch.nn import functional as F

from transformers import BertTokenizer, BertForMaskedLM

from transformers import logging
logging.set_verbosity_error()

DATA_PATH = "./provo_sents.csv"
#DATA_PATH = "./sample.csv"


MODEL_NAME = {
	"bert" : "bert-base-uncased"
}

MODEL = "bert"
TOP_K = 5


def get_output(model, tokenizer, sent):
    
    inputs = tokenizer(" ".join(sent), return_tensors="pt")
    
    with torch.no_grad(): logits = model(**inputs).logits
    probs = F.softmax(logits[0], dim=-1)
        
    return(probs, inputs)


def get_target_id(model, target, tokenizer):
    
    target_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target))

    if(len(target_id) > 1):
        return target_id[0], True
    else:
        return target_id, False


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log2(a / b), 0))



def get_top_k_delta(k, j_probs, ji_probs, tokenizer):
    
    top_k_j_probs, top_k_j_idxs = torch.topk(j_probs, k, sorted=True)
    j_sum_prob = sum(top_k_j_probs)
    #j_sum_surp = sum([-np.log2(x) for x in top_k_j_probs])
    top_k_j_preds = [tokenizer.convert_ids_to_tokens(x) for x in top_k_j_idxs.tolist()]
    
    _, top_k_ji_idxs = torch.topk(ji_probs, k, sorted=True) #These are actually the top indexes
    top_k_ji_probs = [ji_probs[x] for x in top_k_j_idxs] #These are the probs of the index from above
    ji_sum_prob = sum(top_k_ji_probs)
    #ji_sum_surp = sum([-np.log2(x) for x in top_k_ji_probs])
    top_k_ji_preds = [tokenizer.convert_ids_to_tokens(x) for x in top_k_ji_idxs.tolist()]
    
    return j_sum_prob - ji_sum_prob, top_k_j_preds, top_k_ji_preds

def get_pairwise_stats(sent, model, top_k, text_id, sent_id):
    
    # As is the case w/ Provo, we're going to assume that input is tokenized
    split_sent = sent.split(" ")
    print(len(split_sent))

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME[model])
    model = BertForMaskedLM.from_pretrained(MODEL_NAME[model])
    model.eval()

    pairwise_stats = []
    for i in range(len(split_sent)):
        for j in range(i):
            
            w_i = split_sent[i]
            w_j = split_sent[j]

            j_mask = split_sent[0:j] + ["[MASK]"] + split_sent[j+1:len(split_sent)]
            j_probs, j_inputs = get_output(model, tokenizer, j_mask)
            target_id, is_multitok = get_target_id(model, w_j, tokenizer)
            # Get the index of the first mask, which may not be j because of tokenization
            j_mask_token_index = (j_inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][0]
            w_j_surp = - np.log2(j_probs[j_mask_token_index][target_id])

            ji_mask = split_sent[0:j] + ["[MASK]"] + split_sent[j+1:i] + ["[MASK]"] + split_sent[i+1:len(split_sent)]
            ji_probs, ji_inputs = get_output(model, tokenizer, ji_mask)
            ji_mask_token_index = (ji_inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][0]
            w_ji_surp = - np.log2(ji_probs[ji_mask_token_index][target_id])
            
            delta_prob, j_preds, ji_preds = get_top_k_delta(top_k, j_probs[j_mask_token_index], ji_probs[ji_mask_token_index], tokenizer)
            
            pmi = w_ji_surp - w_j_surp
            kld = KL(j_probs[j_mask_token_index], ji_probs[ji_mask_token_index])
            
            result = (text_id, sent_id, w_i, w_j, i, j, w_j_surp.item(), pmi.item(), delta_prob.item(), kld.item(), j_preds, ji_preds, is_multitok)

            pairwise_stats.append(result)
    
    
    return pairwise_stats


def get_stats():
    
    sents = pd.read_csv(DATA_PATH)[]
    #sents = pd.read_csv(DATA_PATH)[0:51]
    #sents = pd.read_csv(DATA_PATH)[50:101]
    #sents = pd.read_csv(DATA_PATH)[100:]


    for index, sent in sents.iterrows():

        sent_stats = get_pairwise_stats(sent["text"], MODEL, TOP_K, sent["Text_ID"], sent["Sentence_Number"])

        df_export = pd.DataFrame(sent_stats, columns = ["text_id", "sent_id", "trigger", "target", "trigger_idx", "target_idx", "target_surp", "pmi", "topk_prob", "kld", "j_preds", "ji_preds", "is_multitok"])
        df_export.to_csv("./results/provo_results_"+str(index)+".csv")


def main():
    get_stats()


if __name__ == '__main__':
	main()

