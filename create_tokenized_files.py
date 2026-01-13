import torch.multiprocessing as mp
from transformers import AutoTokenizer
import os
import numpy as np
import time
import functools
import argparse

# python -W ignore -u create_tokenized_files.py --data-dir <dataset_name> --tokenizer-type Qwen/Qwen2.5-3B-Instruct --max-length 32 --dump-folder-name qwen25_3B-32

def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer


def _tokenize(batch_input):
    tokenizer, max_len, batch_corpus = batch_input[0], batch_input[1], batch_input[2]
    temp = tokenizer.batch_encode_plus(
                    batch_corpus,                           # Sentence to encode.
                    add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                    max_length = max_len,                   # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,           # Construct attn. masks.
                    return_tensors = 'np',                  # Return numpy tensors.
                    truncation=True
            )

    return (temp['input_ids'], temp['attention_mask'])

def convert(corpus, tokenizer, max_len, num_threads, bsz=100000): 
    batches = [(tokenizer, max_len, corpus[batch_start: batch_start + bsz]) for batch_start in range(0, len(corpus), bsz)]

    pool = mp.Pool(num_threads)
    batch_tokenized = pool.map(_tokenize, batches)
    pool.close()

    input_ids = np.vstack([x[0] for x in batch_tokenized])
    attention_mask = np.vstack([x[1] for x in batch_tokenized])

    del batch_tokenized 

    return input_ids, attention_mask

@timeit
def tokenize_dump_memmap(corpus, tokenization_dir, tokenizer, max_len, prefix, num_threads, batch_size=10000000):
    ii = np.memmap(f"{tokenization_dir}/{prefix}_input_ids.dat", dtype='int64', mode='w+', shape=(len(corpus), max_len))
    am = np.memmap(f"{tokenization_dir}/{prefix}_attention_mask.dat", dtype='int64', mode='w+', shape=(len(corpus), max_len))
    for i in range(0, len(corpus), batch_size):
        _input_ids, _attention_mask = convert(corpus[i: i + batch_size], tokenizer, max_len, num_threads)
        ii[i: i + _input_ids.shape[0], :] = _input_ids
        am[i: i + _input_ids.shape[0], :] = _attention_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True, help="Data directory path - with {trn,tst}_X.txt, {trn,tst}_X_Y.txt and Y.txt")
    parser.add_argument("--max-length", type=int, help="Max length for tokenizer", default=32)
    parser.add_argument("--tokenizer-type", type=str, help="Tokenizer to use", default="bert-base-uncased")
    parser.add_argument("--num-threads", type=int, help="Number of threads to use", default=24)
    parser.add_argument("--only", type=str, help="Only encode the list of splits, e.g. 'lbl,tst'", default="")
    parser.add_argument("--dump-folder-name", type=str, help="Dump folder inside dataset folder", default="")


    args = parser.parse_args()

    DATA_DIR = args.data_dir
    
    max_len = args.max_length

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type, do_lower_case=True)
    if "gemma" in args.tokenizer_type:
        # For Gemma, we need to set the padding side to right
        tokenizer.padding_side = "right"
    
    elif "Llama-3.2" in args.tokenizer_type:
        tokenizer.pad_token = tokenizer.eos_token

    if(args.dump_folder_name == ""):
        dump_folder_name = f"{args.tokenization_type}-{max_len}"
    else:
        dump_folder_name = args.dump_folder_name
    tokenization_dir = f"{DATA_DIR}/{dump_folder_name}"
    os.makedirs(tokenization_dir, exist_ok=True)
    
    print(f"Dumping files in {tokenization_dir}...")
    for file_name in os.listdir(DATA_DIR):
        if os.path.isfile(os.path.join(DATA_DIR, file_name)):
            if file_name[-8::] == ".raw.txt" and (len(args.only) == 0 or file_name[0:-8] in args.only.split(",")):
                prefix = file_name[0:-8]
                if prefix != "lbl":
                    prefix += "_doc"
                raw = [x.strip() for x in open(os.path.join(DATA_DIR, file_name), "r", encoding="latin").readlines()]
                if raw:
                    print(f"Dumping for {file_name}...")
                    tokenize_dump_memmap(raw, tokenization_dir, tokenizer, max_len, prefix, args.num_threads)
