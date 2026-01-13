# export CUDA_VISIBLE_DEVICES=0; python3 encode_test_and_labels_mm.py -model-dir <model_dir> -new-dataset LF-AmazonTitles-131K
import os
import argparse
import json
import numpy as np
import gc
import torch
import scipy

from sklearn.decomposition import PCA
from xclib.data import data_utils
from utils import prepare_network, get_lbl_embeddings, get_doc_embeddings, _predict_ova, build_embedding_matrix

granularity = 100

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

def load_model(args, snet):
    saved_state_dict = torch.load(f"{args.model_dir}/state_dict.pt")
    snet.load_state_dict({f"{k}":v for k,v in saved_state_dict.items()})

def initialize_clusters(args, snet):
    label_mapping = np.load(os.path.join(args.model_dir, 'cluster_mapping.npy'))
    snet.set_label_mapping(torch.LongTensor(label_mapping))

def initialize_cnt_repr(args, snet):
    saved_state_dict = torch.load(f"{args.model_dir}/state_dict.pt")
    cnt_representations = saved_state_dict['cnt_representations']
    snet.state_dict()['cnt_representations'].copy_(cnt_representations)


def compute_similar_labels(new_raw_lbl_emb, orig_raw_lbl_emb, topk, thr=0.98, allow_self=False):
    print("Computing the TOP K most similar labels")
    if allow_self:        
        retrieved_lbl = _predict_ova(new_raw_lbl_emb, orig_raw_lbl_emb, topk)
        sim_val = retrieved_lbl.data.reshape(-1, topk)
        sim_mask = (sim_val >= thr).astype(float)
        sim_pos = 1 + (granularity * (1 - sim_val)).astype(int)
        most_sim_lbl = retrieved_lbl.indices.reshape(-1, topk)
    else:
        retrieved_lbl = _predict_ova(new_raw_lbl_emb, orig_raw_lbl_emb, k=topk + 1)
        sim_val = retrieved_lbl.data.reshape(-1, topk + 1)[:, 1::]
        sim_mask = (sim_val >= thr).astype(float)
        sim_pos = 1 + (granularity * (1 - sim_val)).astype(int)
        most_sim_lbl = retrieved_lbl.indices.reshape(-1, topk + 1)[:, 1::]
    return most_sim_lbl, sim_mask, sim_pos

def encode_and_save(args, snet, n, prefix):
    if prefix == "lbl":
        trn_Y_mapping = torch.arange(n)
        embeddings, _ = get_lbl_embeddings(
            args.tokenization_folder, prefix,
            n,
            snet,
            args.max_length,
            bsz=args.bs,
            trn_Y_mapping=trn_Y_mapping,
            approx=False,
            pca_components=None,
            inference=True)        
    else:
        if prefix == "tst":
            is_inference = True
        else:
            is_inference = False
        embeddings = get_doc_embeddings(
            args.tokenization_folder, f"{prefix}_doc",
            n,
            snet,
            args.max_length,
            bsz=args.bs,
            pca_components=None,
            inference=is_inference)
    np.save(os.path.join(args.result_dir, "embeddings", f"{prefix}.ngame.npy"), embeddings)
    del embeddings
    gc.collect()

def main(args):
    args.device = torch.device(args.device)
    args.result_dir = os.path.join(
        args.work_dir, 'results', "X-M1", args.new_dataset, args.version)
    args.data_dir = os.path.join(
        args.work_dir, 'data', args.new_dataset)
    args.tokenization_folder = os.path.join(
        args.data_dir, args.tokenizer if args.tokenizer else f"{args.tokenizer_type}-{args.max_length}")
    args.original_data_dir = os.path.join(
        args.work_dir, 'data', args.dataset)
    args.original_tokenization_folder = os.path.join(
        args.original_data_dir, f"{args.tokenizer_type}-{args.max_length}")
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'embeddings'), exist_ok=True)
    
    args.img_emb_dim = None
    if args.multimodal_xmc:
        num_imgs = args.multimodal_num_imgs
        # Build embedding info for train split
        if args.trn_lbl_fname.endswith(".npz"):
            trn_X_Y = scipy.sparse.load_npz(os.path.join(args.data_dir, args.trn_lbl_fname))
        else:
            trn_X_Y = data_utils.read_gen_sparse(os.path.join(args.data_dir, args.trn_lbl_fname))
        all_trn_embs = np.load(os.path.join(args.data_dir, args.multimodal_trn_img_embs))
        all_trn_embs_map = np.load(os.path.join(args.data_dir, args.multimodal_trn_img_embs_map), allow_pickle=True)
        trn_embs, trn_embs_mask = build_embedding_matrix(trn_X_Y.shape[0], num_imgs, all_trn_embs, all_trn_embs_map)
        
        # Build embedding info for lbl split
        all_lbl_embs = np.load(os.path.join(args.data_dir, args.multimodal_lbl_img_embs))
        all_lbl_embs_map = np.load(os.path.join(args.data_dir, args.multimodal_lbl_img_embs_map), allow_pickle=True)
        lbl_embs, lbl_embs_mask = build_embedding_matrix(args.n_labels, num_imgs, all_lbl_embs, all_lbl_embs_map)
      
        # Build embedding info for val split
        all_val_embs = np.load(os.path.join(args.data_dir, args.multimodal_val_img_embs))
        all_val_embs_map = np.load(os.path.join(args.data_dir, args.multimodal_val_img_embs_map), allow_pickle=True)
        
        # Load validation labels to know number of validation datapoints
        if args.val_lbl_fname.endswith(".npz"):
            val_X_Y = scipy.sparse.load_npz(os.path.join(args.data_dir, args.val_lbl_fname))
        else:
            val_X_Y = data_utils.read_gen_sparse(os.path.join(args.data_dir, args.val_lbl_fname))
        val_embs, val_embs_mask = build_embedding_matrix(val_X_Y.shape[0], num_imgs, all_val_embs, all_val_embs_map)
        # Store in args the embedding dimension
        args.img_emb_dim = trn_embs.shape[2]
   
    

    # Create network architecture (backbone, free vectors, label prototype network, centroids)
    snet = prepare_network(args, trn_embs, trn_embs_mask, lbl_embs, lbl_embs_mask, val_embs, val_embs_mask)
    
    # Load the saved model
    load_model(args, snet)
    # Initialize clusters and free vectors
    if args.prime:
        initialize_cnt_repr(args, snet)
        initialize_clusters(args, snet)
    
    # Encode text and save embeddings
    for file_name in os.listdir(args.data_dir):
        if os.path.isfile(os.path.join(args.data_dir, file_name)):
            if file_name[-8::] == ".raw.txt" and (len(args.only) == 0 or file_name[0:-8] in args.only.split(",")):
                with open(os.path.join(args.data_dir, file_name), "r", encoding="utf-8", errors='ignore') as f:
                    size = list()
                    for bl in blocks(f):
                        size.append(bl.count("\n"))
                    size = sum(size[0:-1]) + bl.strip().count("\n") + 1 + (1 if bl[0]=="\n" else 0)
                print(f"Reading {file_name} file with {size} inputs")
                encode_and_save(args, snet, size, file_name[0:-8])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model-dir", type=str, help="Model dir")
    parser.add_argument("-new-dataset", type=str, help="Dataset name", default='LF-AmazonTitles-131K')
    parser.add_argument("-topk", type=int, help="Number of related labels used to compute cluster and centroid for new labels", default=1)
    parser.add_argument("-only", type=str, help="Only encode the list of splits, e.g. 'lbl,tst'", default="")
    parser.add_argument("-bs", type=int, help="Batch size", default=10000)
    parser.add_argument("-tokenizer", type=str, help="The name of the new tokenizer", default="")
    parser.add_argument("--top1", action=argparse.BooleanOptionalAction)
    parser.add_argument("--return_tokens", action=argparse.BooleanOptionalAction)
    parser.add_argument("--same_lbl", action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    with open(os.path.join(args.model_dir, "executed_script_args.txt"), 'r') as f:
        args.__dict__.update(json.load(f))
    
    print(args)
    main(args)
