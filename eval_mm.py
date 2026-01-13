import os
from tabulate import tabulate
import argparse
from xclib.data import data_utils
from xclib.evaluation import xc_metrics
import numpy as np
from xclib.utils.sparse import normalize
from xclib.utils.matrix import SMatrix
from xclib.utils.shortlist import Shortlist
from tqdm import tqdm
from xclib.utils.sparse import csr_from_arrays
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
import torch
import scipy
import polars as pl
from collections import Counter

# python3 eval_mm.py -work-dir "<root_path>" --version "<experiment_name>" -dataset "LF-AmazonTitles-131K" --no-retrieval_augmented_inference --save-pred

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

def estimate_len(file_name):
    with open(os.path.join(args.data_dir, file_name), "r", encoding="utf-8", errors='ignore') as f:
        size = list()
        for bl in blocks(f):
            size.append(bl.count("\n"))
        size = sum(size[0:-1]) + bl.strip().count("\n") + 1 + (1 if bl[0]=="\n" else 0)
    return size

def keep_top_k_values_per_row(matrix, k):
    """Keep only the top k values per row in a sparse matrix."""
    new_data, new_indices, new_indptr = [], [], [0]
    for i in range(matrix.shape[0]):
        row_start, row_end = matrix.indptr[i], matrix.indptr[i + 1]
        row_data = matrix.data[row_start:row_end]
        row_indices = matrix.indices[row_start:row_end]
        if len(row_data) > k:
            top_k_indices = np.argsort(row_data)[-k:]
            new_data.extend(row_data[top_k_indices])
            new_indices.extend(row_indices[top_k_indices])
        else:
            new_data.extend(row_data)
            new_indices.extend(row_indices)
        new_indptr.append(len(new_data))
    return csr_matrix((new_data, new_indices, new_indptr), shape=matrix.shape)

def create_lbl_mapping(lst):
    mapping, seen, k = list(), dict(), 0
    for item in lst:
        if item not in seen:
            seen[item] = k
            k += 1
        mapping.append(seen[item])
    return np.array(mapping), np.array([k for k, v in sorted(seen.items(), key=lambda item: item[1])])

def reduce_sparse_matrix_using_mapping(matrix, mapping):
    max_values = {}
    for a, b, c in zip(matrix.nonzero()[0], mapping[matrix.nonzero()[1]], matrix.data):
        if (a, b) in max_values:
            max_values[(a, b)] = max(max_values[(a, b)], c)
        else:
            max_values[(a, b)] = c
    tup_val = [(a, b, c) for (a, b), c in max_values.items()]
    return csr_matrix((list(zip(*tup_val))[2], (list(zip(*tup_val))[0], list(zip(*tup_val))[1])), shape=(matrix.shape[0], len(set(mapping))))

def reduce_matrix_using_mapping(matrix, indices):
    reduced_matrix = torch.zeros((matrix.shape[0], len(set(indices))), dtype=matrix.dtype).to(matrix.device)
    for old_col, new_col in enumerate(indices):
        torch.maximum(reduced_matrix[:, new_col], matrix[:, old_col], out=reduced_matrix[:, new_col])
    return reduced_matrix

def _predict_ova(X, clf, k=20, query_batch_size=512, label_batch_size=100000, device="cuda", return_sparse=True, lbl_filter_id_matrix=None, tst_filter_id_matrix=None):
    """Predictions in brute-force manner"""
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    torch.set_grad_enabled(False)
    num_instances, num_labels = len(X), len(clf)
    query_batches = np.array_split(range(num_instances), num_instances//query_batch_size if num_instances > query_batch_size else 1)
    label_batches = np.array_split(range(num_labels), num_labels//label_batch_size if num_labels > label_batch_size else 1)
    output = SMatrix(
        n_rows=num_instances,
        n_cols=num_labels,
        nnz=k)
    X = torch.from_numpy(X)
    clf = torch.from_numpy(clf)
    for i in tqdm(range(len(query_batches))):
        q_ind = query_batches[i]
        s_q_ind, e_q_ind = q_ind[0], q_ind[-1] + 1
        _X = X[s_q_ind: e_q_ind].to(device)
        
        vals, ind = list(), list()
        for lbl_ind in label_batches:
            s_lbl_ind, e_lbl_ind = lbl_ind[0], lbl_ind[-1] + 1
            _clf = clf[s_lbl_ind: e_lbl_ind].to(device).T
            ans = _X @ _clf
            partial_vals, partial_ind = torch.topk(
                    ans, k=k, dim=-1, sorted=True)
            vals.append(partial_vals.cpu().numpy())
            ind.append(lbl_ind[partial_ind.cpu().numpy()])
            del _clf
        vals = np.hstack(vals)
        ind = np.hstack(ind)
        sorted_vals_ind = np.argsort(-vals)[:, 0:k]
        output.update_block(
            s_q_ind, np.take_along_axis(ind, sorted_vals_ind, axis=1), np.take_along_axis(vals, sorted_vals_ind, axis=1))
        del _X
    if return_sparse:
        return output.data()
    else:
        return output.data('dense')[0]

def mean_average_precision(eval_flags):
    rank = (1 + np.tile(np.arange(eval_flags.shape[1]), (eval_flags.shape[0], 1)))
    p = eval_flags.cumsum(axis=1) / rank
    den = eval_flags.cumsum(axis=1)
    den[den==0] = 1
    maps = (eval_flags*p).cumsum(axis=1) / den
    return np.mean(maps, axis=0)

def mean_reciprocal_rank(eval_flags):
    rank = (1 + np.tile(np.arange(eval_flags.shape[1]), (eval_flags.shape[0], 1)))
    first_nonzero_indices = np.argmax(eval_flags != 0, axis=1)
    num = np.zeros_like(eval_flags, dtype=int)
    num[np.arange(eval_flags.shape[0]), first_nonzero_indices] = 1
    mrrs = (num / rank).cumsum(axis=1)
    return np.mean(mrrs, axis=0)

def get_filter_map(fname):
    if fname is not None:
        return np.loadtxt(fname).astype('int')
    else:
        return None


def filter_predictions(pred, mapping):
    if mapping is not None and len(mapping) > 0:
        print("Filtering labels.")
        mapping_ = mapping[mapping[:, 0] < pred.shape[0]]
        pred[mapping_[:, 0], mapping_[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def evaluate(_true, _pred, _train=None, ks=[1, 5, 10, 15], A=0.5, B=0.4):
    """Evaluate function
    * k: used for all the metrics
    """
    # Prepare hits
    _true.indices = _true.indices.astype('int64')
    if _train is not None:
        inv_propen = xc_metrics.compute_inv_propesity(_train, A, B)
    else:
        inv_propen = None
    table = [["Metrics@K"] + [f"TOP {k}" for k in ks]]
    indices, true_labels, ps_indices, inv_psp = xc_metrics._setup_metric(_pred, _true, inv_propen, k=max(ks), sorted=True, use_cython=True)
    eval_flags = xc_metrics._eval_flags(indices, true_labels, None)
    # Compute metrics
    _total_pos = np.asarray(true_labels.sum(axis=1), dtype=np.int32)
    n = np.cumsum(1/np.log2(np.arange(1, true_labels.shape[1]+1)+1))[_total_pos - 1]
    deno = true_labels.sum(axis=1)
    deno[deno == 0] = 1
    deno = 1/deno
    precision = 100*xc_metrics._precision(eval_flags, k=max(ks))
    recall = 100*xc_metrics._recall(eval_flags, deno, k=max(ks))
    hits = 100*xc_metrics._hits(eval_flags)
    ndcg = 100*xc_metrics._ndcg(eval_flags, n, k=max(ks))
    map = 100*mean_average_precision(eval_flags)
    mrr = 100*mean_reciprocal_rank(eval_flags)
    table.append(["P@K"] + [precision[k-1] for k in ks])
    table.append(["R@K"] + [recall[k-1] for k in ks])
    table.append(["Hit@K"] + [hits[k-1] for k in ks])
    table.append(["NDCG@K"] + [ndcg[k-1] for k in ks])
    table.append(["MAP@K"] + [map[k-1] for k in ks])
    table.append(["MRR@K"] + [mrr[k-1] for k in ks])
    if inv_propen is not None:
        eval_flags = np.multiply(inv_psp[indices], eval_flags)
        ps_eval_flags = xc_metrics._eval_flags(ps_indices, true_labels, inv_psp)
        psprecision = 100*xc_metrics._precision(eval_flags, max(ks))/xc_metrics._precision(ps_eval_flags, max(ks))
        psndcg = 100*xc_metrics._ndcg(eval_flags, n, max(ks))/xc_metrics._ndcg(ps_eval_flags, n, max(ks))
        table.append(["PSP@K"] + [psprecision[k-1] for k in ks])
        table.append(["PSNDCG@K"] + [psndcg[k-1] for k in ks])
    metrics = tabulate(table, headers='firstrow', tablefmt='fancy_grid', floatfmt=".2f")
    return metrics

def predict(args, features, clf, k=10, q_bs=512, lbl_bs=100000, \
            raw_trn_queries=None, trn_X_Y=None):#, mapping=None):
    """
    Predict on validation set
    * ova and anns would get 300 (change if required)"""
    predict_ = None
    
    if args.retrieval_augmented_inference:
        k=args.retrieval_augmented_k
    if args.retrieval_augmented_inference and args.retrieval_augmented_split:
        k_queries = args.retrieval_augmented_split_query_k
        k_labels = args.retrieval_augmented_k
        predict_labels_ = lambda x,y: _predict_ova(normalize(x, copy=True), normalize(y, copy=True), k=k_labels, query_batch_size=q_bs, label_batch_size=lbl_bs)#, mapping=mapping)
        predict_queries_ = lambda x,y: _predict_ova(normalize(x, copy=True), normalize(y, copy=True), k=k_queries, query_batch_size=q_bs, label_batch_size=lbl_bs)#, mapping=mapping)
    else:
        predict_ = lambda x,y: _predict_ova(normalize(x, copy=True), normalize(y, copy=True), k=k, query_batch_size=q_bs, label_batch_size=lbl_bs)
        
    if args.retrieval_augmented_inference:
        lambda_val = args.retrieval_augmented_lambda
        if args.retrieval_augmented_split:
            clf_labels = clf
            clf_queries = raw_trn_queries
            pred_labels = predict_labels_(features, clf_labels)
            pred_queries = predict_queries_(features, clf_queries)
            pred = scipy.sparse.hstack([pred_labels, pred_queries])
            # Concat trn_X_Y and a diagonal matrix label_gt
            pred_indices = pred.indices.reshape(pred.shape[0], k_labels + k_queries)
            pred_data = pred.data.reshape(pred.shape[0], k_labels + k_queries)
            pred_data = torch.nn.functional.softmax(torch.from_numpy(pred_data/args.retrieval_augmented_tau), dim=1).numpy()
            pred_normalized = csr_matrix((pred_data.reshape(-1), (np.arange(pred.shape[0]).repeat(k_labels + k_queries), pred_indices.reshape(-1))), shape=pred.shape)
            # Create sparse diagonal matrix as label gt
            label_gt = csr_matrix((np.ones(clf.shape[0]), (np.arange(clf.shape[0]), np.arange(clf.shape[0]))))
            v_matrix = scipy.sparse.vstack([label_gt*(1-lambda_val), trn_X_Y*(lambda_val)])
            pred = pred_normalized*v_matrix
        else:
            # Build a new classifier using labels and enriched queries
            clf_augmented = np.concatenate([clf, raw_trn_queries], axis=0)
            pred = predict_(features, clf_augmented)
            pred_indices = pred.indices.reshape(pred.shape[0], k)
            pred_data = pred.data.reshape(pred.shape[0], k)
            pred_data = torch.nn.functional.softmax(torch.from_numpy(pred_data/args.retrieval_augmented_tau), dim=1).numpy()
            pred_normalized = csr_matrix((pred_data.reshape(-1), (np.arange(pred.shape[0]).repeat(k), pred_indices.reshape(-1))), shape=pred.shape)
            # Create sparse diagonal matrix 
            label_gt = csr_matrix((np.ones(clf.shape[0]), (np.arange(clf.shape[0]), np.arange(clf.shape[0]))))
            # Concat trn_X_Y and a diagonal matrix label_gt
            v_matrix = scipy.sparse.vstack([label_gt*(1-lambda_val), trn_X_Y*(lambda_val)])
            pred = pred_normalized*v_matrix    
    else:
        pred = predict_(features, clf)
    
    return pred

def eval(pred, labels, trn_labels=None, filter_labels=None, A=0.5, B=0.4, ks=[1, 5, 10, 15]):
    """
    Evaluation
    * support for filter file (pass "" or empty file otherwise)"""
    labels.indices = labels.indices.astype('int64')
    if filter_labels and os.path.isfile(filter_labels):
        mapping = get_filter_map(filter_labels)
        pred = filter_predictions(pred, mapping)
    eval = f"\n\nAll datapoints ({labels.shape[0]} {labels.shape[1]})\n"
    eval += evaluate(labels, pred, trn_labels, ks, A, B)
            
    return eval

def save_pred(pred, file_name, pred_dir, k):
    # Get predicted indices and values
    row_indices = []
    row_values = []
    for i in range(pred.shape[0]):
        start, end = pred.indptr[i], pred.indptr[i + 1]
        aux_indices = np.argsort(-pred.data[start:end])
        row_indices.append(pred.indices[start:end][aux_indices])
        row_values.append(pred.data[start:end][aux_indices])
    # Convert list of lists with different sizes to a numpy matrix
    row_indices_padded = np.array([np.pad(row, (0, k - len(row)), constant_values=-1) for row in row_indices])
    row_values_padded = np.array([np.pad(row, (0, k - len(row)), constant_values=0) for row in row_values])
    np.save(os.path.join(pred_dir, f"{file_name}_indexes_{k}.npy"), row_indices_padded)
    np.save(os.path.join(pred_dir, f"{file_name}_scores_{k}.npy"), row_values_padded)

def main(args):
    args.result_dir = os.path.join(args.work_dir, 'results', "X-M1", args.dataset, args.version)
    args.emb_dir = os.path.join(args.result_dir, 'embeddings')
    args.pred_dir = os.path.join(args.result_dir, 'predictions')
    args.data_dir = os.path.join(args.work_dir, 'data', args.dataset)
    original_dataset = args.original_dataset if args.original_dataset else args.dataset
    if args.trn_lbl_fname.endswith(".npz"):
        trn_Y_fname = os.path.join(args.work_dir, 'data', original_dataset, "trn_X_Y.npz")
    else:
        trn_Y_fname = os.path.join(args.work_dir, 'data', original_dataset, "trn_X_Y.txt")
        
    args.k = [int(k) for k in args.k.split(",")]
    id_mapping = None
    # Load label and val embeddings
    lbl_emb = np.load(os.path.join(args.emb_dir, f"lbl.ngame.npy")).astype("float32") # label features embeddings
    
    if args.retrieval_augmented_inference:
        raw_trn_queries = np.load(os.path.join(args.emb_dir, f"trn.ngame.npy")).astype("float32")
        
        if args.tst_lbl_fname.endswith(".npz"):
            trn_X_Y = data_utils.read_gen_sparse(os.path.join(args.data_dir, trn_Y_fname))
        else:
            trn_X_Y = data_utils.read_gen_sparse(os.path.join(args.data_dir, args.trn_lbl_fname))
    else:
        raw_trn_queries = None
        trn_X_Y = None
    
    
    for file_name in os.listdir(args.emb_dir):
        if os.path.isfile(os.path.join(args.emb_dir, file_name)):
            if file_name[-10::] == ".ngame.npy" and file_name[0:-10] != "lbl" and (len(args.only) == 0 or file_name[0:-10] in args.only.split(",")):
                print(f"Running inference on {file_name}")

                # Define number of desired labels to be retrieved
                k = max(args.k)
                # Define the file name for filtering test
                filter_file_name = os.path.join(args.data_dir, "filter_labels_test.txt") # Needs to be changed for different test sets
                # Compute the maximum number of filtered labels per query to assure we retrieve items after filtering
                if os.path.isfile(filter_file_name):
                    arr = np.loadtxt(filter_file_name).astype('int')
                    if arr.sum() > 0:
                        k += Counter(arr[:, 0]).most_common(1)[0][1]

                if args.tst_lbl_fname.endswith(".npz"):
                    tst_Y_fname = os.path.join(args.data_dir, f"{file_name[0:-10]}_X_Y.npz")
                else:
                    tst_Y_fname = os.path.join(args.data_dir, f"{file_name[0:-10]}_X_Y.txt")
                
                tst_X = np.load(os.path.join(args.emb_dir, file_name)).astype("float32") # test document features
                # Predict
                pred = predict(
                    args,
                    features=tst_X,
                    clf=lbl_emb,
                    k=max(args.k),
                    q_bs=args.q_bs,
                    lbl_bs=args.lbl_bs,
                    raw_trn_queries=raw_trn_queries,
                    trn_X_Y=trn_X_Y)
                

                # Save predictions
                if args.save_pred:
                    print("Saving expanded predictions")
                    os.makedirs(args.result_dir, exist_ok=True)
                    os.makedirs(args.pred_dir, exist_ok=True)
                    save_pred(pred, f"{file_name[0:-10]}{'_expanded' if id_mapping is not None else ''}", args.pred_dir, k)

                #Reduce label space
                if id_mapping is not None:
                    pred = reduce_sparse_matrix_using_mapping(pred, id_mapping)
                    if pred2 is not None:
                        pred2 = reduce_sparse_matrix_using_mapping(pred2, id_mapping)
                    # Save predictions
                    if args.save_pred:
                        print("Saving reduced predictions")
                        save_pred(pred, f"{file_name[0:-10]}_reduced", args.pred_dir, k)

                #Evaluate
                if os.path.isfile(tst_Y_fname):
                    if tst_Y_fname.endswith(".npz"):
                        tst_Y = scipy.sparse.load_npz(tst_Y_fname)
                    else:
                        tst_Y = data_utils.read_sparse_file(tst_Y_fname)
    
                    if trn_Y_fname.endswith(".npz"):
                        trn_Y = scipy.sparse.load_npz(trn_Y_fname)
                    else:
                        trn_Y = data_utils.read_sparse_file(trn_Y_fname)

                        
                    acc = eval(
                        pred=pred,
                        labels=tst_Y,
                        trn_labels=trn_Y,
                        filter_labels=filter_file_name,
                        A=args.A,
                        B=args.B,
                        ks=args.k)
                    print(acc.replace(",", "\t"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-work-dir", type=str, help="Work dir")
    parser.add_argument("--version", type=str, help="Version of the run")
    parser.add_argument("-dataset", type=str, help="Dataset name")
    parser.add_argument("-q_bs", type=int, help="Batch size for queries", default=512)
    parser.add_argument("-lbl_bs", type=int, help="Batch size for labels", default=2000000)
    parser.add_argument("-A", type=float, help="The propensity factor A" , default=0.6)
    parser.add_argument("-B", type=float, help="The propensity factor B", default=2.6)
    parser.add_argument("-k", type=str, help="List of K values", default='1,5,10,15')
    parser.add_argument("-only", type=str, help="Only predict with the list of splits, e.g. 'tst,tst2'", default="tst")
    parser.add_argument("--save-pred", action=argparse.BooleanOptionalAction)
    parser.add_argument("--original-dataset", type=str, help="Name of the original dataset", default='')
    parser.add_argument("--trn-lbl-fname", type=str, required=False, help="Train label file name", default="trn_X_Y.txt")
    parser.add_argument("--tst-lbl-fname", type=str, required=False, help="Train label file name", default="tst_X_Y.txt")
    parser.add_argument("--retrieval_augmented_inference", action=argparse.BooleanOptionalAction)
    parser.add_argument("--retrieval_augmented_k", type=int, help="top-k retrieved in retrieval augmented inference", default=500)
    parser.add_argument("--retrieval_augmented_tau", type=float, help="temperature for softmax in retrieval augmented inference", default=0.05)
    parser.add_argument("--retrieval_augmented_lambda", type=float, help="weight to give to search on query space", default=0.1)
    parser.add_argument("--retrieval_augmented_split", action=argparse.BooleanOptionalAction)
    parser.add_argument("--retrieval_augmented_split_query_k", type=int, help="top-k retrieved in retrieval augmented inference", default=50)
    
    args = parser.parse_args()
    
    main(args)
