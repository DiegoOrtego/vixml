from typing import Any

import os
import sys
import json
import random
import time
import gc
import pickle
import functools
import inspect

import numpy as np
import csv
import scipy
from scipy.sparse import csr_matrix
from tqdm import tqdm
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import transformers
from sklearn.preprocessing import normalize

from xclib.utils.clustering import cluster_balance, b_kmeans_dense
from xclib.data import data_utils
from xclib.utils.dense import compute_centroid
import xclib.evaluation.xc_metrics as xc_metrics
from xclib.utils.matrix import SMatrix

from loss import RegLoss, ATripletMarginLossOHNMDM


def configure_paths(args: Any) -> None:
	"""Configure dataset/model/result paths and ensure directories exist.

	Mutates `args` in-place to set `model_dir`, `result_dir`, `data_dir`,
	`tokenization_folder`, and `device`.
	"""
	args.device = torch.device(args.device)
	if args.dataset_subfolder is not None:
		args.model_dir = os.path.join(
			args.work_dir, 'models', "X-M1", args.dataset_subfolder, args.dataset, args.version)
		args.result_dir = os.path.join(
			args.work_dir, 'results', "X-M1", args.dataset_subfolder, args.dataset, args.version)
		args.data_dir = os.path.join(
			args.work_dir, 'data', args.dataset_subfolder, args.dataset)
	else:
		args.model_dir = os.path.join(
			args.work_dir, 'models' , "X-M1", args.dataset, args.version)
		args.result_dir = os.path.join(
			args.work_dir, 'results', "X-M1", args.dataset, args.version)
		args.data_dir = os.path.join(
			args.work_dir, 'data', args.dataset)
	args.tokenization_folder = os.path.join(
		args.data_dir, f"{args.tokenizer_type}-{args.max_length}")
	os.makedirs(args.model_dir, exist_ok=True)
	os.makedirs(args.result_dir, exist_ok=True)
	os.makedirs(os.path.join(args.result_dir, 'embeddings'), exist_ok=True)


def set_seed(seed: int) -> None:
	"""Set Python, NumPy, and Torch RNG seeds (incl. CUDA)."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def parse_curriculum(args: Any) -> None:
	"""Parse curriculum and K list settings from `args` in-place."""
	if args.curr_steps == "":
		args.curr_steps = set()
	else:
		args.curr_steps = set(map(int, args.curr_steps.split(",")))
	args.ks = [int(k) for k in args.ks.split(",")]
	if args.k < max(args.ks):
		args.k = max(args.ks)


# ---- General utilities ----

def timeit(func):
	"""Print the runtime of the decorated function"""
	@functools.wraps(func)
	def wrapper_timer(*args, **kwargs):
		start_time = time.perf_counter()
		value = func(*args, **kwargs)
		end_time = time.perf_counter()
		run_time = end_time - start_time
		print(f"Finished {func.__name__} in {run_time:.4f} secs")
		return value
	return wrapper_timer


def mean_average_precision(eval_flags):
	rank = (1 + np.tile(np.arange(eval_flags.shape[1]), (eval_flags.shape[0], 1)))
	p = eval_flags.cumsum(axis=1) / rank
	den = eval_flags.cumsum(axis=1)
	den[den == 0] = 1
	maps = (eval_flags * p).cumsum(axis=1) / den
	return np.mean(maps, axis=0)


def mean_reciprocal_rank(eval_flags):
	rank = (1 + np.tile(np.arange(eval_flags.shape[1]), (eval_flags.shape[0], 1)))
	first_nonzero_indices = np.argmax(eval_flags != 0, axis=1)
	num = np.zeros_like(eval_flags, dtype=int)
	num[np.arange(eval_flags.shape[0]), first_nonzero_indices] = 1
	mrrs = (num / rank).cumsum(axis=1)
	return np.mean(mrrs, axis=0)


def get_filter_map(fname):
	"""Load filter file as numpy array"""
	if fname is not None and fname != "":
		return np.loadtxt(fname).astype('int')
	else:
		return None


def filter_predictions(pred, mapping):
	"""Filter predictions using given mapping"""
	if mapping is not None and len(mapping) > 0:
		print("Filtering labels.")
		pred[mapping[:, 0], mapping[:, 1]] = 0
		pred.eliminate_zeros()
	return pred


def build_embedding_matrix(num_datapoints, num_imgs, all_embs, all_embs_map):
    # Bound the number of images to num_imgs in all_trn_embs_map
    all_embs_map = np.array([x[:num_imgs] if x!=-1 else x for x in all_embs_map], dtype=object)
    # Initialize embeddings with zeros
    embs = np.zeros((num_datapoints, num_imgs, all_embs.shape[1]), dtype='float32')
    # Initialize the mask with zeros (no image embedding by default)
    mask = np.zeros((num_datapoints, num_imgs), dtype='int')
    # Create a mask for positions where all_trn_embs_map is not -1
    valid_indices = np.where(all_embs_map != -1)[0]

    # Flatten the valid indices and corresponding lists
    flat_indices = np.concatenate(all_embs_map[valid_indices])
    flat_positions = np.repeat(valid_indices, [len(lst) for lst in all_embs_map[valid_indices]])

    # Determine the slot (dimension 1) for each embedding
    slots = np.concatenate([np.arange(len(lst)) for lst in all_embs_map[valid_indices]])

    # Assign embeddings to the corresponding positions and slots
    embs[flat_positions, slots, :] = all_embs[flat_indices]
    # Set the mask to True for valid positions and slots
    mask[flat_positions, slots] = 1
    return embs, mask


def evaluate(_true, _pred, _train, k_r, ks, A, B):
	_true.indices = _true.indices.astype('int64')
	if _train is not None:
		inv_propen = xc_metrics.compute_inv_propesity(_train, A, B)
	else:
		inv_propen = None
	table = [["Metrics@K"] + [f"TOP {k}" for k in ks]]
	indices, true_labels, ps_indices, inv_psp = xc_metrics._setup_metric(_pred, _true, inv_propen, k=max(ks), sorted=True, use_cython=True)
	eval_flags = xc_metrics._eval_flags(indices, true_labels, None)
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
	metrics = tabulate(table, headers='firstrow', tablefmt='fancy_grid', floatfmt=".2f")
	rec = xc_metrics.recall(_pred, _true, k_r)[-1] * 100
	return f"{metrics}\nR@{k_r} {rec}", {f'P@{k}': precision[k-1] for k in ks}


def evaluate_with_filter(true_labels, predicted_labels, train_labels, filter_labels, k, ks, A, B):
	if filter_labels is not None:
		mapping = get_filter_map(filter_labels)
		predicted_labels = filter_predictions(predicted_labels, mapping)
	eval, metrics_dict = evaluate(true_labels, predicted_labels, train_labels, k, ks, A, B)
	return f"\nAll datapoints ({true_labels.shape[0]}):\n{eval}\n", metrics_dict


def _predict_ova(X, clf, k=20, batch_size=32, device="cuda", return_sparse=True, fp16=False):
	"""Predictions in brute-force manner"""
	torch.set_grad_enabled(False)
	num_instances, num_labels = len(X), len(clf)
	batches = np.array_split(range(num_instances), num_instances//batch_size)
	output = SMatrix(n_rows=num_instances, n_cols=num_labels, nnz=k)
	if fp16:
		X = torch.from_numpy(X).half()
		clf = torch.from_numpy(clf).half().to(device).T
	else:
		X = torch.from_numpy(X)
		clf = torch.from_numpy(clf).to(device).T
	for ind in tqdm(batches):
		s_ind, e_ind = ind[0], ind[-1] + 1
		_X = X[s_ind: e_ind].to(device)
		ans = _X @ clf
		vals, ind = torch.topk(ans, k=k, dim=-1, sorted=True)
		output.update_block(s_ind, ind.cpu().numpy(), vals.cpu().numpy())
		del _X
	if return_sparse:
		return output.data()
	else:
		return output.data('dense')[0]


def predict_and_eval(features, clf, labels, trn_labels, filter_labels, A, B, k=10, ks=[1,2,3,4,5], mode='ova'):
	pred = _predict_ova(normalize(features, copy=True), normalize(clf, copy=True), k=k)
	labels.indices = labels.indices.astype('int64')
	res, metrics_dict = evaluate_with_filter(labels, pred, trn_labels, filter_labels, k, ks, A, B)
	print(res)
	return res, pred, metrics_dict


from contextlib import contextmanager

@contextmanager
def evaluating(net):
	istrain = net.training
	try:
		net.eval()
		yield net
	finally:
		if istrain:
			net.train()


def _collate_fn(batch, num_labels):
	lens = []
	batch_labels = []
	random_pos_indices = []
	for item in batch:
		lens.append(len(item[2]))
		batch_labels.append(item[2])
		random_pos_indices.append(item[3])
	batch_size = len(batch_labels)
	rows = np.repeat(range(batch_size), lens)
	cols = np.concatenate(batch_labels, axis=None)
	data = np.ones((len(rows), ), dtype='bool')
	A = csr_matrix((data, (rows, cols)), shape=(batch_size, num_labels))
	cols = np.concatenate(random_pos_indices, axis=None)
	rows = np.arange(len(cols))
	data = np.ones((len(rows), ), dtype='bool')
	B = csr_matrix((data, (cols, rows)), shape=(num_labels, len(cols)))
	batch_selection = (A @ B).toarray().astype('float32')
	ip_ind = np.vstack([x[0] for x in batch])
	ip_mask = np.vstack([x[1] for x in batch])
	op_ind = np.vstack([x[4] for x in batch])
	op_mask = np.vstack([x[5] for x in batch])
	lbl_ind = cols
	return ip_ind, ip_mask, op_ind, op_mask, batch_selection, lbl_ind


def clip_batch_lengths(ind, mask, max_len):
	_max = min(np.max(np.sum(mask, axis=1)), max_len)
	return ind[:, :_max], mask[:, :_max]


def collate_fn(batch, max_len, num_labels, trn_dataset):
	batch_data = {}
	batch_size = len(batch)
	batch_data['batch_size'] = torch.tensor(batch_size, dtype=torch.int32)
	ip_ind, ip_mask, op_ind, op_mask, batch_selection, lbl_ind = _collate_fn(batch, num_labels)
	ip_ind, ip_mask = clip_batch_lengths(ip_ind, ip_mask, max_len)
	op_ind, op_mask = clip_batch_lengths(op_ind, op_mask, max_len)
	batch_data['indices'] = torch.LongTensor([item[-2] for item in batch])
	batch_data['ip_ind'] = torch.from_numpy(ip_ind)
	batch_data['ip_mask'] = torch.from_numpy(ip_mask)
	batch_data['op_ind'] = torch.from_numpy(op_ind)
	batch_data['op_mask'] = torch.from_numpy(op_mask)
	batch_data['Y'] = torch.from_numpy(batch_selection)
	batch_data['lbl_ind'] = torch.LongTensor(lbl_ind.ravel())
	batch_data['Y_mask'] = None
	return batch_data


def mean_pooling(last_hidden_state, attention_mask):
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
	return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ---- Encoder and model classes ----

class HFTransformerInputLayer(nn.Module):
	"""Sentence transformer (supports decoder models and LoRA)."""
	def __init__(self, transformer='roberta-base', decoder_model=False, lora_ft=False, lora_rank=64, \
		lora_alpha=64, lora_modules='q_proj,k_proj,v_proj,o_proj', use_liger_kernel=False, \
		img_emb_dim=None, num_imgs=1, \
		multimodal_concat_order="text_image", img_prefix=None, txt_prefix=None, \
		closing_suffix=None, decoder_model_pooling='mean', tok_max_len=32):
		super(HFTransformerInputLayer, self).__init__()
		self.img_emb_dim = img_emb_dim
		self.decoder_model = decoder_model
		self.decoder_model_pooling = decoder_model_pooling
		self.encoder_name = transformer
		self.multimodal_concat_order = multimodal_concat_order
		self.num_imgs = num_imgs
		if isinstance(transformer, str):
			if decoder_model:
				from transformers import AutoModel, AutoTokenizer
				if use_liger_kernel:
					if "Qwen2" in transformer:
						from liger_kernel.transformers import apply_liger_kernel_to_qwen2
						apply_liger_kernel_to_qwen2()
					elif "Qwen3" in transformer:
						from liger_kernel.transformers import apply_liger_kernel_to_qwen3
						apply_liger_kernel_to_qwen3()
					elif "Llama-3.2" in transformer:
						from liger_kernel.transformers import apply_liger_kernel_to_llama
						apply_liger_kernel_to_llama()
				max_len = tok_max_len
				if img_emb_dim is not None:
					tokenizer = AutoTokenizer.from_pretrained(transformer, do_lower_case=True)
					self.has_img_prefix = True
					self.has_txt_prefix = True
					self.has_closing_suffix = True
					prompt_len = 0
					if img_prefix and len(img_prefix) > 0:
						self.img_prefix = tokenizer(img_prefix, return_tensors="pt")
						prompt_len += len(self.img_prefix['input_ids'][0])
					else:
						self.has_img_prefix = False
						self.img_prefix = None
					if txt_prefix and len(txt_prefix) > 0:
						self.txt_prefix = tokenizer(txt_prefix, return_tensors="pt")
						prompt_len += len(self.txt_prefix['input_ids'][0])
					else:
						self.has_txt_prefix = False
						self.txt_prefix = None
					if closing_suffix and len(closing_suffix) > 0:
						self.closing_suffix = tokenizer(closing_suffix, return_tensors="pt")
						prompt_len += len(self.closing_suffix['input_ids'][0])
					else:
						self.has_closing_suffix = False
						self.closing_suffix = None
					max_len = tok_max_len + num_imgs + prompt_len
				else:
					tokenizer = AutoTokenizer.from_pretrained(transformer, do_lower_case=True)
					self.has_txt_prefix = True
					self.has_closing_suffix = True
					prompt_len = 0
					if txt_prefix and len(txt_prefix) > 0:
						self.txt_prefix = tokenizer(txt_prefix, return_tensors="pt")
						prompt_len += len(self.txt_prefix['input_ids'][0])
					else:
						self.has_txt_prefix = False
						self.txt_prefix = None
					if closing_suffix and len(closing_suffix) > 0:
						self.closing_suffix = tokenizer(closing_suffix, return_tensors="pt")
						prompt_len += len(self.closing_suffix['input_ids'][0])
					else:
						self.has_closing_suffix = False
						self.closing_suffix = None
					max_len = tok_max_len + prompt_len
				self.transformer = AutoModel.from_pretrained(
					transformer,
					trust_remote_code=True,
					torch_dtype=torch.bfloat16,
					max_length=max_len,
					device_map="cpu",
     				attn_implementation="eager",
         			use_cache=False,
				)
				if lora_ft:
					lora_modules = lora_modules.split(",")
					from peft import LoraConfig, get_peft_model, TaskType
					peft_config = LoraConfig(
						r=lora_rank,
						lora_alpha=lora_alpha,
						use_rslora=True,
						bias="none",
						lora_dropout=0.0,
						task_type=TaskType.FEATURE_EXTRACTION,
						target_modules=lora_modules,
					)
					self.transformer = get_peft_model(self.transformer, peft_config)
					self.transformer.print_trainable_parameters()
					self.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
			else:
				self.transformer = transformers.AutoModel.from_pretrained(transformer, trust_remote_code=True)
		else:
			self.transformer = transformer
		if img_emb_dim is not None:
			if decoder_model:
				embed_layer_dim = self.transformer.embed_tokens.weight.shape[1]
			else:
				embed_layer_dim = self.transformer.embeddings.word_embeddings.weight.shape[1]
			self.img_emb_projection = torch.nn.Linear(self.img_emb_dim, embed_layer_dim)

	def forward(self, data, img_emb=None, img_emb_mask=None, label_forward=False, epoch=None, batch_idx=None, len_dataloader=None):
		if img_emb is not None and self.decoder_model == False:
			word_embeddings = self.transformer.embeddings.word_embeddings(data['input_ids'])
			attention_mask = data['attention_mask']
			img_emb = self.img_emb_projection(img_emb)
			if self.num_imgs == 1:
				if len(img_emb.shape) == 2:
					img_emb = img_emb.unsqueeze(1)
					img_emb_mask = img_emb_mask.unsqueeze(1)
				elif len(img_emb.shape) == 1:
					img_emb = img_emb.unsqueeze(0).unsqueeze(0)
					img_emb_mask = img_emb_mask.unsqueeze(0).unsqueeze(0)
			elif self.num_imgs > 1 and len(img_emb.shape) < 3:
				img_emb = img_emb.unsqueeze(0)
				img_emb_mask = img_emb_mask.unsqueeze(0)
			word_embeddings = torch.cat((img_emb, word_embeddings), dim=1)
			attention_mask = torch.cat((img_emb_mask, attention_mask), dim=1).long()
			outputs = self.transformer(inputs_embeds=word_embeddings, attention_mask=attention_mask)
			last_hidden_state = outputs.last_hidden_state
			return mean_pooling(last_hidden_state, attention_mask)
		else:
			if img_emb is not None and self.decoder_model == True:
				word_embeddings = self.transformer.embed_tokens(data['input_ids'])
				if len(img_emb.shape) < 3:
					img_emb = img_emb.unsqueeze(0)
					img_emb_mask = img_emb_mask.unsqueeze(0)
					if len(img_emb.shape) == 2:
						img_emb = img_emb.unsqueeze(1)
						img_emb_mask = img_emb_mask.unsqueeze(1)
				img_emb = self.img_emb_projection(img_emb)
				img_prefix = None
				text_prefix = None
				closing_suffix = None
				model = self.transformer
				if label_forward is True:
					label_forward = False
				if hasattr(self, 'has_img_prefix') and self.has_img_prefix:
					img_prefix = model.embed_tokens(self.img_prefix['input_ids'].squeeze().to(word_embeddings.device))
					img_prefix = img_prefix.unsqueeze(0).expand(word_embeddings.shape[0], -1, -1)
					img_prefix_mask = self.img_prefix['attention_mask'].expand(img_emb.shape[0], -1).to(img_emb_mask.device)
				if hasattr(self, 'has_txt_prefix') and self.has_txt_prefix and not label_forward:
					text_prefix = model.embed_tokens(self.txt_prefix['input_ids'].squeeze().to(word_embeddings.device))
					text_prefix = text_prefix.unsqueeze(0).expand(word_embeddings.shape[0], -1, -1)
					text_prefix_mask = self.txt_prefix['attention_mask'].expand(word_embeddings.shape[0], -1).to(word_embeddings.device)
				if hasattr(self, 'has_closing_suffix') and self.has_closing_suffix and not label_forward:
					closing_suffix = model.embed_tokens(self.closing_suffix['input_ids'].squeeze().to(word_embeddings.device))
					closing_suffix = closing_suffix.unsqueeze(0).expand(word_embeddings.shape[0], -1, -1)
					closing_suffix_mask = self.closing_suffix['attention_mask'].expand(word_embeddings.shape[0], -1).to(word_embeddings.device)
				if self.multimodal_concat_order == "image_text":
					if not label_forward:
						if img_prefix is not None:
							new_word_embeddings = torch.cat((img_prefix, img_emb), dim=1)
						else:
							new_word_embeddings = img_emb
						if text_prefix is not None:
							new_word_embeddings = torch.cat((new_word_embeddings, text_prefix), dim=1)
						new_word_embeddings = torch.cat((new_word_embeddings, word_embeddings), dim=1)
						if closing_suffix is not None:
							new_word_embeddings = torch.cat((new_word_embeddings, closing_suffix), dim=1)
						if img_prefix is not None:
							new_attention_mask = torch.cat((img_prefix_mask, img_emb_mask), dim=1)
						else:
							new_attention_mask = img_emb_mask
						if text_prefix is not None:
							new_attention_mask = torch.cat((new_attention_mask, text_prefix_mask), dim=1)
						new_attention_mask = torch.cat((new_attention_mask, data['attention_mask']), dim=1).long()
						if closing_suffix is not None:
							new_attention_mask = torch.cat((new_attention_mask, closing_suffix_mask), dim=1).long()
				elif self.multimodal_concat_order == "text_image":
					if not label_forward:
						if text_prefix is not None:
							new_word_embeddings = torch.cat((text_prefix, word_embeddings), dim=1)
						else:
							new_word_embeddings = word_embeddings
						if img_prefix is not None:
							new_word_embeddings = torch.cat((new_word_embeddings, img_prefix), dim=1)
						new_word_embeddings = torch.cat((new_word_embeddings, img_emb), dim=1)
						if closing_suffix is not None:
							new_word_embeddings = torch.cat((new_word_embeddings, closing_suffix), dim=1)
						if text_prefix is not None:
							new_attention_mask = torch.cat((text_prefix_mask, data['attention_mask']), dim=1)
						else:
							new_attention_mask = data['attention_mask']
						if img_prefix is not None:
							new_attention_mask = torch.cat((new_attention_mask, img_prefix_mask), dim=1)
						new_attention_mask = torch.cat((new_attention_mask, img_emb_mask), dim=1).long()
						if closing_suffix is not None:
							new_attention_mask = torch.cat((new_attention_mask, closing_suffix_mask), dim=1).long()
				else:
					raise ValueError("Invalid multimodal_concat_order. Choose 'image_text' or 'text_image'.")
				data["inputs_embeds"] = new_word_embeddings
				data['attention_mask'] = new_attention_mask
				data["input_ids"] = None
			elif img_emb is None and self.decoder_model == True:
				word_embeddings = self.transformer.embed_tokens(data['input_ids'])
				text_prefix = None
				closing_suffix = None
				model = self.transformer
				if label_forward:
					label_forward = False
				if hasattr(self, 'has_txt_prefix') and self.has_txt_prefix and not label_forward:
					text_prefix = model.embed_tokens(self.txt_prefix['input_ids'].squeeze().to(word_embeddings.device))
					text_prefix = text_prefix.unsqueeze(0).expand(word_embeddings.shape[0], -1, -1)
					text_prefix_mask = self.txt_prefix['attention_mask'].expand(word_embeddings.shape[0], -1).to(word_embeddings.device)
				if hasattr(self, 'has_closing_suffix') and self.has_closing_suffix and not label_forward:
					closing_suffix = model.embed_tokens(self.closing_suffix['input_ids'].squeeze().to(word_embeddings.device))
					closing_suffix = closing_suffix.unsqueeze(0).expand(word_embeddings.shape[0], -1, -1)
					closing_suffix_mask = self.closing_suffix['attention_mask'].expand(word_embeddings.shape[0], -1).to(word_embeddings.device)
				if text_prefix is not None:
					new_word_embeddings = torch.cat((text_prefix, word_embeddings), dim=1)
				else:
					new_word_embeddings = word_embeddings
				if closing_suffix is not None:
					new_word_embeddings = torch.cat((new_word_embeddings, closing_suffix), dim=1)
				if text_prefix is not None:
					new_attention_mask = torch.cat((text_prefix_mask, data['attention_mask']), dim=1)
				else:
					new_attention_mask = data['attention_mask']
				if closing_suffix is not None:
					new_attention_mask = torch.cat((new_attention_mask, closing_suffix_mask), dim=1).long()
				data["inputs_embeds"] = new_word_embeddings
				data['attention_mask'] = new_attention_mask
				data["input_ids"] = None
			outputs = self.transformer(**data)
			if self.decoder_model_pooling == "mean":
				last_hidden_state = outputs[0]
				return mean_pooling(last_hidden_state, data["attention_mask"])
			elif self.decoder_model_pooling == "last_token":
				hidden_states = outputs.last_hidden_state
				reversed_x = torch.flip(data['attention_mask'], dims=[1])
				idx_from_end = torch.argmax((reversed_x == 1).int(), dim=1)
				last_token_indices = data['attention_mask'].size(1) - 1 - idx_from_end
				batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
				last_token_embs = hidden_states[batch_indices, last_token_indices]
				return last_token_embs


class CustomEncoder(torch.nn.Module):
	def __init__(self, encoder_name, transform_dim, decoder_model=False, \
		lora_ft=False, lora_rank=64, lora_alpha=64, lora_modules='q_proj,k_proj,v_proj,o_proj', \
		use_liger_kernel=False, \
		img_emb_dim=None, num_imgs=1, multimodal_concat_order="text_image", \
		img_prefix=None, txt_prefix=None, closing_suffix=None, decoder_model_pooling='mean', \
		tok_max_len=32):
		super(CustomEncoder, self).__init__()
		self.encoder = HFTransformerInputLayer(encoder_name, decoder_model, lora_ft, lora_rank, \
			lora_alpha, lora_modules, use_liger_kernel, \
			img_emb_dim, num_imgs, multimodal_concat_order, \
			img_prefix, txt_prefix, closing_suffix, decoder_model_pooling, tok_max_len)
		self.output_dim = self.encoder.transformer.config.hidden_size
		self.transform_dim = transform_dim
		if self.transform_dim != -1:
			self.transform = nn.Linear(self.output_dim, self.transform_dim)

	def forward(self, input_ids, attention_mask, img_emb=None, img_emb_mask=None, label_forward=False, batch_idx=None, epoch=None, len_dataloader=None):
		emb = self.encoder({'input_ids': input_ids, 'attention_mask': attention_mask}, img_emb=img_emb, img_emb_mask=img_emb_mask, label_forward=label_forward, batch_idx=batch_idx, epoch=epoch, len_dataloader=len_dataloader)
		if self.transform_dim != -1:
			return self.transform(emb)
		else:
			return emb

	@property
	def repr_dims(self):
		return self.output_dim if self.transform_dim == -1 else self.transform_dim


class SingleLayerTransformerEncoder(nn.Module):
	def __init__(self, d_model=512, n_head=1, dim_feedforward=2048, dropout=0.1, norm_first=False):
		super(SingleLayerTransformerEncoder, self).__init__()
		self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation='gelu', norm_first=norm_first)

	def forward(self, x, src_key_padding_mask=None):
		x = torch.stack(x)
		if src_key_padding_mask is not None:
			output = self.encoder_layer(x, src_key_padding_mask=src_key_padding_mask)
			valid_mask = (src_key_padding_mask == 0).float().transpose(0, 1).unsqueeze(-1).expand(-1, -1, output.shape[2])
			masked_output = output * valid_mask
			sum_output = torch.sum(masked_output, dim=0)
			valid_counts = torch.sum(valid_mask, dim=0)
			valid_counts = torch.clamp(valid_counts, min=1.0)
			mean_pooled = sum_output / valid_counts
			return mean_pooled
		else:
			output = self.encoder_layer(x)
			return torch.mean(output, 0)


class SiameseNetwork(torch.nn.Module):
	def __init__(self, args, \
				trn_doc_embs=None, trn_doc_embs_mask=None, \
				lbl_embs=None, lbl_embs_mask=None, val_doc_embs=None, val_doc_embs_mask=None):
		super(SiameseNetwork, self).__init__()
		self.padding_idx = 0
		self.encoder = CustomEncoder(
			args.encoder_name,
			args.transform_dim,
			args.decoder_model,
			args.lora_ft,
			args.lora_rank,
			args.lora_alpha,
			args.lora_modules,
			args.use_liger_kernel,
			args.img_emb_dim,
			args.multimodal_num_imgs,
			args.multimodal_concat_order,
			args.img_prefix,
			args.txt_prefix,
			args.closing_suffix,
			args.decoder_model_pooling,
			args.max_length,
		)
		self.combiner = SingleLayerTransformerEncoder(
			d_model=self.encoder.repr_dims,
			n_head=args.combiner_heads,
			dim_feedforward=args.combiner_dim,
			dropout=0.1,
		)
		self.register_buffer('cnt_representations', torch.Tensor(args.n_labels, self.encoder.repr_dims))
		if args.prime:
			if args.n_hlp > 0:
				self.weight = torch.nn.Parameter(torch.Tensor(args.n_clusters + args.n_hlp, self.encoder.repr_dims))
			else:
				self.weight = torch.nn.Parameter(torch.Tensor(args.n_clusters, self.encoder.repr_dims))
			torch.nn.init.xavier_uniform_(self.weight.data)
		self.use_images = False
		if args.img_emb_dim is not None:
			self.use_images = True
			if trn_doc_embs is not None:
				self.register_buffer('trn_doc_embs', torch.tensor(trn_doc_embs, dtype=torch.float32), persistent=False)
				self.register_buffer('trn_doc_embs_mask', torch.tensor(trn_doc_embs_mask, dtype=torch.long), persistent=False)
			self.register_buffer('lbl_embs', torch.tensor(lbl_embs, dtype=torch.float32), persistent=False)
			self.register_buffer('lbl_embs_mask', torch.tensor(lbl_embs_mask, dtype=torch.long), persistent=False)
			self.register_buffer('val_doc_embs', torch.tensor(val_doc_embs, dtype=torch.float32), persistent=False)
			self.register_buffer('val_doc_embs_mask', torch.tensor(val_doc_embs_mask, dtype=torch.long), persistent=False)
		self.device = args.device
		self.ema_w_centroids = args.ema_w_centroids
		self.label_mapping = None
		self.n_clusters = args.n_clusters
		self.n_hlp = args.n_hlp
		self.prime = args.prime

	def set_label_mapping(self, label_mapping: torch.LongTensor):
		self.label_mapping = label_mapping

	def update_clusters(self, X, trn_X_Y, n_threads=12):
		if self.n_hlp > 0:
			n_labels = trn_X_Y.shape[1]
			mapping = np.full(n_labels, fill_value=-1, dtype='int')
			freq = np.array(trn_X_Y.sum(axis=0)).ravel()
			indices = np.argsort(freq)
			vanilla_indices = indices[:-self.n_hlp]
			hlp_indices = indices[-self.n_hlp:]
			for m, ind in enumerate(hlp_indices):
				mapping[ind] = m
			self.clusters, vanilla_mapping = cluster_balance(X=X[vanilla_indices].astype('float32'), clusters=[np.arange(len(vanilla_indices), dtype='int64')], num_clusters=self.n_clusters, splitter=b_kmeans_dense, num_threads=n_threads, verbose=True)
			for i, m in zip(vanilla_indices, vanilla_mapping):
				mapping[i] = self.n_hlp + m
			self.label_mapping = torch.LongTensor(mapping)
			return mapping
		else:
			self.clusters, label_mapping = cluster_balance(X=X.astype('float32'), clusters=[np.arange(len(X), dtype='int64')], num_clusters=self.n_clusters, splitter=b_kmeans_dense, num_threads=n_threads, verbose=True)
			self.label_mapping = torch.LongTensor(label_mapping)
			return label_mapping

	def encode(self, doc_input_ids, doc_attention_mask):
		return F.normalize(self.encoder(doc_input_ids.to(self.device), doc_attention_mask.to(self.device)))

	def encode_document(self, doc_input_ids, doc_ind, doc_attention_mask, inference=False, first_forward=False, label_forward=False, epoch=None, batch_idx=None, len_dataloader=None):
		img_doc_embs = None
		img_doc_embs_mask = None
		if self.use_images and first_forward == False:
			if label_forward:
				img_doc_embs = self.lbl_embs[doc_ind.to("cpu")].squeeze().to(self.device)
				img_doc_embs_mask = self.lbl_embs_mask[doc_ind.to("cpu")].squeeze().to(self.device)
			else:
				if inference:
					img_doc_embs = self.val_doc_embs[doc_ind.to("cpu")].squeeze().to(self.device)
					img_doc_embs_mask = self.val_doc_embs_mask[doc_ind.to("cpu")].squeeze().to(self.device)
				else:
					img_doc_embs = self.trn_doc_embs[doc_ind.to("cpu")].squeeze().to(self.device)
					img_doc_embs_mask = self.trn_doc_embs_mask[doc_ind.to("cpu")].squeeze().to(self.device)
		output = self.encoder(doc_input_ids.to(self.device), doc_attention_mask.to(self.device), img_emb=img_doc_embs, img_emb_mask=img_doc_embs_mask, label_forward=label_forward, epoch=epoch, batch_idx=batch_idx, len_dataloader=len_dataloader)
		return F.normalize(output)

	def encode_label(self, lbl_input_ids, lbl_attention_mask, cls_ids, lbl_ids, label_forward=True, epoch=None, batch_idx=None, len_dataloader=None):
		sequence = []
		img_lbl_embs = None
		img_lbl_embs_mask = None
		if self.use_images:
			img_lbl_embs = self.lbl_embs[lbl_ids.to("cpu")].squeeze().to(self.device)
			img_lbl_embs_mask = self.lbl_embs_mask[lbl_ids.to("cpu")].squeeze().to(self.device)
		u = self.encoder(lbl_input_ids.to(self.device), lbl_attention_mask.to(self.device), img_emb=img_lbl_embs, img_emb_mask=img_lbl_embs_mask, label_forward=label_forward, epoch=epoch, batch_idx=batch_idx, len_dataloader=len_dataloader)
		sequence.append(u)
		if self.prime:
			fv = self.weight[cls_ids.to("cpu")].squeeze().to(self.device)
			if len(fv.shape) == 1:
				fv = fv.unsqueeze(0)
			sequence.append(fv)
			centroids = self.cnt_representations[lbl_ids.to("cpu")].squeeze().to(self.device)
			if len(centroids.shape) == 1:
				centroids = centroids.unsqueeze(0)
			sequence.append(centroids)
		if len(sequence) > 1:
			return F.normalize(u), F.normalize(self.combiner(sequence))
		elif len(sequence) == 0:
			raise Exception("No vector in the self-attention module!")
		else:
			return F.normalize(u), F.normalize(u)

	def forward(self, doc_input_ids, doc_attention_mask, lbl_input_ids, lbl_attention_mask, lbl_ind, doc_ind, inference=False, first_forward=False, label_forward=False, epoch=None, batch_idx=None, len_dataloader=None):
		if self.label_mapping is not None and lbl_ind is not None:
			cls_ids = self.label_mapping[lbl_ind.to("cpu")].to(self.device)
		else:
			cls_ids = None
		if doc_input_ids is None:
			raw_label_embeddings, label_embeddings = self.encode_label(lbl_input_ids, lbl_attention_mask, cls_ids, lbl_ind, label_forward=True, epoch=epoch, batch_idx=batch_idx, len_dataloader=len_dataloader)
			return raw_label_embeddings, label_embeddings, None, None
		elif lbl_input_ids is None:
			output = self.encode_document(doc_input_ids, doc_ind, doc_attention_mask, inference, first_forward, label_forward, epoch=epoch, batch_idx=batch_idx, len_dataloader=len_dataloader)
			return output, None, None
		doc_embeddings = self.encode_document(doc_input_ids, doc_ind, doc_attention_mask, inference, epoch=epoch, batch_idx=batch_idx, len_dataloader=len_dataloader)
		raw_label_embeddings, label_embeddings = self.encode_label(lbl_input_ids, lbl_attention_mask, cls_ids, lbl_ind, label_forward=True, epoch=epoch, batch_idx=batch_idx, len_dataloader=len_dataloader)
		return doc_embeddings, raw_label_embeddings, label_embeddings

	@property
	def repr_dims(self):
		return self.encoder.repr_dims


class MySampler(torch.utils.data.Sampler[int]):
	def __init__(self, order):
		self.order = order.copy()
	def update_order(self, x):
		self.order[:] = x[:]
	def __iter__(self):
		return iter(self.order)
	def __len__(self) -> int:
		return len(self.order)


class MyDataParallel(torch.nn.DataParallel):
	def __getattr__(self, name):
		try:
			return super().__getattr__(name)
		except AttributeError:
			return getattr(self.module, name)


@timeit
def get_lbl_embeddings(tokenization_folder, prefix, num_Z, model, max_len, trn_Y_mapping=None, bsz=2048, approx=False, pca_components=None, inference=False, bfloat16=False):
	input_ids = np.memmap(f"{tokenization_folder}/{prefix}_input_ids.dat", mode='r', shape=(num_Z, max_len), dtype='int64')
	attention_mask = np.memmap(f"{tokenization_folder}/{prefix}_attention_mask.dat", mode='r', shape=(num_Z, max_len), dtype='int64')
	if pca_components is not None:
		pca_vector = torch.FloatTensor(pca_components.T).to(model.device)
	with evaluating(model), torch.no_grad():
		for i in range(0, num_Z, bsz):
			batch_input_ids = torch.LongTensor(np.array(input_ids[i: i + bsz], copy=True))
			batch_attention_mask = torch.LongTensor(np.array(attention_mask[i: i + bsz], copy=True))
			lbl_ids = torch.arange(i, i + len(batch_attention_mask))
			batch_embeddings2 = None
			original_ids = trn_Y_mapping[lbl_ids] if trn_Y_mapping is not None else lbl_ids
			cast_type = torch.bfloat16 if bfloat16 else torch.float16
			with torch.amp.autocast('cuda', dtype=cast_type):
				raw_batch_embeddings, enriched_batch_embeddings, _, _ = model(None, None, batch_input_ids, batch_attention_mask, original_ids, None, inference=inference, label_forward=True)
			if approx:
				with torch.amp.autocast('cuda', dtype=cast_type):
					raw_batch_embeddings, orig_enriched_batch_embeddings, _, _ = model(None, None, batch_input_ids, batch_attention_mask, original_ids, None, inference=inference, label_forward=True)
				batch_embeddings = enriched_batch_embeddings
				batch_embeddings[original_ids != -1] = orig_enriched_batch_embeddings[original_ids != -1]
				batch_embeddings2 = raw_batch_embeddings
			else:
				batch_embeddings = raw_batch_embeddings
				batch_embeddings[original_ids != -1] = enriched_batch_embeddings[original_ids != -1]
			if pca_components is not None:
				batch_embeddings = torch.matmul(batch_embeddings, pca_vector)
			if i == 0:
				embeddings = np.zeros((num_Z, batch_embeddings.shape[1]))
				if batch_embeddings2 is not None:
					embeddings2 = np.zeros((num_Z, batch_embeddings2.shape[1]))
			embeddings[i: i + batch_input_ids.shape[0]] = batch_embeddings.cpu().numpy()
			if batch_embeddings2 is not None:
				embeddings2[i: i + batch_input_ids.shape[0]] = batch_embeddings2.cpu().numpy()
	return embeddings.astype('float32'), embeddings2.astype('float32') if batch_embeddings2 is not None else None


@timeit
def get_doc_embeddings(tokenization_folder, prefix, num_Z, model, max_len, bsz=2048, pca_components=None, first_forward=False, inference=False, label_forward=False, bfloat16=False):
	input_ids = np.memmap(f"{tokenization_folder}/{prefix}_input_ids.dat", mode='r', shape=(num_Z, max_len), dtype='int64')
	attention_mask = np.memmap(f"{tokenization_folder}/{prefix}_attention_mask.dat", mode='r', shape=(num_Z, max_len), dtype='int64')
	if pca_components is not None:
		pca_vector = torch.FloatTensor(pca_components.T).to(model.device)
	with evaluating(model), torch.no_grad():
		for i in tqdm(range(0, num_Z, bsz), desc="Processing batches"):
			batch_input_ids = torch.LongTensor(np.array(input_ids[i: i + bsz], copy=True))
			batch_attention_mask = torch.LongTensor(np.array(attention_mask[i: i + bsz], copy=True))
			doc_ind = torch.arange(i, i + len(batch_attention_mask))
			cast_type = torch.bfloat16 if bfloat16 else torch.float16
			with torch.amp.autocast('cuda', dtype=cast_type):
				_batch_embeddings, _, _ = model(batch_input_ids, batch_attention_mask, None, None, None, doc_ind, first_forward=first_forward, inference=inference, label_forward=label_forward)
			if pca_components is not None:
				_batch_embeddings = torch.matmul(_batch_embeddings, pca_vector)
			if i == 0:
				embeddings = np.zeros((num_Z, _batch_embeddings.shape[1]))
			embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings.cpu().numpy()
	return embeddings.astype('float32')


# ---- Training and orchestration ----

def prepare_loss(args, train_loader):
	criterion = ATripletMarginLossOHNMDM(
		args=args,
		margin_min=args.margin_min,
		margin_max=args.margin_max,
		k=args.num_negatives,
		reduction=args.reduction,
		num_violators=True,
		apply_softmax=args.agressive_loss,
		select_fixed_pos=args.fixed_pos,
		train_loader=train_loader,
	)
	return criterion


def prepare_network(args, trn_doc_embs=None, trn_doc_embs_mask=None, lbl_embs=None, lbl_embs_mask=None, val_doc_embs=None, val_doc_embs_mask=None):
	print("==> Creating model, optimizer...")
	if hasattr(args, "lora_ft") == False:
		args.lora_ft = False
	if hasattr(args, "decoder_model") == False:
		args.decoder_model = False
	snet = SiameseNetwork(
		args,
		trn_doc_embs,
		trn_doc_embs_mask,
		lbl_embs,
		lbl_embs_mask,
		val_doc_embs,
		val_doc_embs_mask,
	)
	if torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs!")
		snet = MyDataParallel(snet)
	snet.to(args.device)
	setattr(snet, 'cnt_representations', snet.cnt_representations.cpu())
	if args.multimodal_xmc:
		if trn_doc_embs is not None:
			setattr(snet, 'trn_doc_embs', snet.trn_doc_embs.cpu())
			setattr(snet, 'trn_doc_embs_mask', snet.trn_doc_embs_mask.cpu())
		setattr(snet, 'lbl_embs', snet.lbl_embs.cpu())
		setattr(snet, 'lbl_embs_mask', snet.lbl_embs_mask.cpu())
		setattr(snet, 'val_doc_embs', snet.val_doc_embs.cpu())
		setattr(snet, 'val_doc_embs_mask', snet.val_doc_embs_mask.cpu())
	torch.cuda.empty_cache()
	gc.collect()
	print(snet)
	return snet


def prepare_optimizer_and_scheduler(args, snet, t_total):
	no_decay = ['bias', 'LayerNorm.weight']
	no_lr = []
	if args.multimodal_xmc:
		no_lr = ['img_emb_projection']
	if args.lr_combiner is None:
		gp = [
			{'params': [p for n, p in snet.encoder.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in no_lr)], 'weight_decay': 0.01},
			{'params': [p for n, p in snet.encoder.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in no_lr)], 'weight_decay': 0.0},
			{'params': [p for n, p in snet.combiner.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in no_lr)], 'weight_decay': 0.01},
			{'params': [p for n, p in snet.combiner.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in no_lr)], 'weight_decay': 0.0},
		]
	else:
		gp = [
			{'params': [p for n, p in snet.encoder.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in no_lr)], 'weight_decay': 0.01},
			{'params': [p for n, p in snet.encoder.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in no_lr)], 'weight_decay': 0.0},
		]
		gp = gp + [
			{'params': [p for n, p in snet.combiner.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in no_lr)], 'weight_decay': 0.01, 'lr': args.lr_combiner},
			{'params': [p for n, p in snet.combiner.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in no_lr)], 'weight_decay': 0.0, 'lr': args.lr_combiner},
		]
	if args.multimodal_xmc:
		gp = gp + [
			{'params': [p for n, p in snet.encoder.encoder.img_emb_projection.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.multimodal_xmc_img_proj_lr},
			{'params': [p for n, p in snet.encoder.encoder.img_emb_projection.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.multimodal_xmc_img_proj_lr},
		]
	if args.prime == True:
		if args.lr_combiner is None:
			gp = gp + [{'params': snet.weight, 'weight_decay': 0.0, 'lr': args.lr}]
		else:
			gp = gp + [{'params': snet.weight, 'weight_decay': 0.0, 'lr': args.lr_combiner}]
	optimizer = torch.optim.AdamW(gp, lr=args.lr, eps=1e-06)
	t_total = t_total * args.epochs
	scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total)
	return optimizer, scheduler


def prepare_data(args):
	if not os.path.exists(args.tokenization_folder):
		print("Please create tokenization memmaps for this dataset using CreateTokenizedFiles.py as a one time effort")
		sys.exit(0)
	print("==> Creating Dataloader...")
	train_dataset = DatasetD(
		args,
		os.path.join(args.data_dir, args.trn_lbl_fname),
		args.tokenization_folder,
		args.max_length,
		args.A,
		args.B,
	)
	args.n_labels = train_dataset.labels.shape[1]
	train_order = np.random.permutation(len(train_dataset))
	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		num_workers=12,
		prefetch_factor=5,
		collate_fn=functools.partial(collate_fn, max_len=args.max_length, num_labels=train_dataset.num_labels, trn_dataset=train_dataset),
		batch_sampler=torch.utils.data.sampler.BatchSampler(MySampler(train_order), args.batch_size, False),
	)
	return train_loader


@timeit
def cluster_items(X, depth, n_threads):
	n_clusters = 2 ** (depth - 1)
	clusters, _ = cluster_balance(X=X.copy(), clusters=[np.arange(len(X), dtype='int')], num_clusters=n_clusters, splitter=b_kmeans_dense, num_threads=n_threads, verbose=True)
	clustering_mat = csr_matrix((np.ones(sum([len(c) for c in clusters])), np.concatenate(clusters), np.cumsum([0, *[len(c) for c in clusters]])), shape=(len(clusters), X.shape[0]))
	return clustering_mat


def validate(args, snet, valid_labels, mode='ova'):
	if args.trn_lbl_fname.endswith(".npz"):
		trn_X_Y = scipy.sparse.load_npz(os.path.join(args.data_dir, args.trn_lbl_fname))
	else:
		trn_X_Y = data_utils.read_gen_sparse(os.path.join(args.data_dir, args.trn_lbl_fname))
	if args.val_lbl_fname.endswith(".npz"):
		val_X_Y = scipy.sparse.load_npz(os.path.join(args.data_dir, args.val_lbl_fname))
	else:
		val_X_Y = data_utils.read_gen_sparse(os.path.join(args.data_dir, args.val_lbl_fname))
	trn_Y_mapping = torch.full(size=(val_X_Y.shape[1],), fill_value=-1)
	trn_Y_mapping[valid_labels] = torch.arange(len(valid_labels))
	label_embeddings, _ = get_lbl_embeddings(args.tokenization_folder, "lbl", val_X_Y.shape[1], snet, args.max_length, trn_Y_mapping=trn_Y_mapping, bfloat16=args.bfloat16)
	val_doc_embeddings = get_doc_embeddings(args.tokenization_folder, f"{args.val_prefix}_doc", val_X_Y.shape[0], snet, args.max_length, inference=True, bfloat16=args.bfloat16)
	trn_doc_embeddings = get_doc_embeddings(args.tokenization_folder, "trn_doc", trn_X_Y.shape[0], snet, args.max_length, bfloat16=args.bfloat16)
	np.save(os.path.join(args.result_dir, 'embeddings', 'trn.ngame.npy'), trn_doc_embeddings)
	del trn_doc_embeddings
	np.save(os.path.join(args.result_dir, 'embeddings', f'{args.val_prefix}.ngame.npy'), val_doc_embeddings)
	np.save(os.path.join(args.result_dir, 'embeddings', 'lbl.ngame.npy'), label_embeddings)
	filter_labels = None if args.filter_labels == "" else os.path.join(args.data_dir, args.filter_labels)
	print("\n\nINFERENCE LABEL SPACE\n\n")
	res, pred, metrics_dict = predict_and_eval(val_doc_embeddings, label_embeddings, val_X_Y, trn_X_Y, filter_labels, A=args.A, B=args.B, k=args.k, ks=args.ks, mode=mode)
	del val_doc_embeddings, label_embeddings
	gc.collect()
	return res, pred, metrics_dict


def train(args, snet, criterion, optimizer, scheduler, train_loader, args2save):
	if snet.prime:
		reg_criterion = RegLoss(k=args.num_negatives)
	emb_bank = np.zeros((len(train_loader.dataset), snet.repr_dims), 'float32')
	fp = open(os.path.join(args.result_dir, 'logs.txt'), 'w')
	vio_history = []
	loss_history = []
	start_time = time.time()
	val_time = 0
	n_iter = 0
	scaler = torch.amp.GradScaler('cuda')
	for epoch in range(args.current_epoch, args.epochs):
		snet.train()
		torch.set_grad_enabled(True)
		pbar = tqdm(train_loader)
		for batch_idx, data in enumerate(pbar):
			snet.zero_grad()
			cast_type = torch.bfloat16 if args.bfloat16 else torch.float16
			with torch.amp.autocast('cuda', dtype=cast_type):
				ip_embeddings, raw_op_embeddings, op_embeddings = snet(data['ip_ind'], data['ip_mask'], data['op_ind'], data['op_mask'], data['lbl_ind'], data['indices'], epoch=args.current_epoch, batch_idx=batch_idx, len_dataloader=len(train_loader))
			emb_bank[data['indices']] = ip_embeddings.detach().cpu().numpy()
			d = torch.repeat_interleave(ip_embeddings.detach().cpu(), repeats=1, dim=0)
			l_ind = data['lbl_ind']
			if args.prime and args.use_exact_centroids == False:
				snet.cnt_representations[l_ind] = snet.ema_w_centroids * snet.cnt_representations[l_ind] + (1 - snet.ema_w_centroids) * d
			loss_dict = {'epoch': epoch, 'batch': batch_idx, 'main_loss': None, 'loss_u': None, 'loss_uv': None, 'loss_i': None, 'loss_r': None, 'violators_uv': None}
			gt = data['Y'].to(args.device)
			with torch.amp.autocast('cuda', dtype=cast_type):
				sim_i = ip_embeddings @ raw_op_embeddings.T
				loss_u, violators_u = criterion(sim_i, gt)
				if snet.prime:
					loss_i, _ = criterion(raw_op_embeddings @ ip_embeddings.T, data['Y'].T.to(args.device))
					sim_f = ip_embeddings @ op_embeddings.T
					loss_uv, violators_uv = criterion(sim_f, gt)
					loss_r = reg_criterion(sim_i, sim_f, gt)
					loss = loss_u + loss_uv + loss_i + args.mi_reg_weight * loss_r
					pbar.set_description("epoch: {}, loss: {:4e}, loss_uv: {:4e}, loss_u: {:4e}, violators_uv: {}".format(epoch, loss.item(), loss_uv.item(), loss_u.item(), violators_uv.item()))
					loss_dict['loss_u'] = loss_u.item()
					loss_dict['loss_uv'] = loss_uv.item()
					loss_dict['loss_r'] = loss_r.item()
					if 'loss_i' in locals():
						loss_dict['loss_i'] = loss_i.item()
					loss_dict['main_loss'] = loss.item()
					loss_dict['violators_uv'] = violators_uv.item()
				else:
					loss = loss_u
					pbar.set_description("epoch: {}, loss: {:4e}, num_violators: {}".format(epoch, loss.item(), violators_u))
					loss_dict['loss_u'] = loss_u.item()
					if 'loss_i' in locals():
						loss_dict['loss_i'] = loss_i.item()
					loss_dict['main_loss'] = loss.item()
			loss_history.append(loss.item())
			scaler.scale(loss).backward()
			if args.decoder_model:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(snet.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			n_iter += 1

		if epoch in args.curr_steps:
			args.cl_size *= 2
			print(f"Changing cluster size to {args.cl_size}")
		if epoch >= args.cl_start:
			if (epoch - args.cl_start) % args.cl_update == 0:
				print(f'Updating clusters with cluster size {args.cl_size} (using stale embeddings)')
				embs = emb_bank.copy()
				tree_depth = int(np.ceil(np.log(embs.shape[0] / args.cl_size) / np.log(2))) + 1
				print(f"tree depth = {tree_depth}")
				cluster_mat = cluster_items(embs, tree_depth, 16).tocsr()
				del embs
				gc.collect()
			print('Updating train order...')
			cmat = cluster_mat[np.random.permutation(cluster_mat.shape[0])]
			train_loader.batch_sampler.sampler.update_order(cmat.indices)
		else:
			train_loader.batch_sampler.sampler.update_order(np.random.permutation(len(train_loader.dataset)))
		if args.use_exact_centroids == True:
			if args.use_exact_centroids_stale == True and (epoch != args.epochs - 1):
				snet.cnt_representations.copy_(torch.from_numpy(compute_centroid(emb_bank.copy(), train_loader.dataset.labels, reduction='mean')))
			else:
				initialize_cnt_repr(args, snet, train_loader.dataset.labels, first_forward=False)
		if args.save_n_epochs > 0 and epoch % args.save_n_epochs == 0:
			save_model(args, snet, scheduler, args2save, epoch + 1)
		if (args.eval_interval != -1 and ((epoch % args.eval_interval == 0) or (epoch == args.epochs - 1))):
			_t = time.time()
			res, _, metrics_dict = validate(args, snet, train_loader.dataset.valid_labels, mode=args.pred_mode)
			val_time = val_time + time.time() - _t
			fp.write(f"epoch: {epoch}\n{res}")
			fp.flush()
			metrics_csv = os.path.join(args.result_dir, 'validation_metrics.csv')
			write_header = not os.path.exists(metrics_csv)
			with open(metrics_csv, 'a', newline='') as csvfile:
				writer = csv.DictWriter(csvfile, fieldnames=['epoch'] + list(metrics_dict.keys()))
				if write_header:
					writer.writeheader()
				row = {'epoch': epoch}
				row.update(metrics_dict)
				writer.writerow(row)
	total_time = time.time() - start_time
	pickle.dump({'vio': vio_history, 'loss': loss_history}, open(os.path.join(args.result_dir, 'train_history.pkl'), 'wb'))
	fp.write(f"Total time: {total_time} sec.\n")
	fp.write(f"Validation time: {val_time}\n")
	fp.write(f"Train time: {total_time - val_time}\n")
	fp.close()


def save_model(args, snet, scheduler, args2save, epoch=0):
	print("Saving the model...")
	snet.eval()
	state_dict = {}
	for k, v in snet.state_dict().items():
		state_dict[k.replace("module.", "")] = v
	torch.save(state_dict, f"{args.model_dir}/state_dict.pt")
	with open(f"{args.model_dir}/executed_script.py", "w") as fout:
		print(inspect.getsource(sys.modules[__name__]), file=fout)
	args2save["current_epoch"] = epoch
	with open(f"{args.model_dir}/executed_script_args.txt", "w") as fout:
		json.dump(args2save, fout, indent=2)
	with open(f"{args.model_dir}/scheduler_and_rnd_st.bin", "wb") as fout:
		pickle.dump([scheduler.state_dict(), np.random.get_state(), torch.get_rng_state()], fout)
	if args.avoid_save_hf_model == False:
		save_directory = os.path.join(args.model_dir, 'hf_transformer_encoder')
		os.makedirs(save_directory, exist_ok=True)
		snet.encoder.encoder.transformer.save_pretrained(save_directory, safe_serialization=True)
		from transformers import AutoTokenizer
		tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
		tokenizer.save_pretrained(save_directory)
	snet.train()


def load_model(args, snet):
	saved_state_dict = torch.load(f"{args.model_dir}/state_dict.pt")
	snet.load_state_dict({f"{k}": v for k, v in saved_state_dict.items()})


def load_scheduler_and_seeds(args, scheduler, optimizer):
	with open(f"{args.model_dir}/scheduler_and_rnd_st.bin", "rb") as fout:
		rnd_st = pickle.load(fout)
	scheduler.load_state_dict(rnd_st[0])
	optimizer.load_state_dict(scheduler.optimizer.state_dict())
	np.random.set_state(rnd_st[1])
	torch.set_rng_state(rnd_st[2])


def load_cnt_repr(args, snet):
	saved_state_dict = torch.load(f"{args.model_dir}/state_dict.pt")
	cnt_representations = saved_state_dict['cnt_representations']
	snet.cnt_representations.copy_(cnt_representations)


def initialize_cnt_repr(args, snet, trn_X_Y, first_forward=False):
	embeddings = get_doc_embeddings(args.tokenization_folder, "trn_doc", trn_X_Y.shape[0], snet, args.max_length, first_forward=first_forward, bfloat16=args.bfloat16)
	snet.cnt_representations.copy_(torch.from_numpy(compute_centroid(embeddings, trn_X_Y, reduction='mean')))


def load_clusters(args, snet):
	label_mapping = np.load(os.path.join(args.model_dir, 'cluster_mapping.npy'))
	snet.set_label_mapping(torch.LongTensor(label_mapping))


def initialize_clusters(args, snet, trn_X_Y, first_forward=False):
	label_embeddings = get_doc_embeddings(args.tokenization_folder, "lbl", trn_X_Y.shape[1], snet, args.max_length, first_forward=first_forward, label_forward=True, bfloat16=args.bfloat16)
	label_mapping = snet.update_clusters(label_embeddings, trn_X_Y)
	if args.init_fv_w_embeddings is not None:
		if args.init_fv_w_embeddings:
			rows = []
			cols = []
			for cluster_idx, cluster_array in enumerate(snet.clusters):
				for element in cluster_array:
					rows.append(cluster_idx)
					cols.append(element)
			data = np.ones(len(rows), dtype=np.float32)
			fv_map_gt = csr_matrix((data, (rows, cols)), shape=(args.n_clusters, args.n_labels))
			snet.weight.data.copy_(torch.from_numpy(compute_centroid(label_embeddings, fv_map_gt.transpose().tocsr(), reduction='mean')).float())
		
	np.save(os.path.join(args.model_dir, 'cluster_mapping.npy'), label_mapping)


# ---- Dataset ----

class DatasetD(torch.utils.data.Dataset):
	def __init__(self, args, lbl_fname, tokenization_folder, max_len, A=0.55, B=1.5):
		self.max_len = max_len
		if lbl_fname.endswith(".npz"):
			self.labels = scipy.sparse.load_npz(lbl_fname)
		else:
			self.labels = data_utils.read_gen_sparse(lbl_fname)
		self.valid_labels = np.arange(self.labels.shape[1])
		print("#valid labels is: {}".format(len(self.valid_labels)))
		if args.pos_freq_sampling == True:
			self.prob = xc_metrics.compute_inv_propesity(self.labels, A, B)
		self.X_input_ids = np.memmap(f"{tokenization_folder}/trn_doc_input_ids.dat", mode='r', shape=(self.labels.shape[0], max_len), dtype='int64')
		self.X_attention_mask = np.memmap(f"{tokenization_folder}/trn_doc_attention_mask.dat", mode='r', shape=(self.labels.shape[0], max_len), dtype='int64')
		self.Y_input_ids = np.memmap(f"{tokenization_folder}/lbl_input_ids.dat", mode='r', shape=(self.labels.shape[1], max_len), dtype='int64')[self.valid_labels]
		self.Y_attention_mask = np.memmap(f"{tokenization_folder}/lbl_attention_mask.dat", mode='r', shape=(self.labels.shape[1], max_len), dtype='int64')[self.valid_labels]
		self.labels = self.labels.T.tocsr()[self.valid_labels].T.tocsr()
		args.img_emb_dim = None
		if args.multimodal_xmc:
			num_imgs = args.multimodal_num_imgs
			all_trn_embs = np.load(os.path.join(args.data_dir, args.multimodal_trn_img_embs))
			all_trn_embs_map = np.load(os.path.join(args.data_dir, args.multimodal_trn_img_embs_map), allow_pickle=True)
			self.trn_embs, self.trn_embs_mask = build_embedding_matrix(self.labels.shape[0], num_imgs, all_trn_embs, all_trn_embs_map)
			del all_trn_embs, all_trn_embs_map
			all_lbl_embs = np.load(os.path.join(args.data_dir, args.multimodal_lbl_img_embs))
			all_lbl_embs_map = np.load(os.path.join(args.data_dir, args.multimodal_lbl_img_embs_map), allow_pickle=True)
			self.lbl_embs, self.lbl_embs_mask = build_embedding_matrix(self.labels.shape[1], num_imgs, all_lbl_embs, all_lbl_embs_map)
			del all_lbl_embs, all_lbl_embs_map
			all_val_embs = np.load(os.path.join(args.data_dir, args.multimodal_val_img_embs))
			all_val_embs_map = np.load(os.path.join(args.data_dir, args.multimodal_val_img_embs_map), allow_pickle=True)
			# Load validation labels to know number of validation datapoints
			if args.val_lbl_fname.endswith(".npz"):
				val_X_Y = scipy.sparse.load_npz(os.path.join(args.data_dir, args.val_lbl_fname))
			else:
				val_X_Y = data_utils.read_gen_sparse(os.path.join(args.data_dir, args.val_lbl_fname))
			self.val_embs, self.val_embs_mask = build_embedding_matrix(val_X_Y.shape[0], num_imgs, all_val_embs, all_val_embs_map)
			del all_val_embs, all_val_embs_map
			# Store in args the embedding dimension
			args.img_emb_dim = self.trn_embs.shape[2]
		else:
			self.trn_embs = None
			self.trn_embs_mask = None
			self.lbl_embs = None
			self.lbl_embs_mask = None
			self.val_embs = None
			self.val_embs_mask = None
		self._pos_freq_sampling = args.pos_freq_sampling

	def __getitem__(self, index):
		"""Get a label at index"""
		# Get a randomly sampled positive data point
		pos_indices = self.labels[index].indices
		if self._pos_freq_sampling == True:
			p = self.prob[pos_indices]
			p = p / sum(p)
			pos_ind = np.random.choice(pos_indices, p=p, size=1)
		else:
			pos_ind = np.random.choice(pos_indices, size=1)
		
		return (self.X_input_ids[index], self.X_attention_mask[index],
					pos_indices, pos_ind, self.Y_input_ids[pos_ind],
					self.Y_attention_mask[pos_ind], index, pos_ind)

	def __len__(self):
		return len(self.X_input_ids)

	@property
	def num_labels(self):
		return len(self.valid_labels)
