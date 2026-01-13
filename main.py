# Entry-point
import os
import json
import argparse
from utils import (
    configure_paths,
    set_seed,
    parse_curriculum,
    prepare_data,
    prepare_loss,
    prepare_network,
    prepare_optimizer_and_scheduler,
    train,
    initialize_cnt_repr,
    initialize_clusters,
    load_model,
    load_cnt_repr,
    load_clusters,
    load_scheduler_and_seeds,
    save_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("-work-dir", type=str, help="Work dir")
    parser.add_argument("-dataset", type=str, help="Dataset name", default='LF-AmazonTitles-131K')
    parser.add_argument("-dataset_subfolder", type=str, help="Dataset subfolder under data", default=None)
    parser.add_argument("--trn-lbl-fname", type=str, required=False, help="Train label file name", default="trn_X_Y.txt")
    parser.add_argument("--val-lbl-fname", type=str, required=False, help="Train label file name", default="tst_X_Y.txt")
    parser.add_argument("--val_prefix", type=str, required=False, help="Train label file name", default="tst")
    parser.add_argument("--version", type=str, help="Version of the run", default="0")
    # General training
    parser.add_argument("--seed", type=int, help="Random seed", default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    parser.add_argument("-e", "--epochs", type=int, help="The number of epochs to run for", default=600)
    parser.add_argument("-b", "--batch-size", type=int, help="The batch size", default=4096)
    parser.add_argument("-lr", type=float, help="learning rate", default=0.0002)
    parser.add_argument("--num_warmup_steps", type=int, help="Number of warmup steps", default=100)
    parser.add_argument("--device", type=str, help="device to run", default="cuda")
    parser.add_argument("--current_epoch", type=int, help="Start from N epochs", default=0)
    # PEFT
    parser.add_argument("--lora_ft", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lora_rank", type=int, help="", default=64)
    parser.add_argument("--lora_alpha", type=int, help="", default=64)
    parser.add_argument("--lora_modules", type=str, help="", default='q_proj,k_proj,v_proj,o_proj')
    # Loss
    parser.add_argument("--loss-type", type=str, help="Squared or sqrt", default='ohnm')
    parser.add_argument("--margin_min", type=float, help="Min bound for the dynamic margin", default=0.1)
    parser.add_argument("--margin_max", type=float, help="Max bound for the dynamic margin", default=0.3)
    parser.add_argument("--agressive-loss", action=argparse.BooleanOptionalAction)
    parser.add_argument("--reduction", type=str, help="mean/custom", default='mean')
    parser.add_argument("--num-negatives", type=int, help="Number of negatives to use", default=10)
    parser.add_argument("--num-violators", action="store_true", help="Should average number of violators be printed")
    # Backbone model
    parser.add_argument("--tokenizer-type", type=str, help="Tokenizer to use", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, help="Max length for tokenizer", default=32)
    parser.add_argument("--encoder-name", type=str, help="Encoder to use", default="msmarco-distilbert-base-v4")
    parser.add_argument("--transform-dim", type=int, help="Transform bert embeddings to size", default=-1)
    # PRIME
    parser.add_argument("--prime", action=argparse.BooleanOptionalAction)
    parser.add_argument("-lr_combiner", type=float, help="learning rate of combiner module", default=None)
    parser.add_argument("--combiner_heads", type=int, help="", default=1)
    parser.add_argument("--combiner_dim", type=int, help="", default=1024)
    parser.add_argument("--ema_w_centroids", type=float, help="ema_w_centroids", default=0.95)
    parser.add_argument("--n-clusters", type=int, help="", default=65536) # Free vectors
    parser.add_argument("--n-hlp", type=int, help="", default=5000)
    parser.add_argument("--mi_reg_weight", type=float, help="", default=0.1)
    parser.add_argument("--pos_freq_sampling", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fixed_pos", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_exact_centroids", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_exact_centroids_stale", action=argparse.BooleanOptionalAction)
    parser.add_argument("--init_fv_w_embeddings", action=argparse.BooleanOptionalAction)
    # NGAME negative sampling
    parser.add_argument("--cl-size", type=int, help="cluster size", default=32)
    parser.add_argument("--curr-steps", type=str, help="double cluster size at each step (csv)", default="")
    parser.add_argument("--cl-start", type=int, help="", default=999999)
    parser.add_argument("--cl-update", type=int, help="", default=5)
    # Dual-decoder Learning
    parser.add_argument("--decoder_model", action=argparse.BooleanOptionalAction)
    parser.add_argument("--decoder_model_pooling", type=str, help="Pooling strategy for the text", default="mean")
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_liger_kernel", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_gradient_checkpointing", action=argparse.BooleanOptionalAction)
    # Multi-modality
    parser.add_argument("--multimodal_xmc", action=argparse.BooleanOptionalAction)
    parser.add_argument("--multimodal_num_imgs", type=int, help="", default=1)
    parser.add_argument("-multimodal_xmc_img_proj_lr", type=float, help="learning rate image projection", default=0.001)
    parser.add_argument("--multimodal_trn_img_embs", type=str, help="", default='trn_embs_mult_imgs_3_1152_doh_siglip2.npy')
    parser.add_argument("--multimodal_val_img_embs", type=str, help="", default='tst_embs_mult_imgs_3_1152_doh_siglip2.npy')
    parser.add_argument("--multimodal_lbl_img_embs", type=str, help="", default='lbl_embs_mult_imgs_3_1152_doh_siglip2.npy')
    parser.add_argument("--multimodal_trn_img_embs_map", type=str, help="", default='train_imgs_384_True_3_doh.bin_ids_Final.npy')
    parser.add_argument("--multimodal_val_img_embs_map", type=str, help="", default='test_imgs_384_True_3_doh.bin_ids_Final.npy')
    parser.add_argument("--multimodal_lbl_img_embs_map", type=str, help="", default='labels_imgs_384_True_3_doh.bin_ids_Final.npy')
    # Prompting
    parser.add_argument("--multimodal_concat_order", type=str, help="Order of concatenation for multimodal inputs", default="image_text")
    parser.add_argument("--img_prefix", type=str, help="Prefix for image", default="This product image")
    parser.add_argument("--txt_prefix", type=str, help="Prefix for text", default="and its text")
    parser.add_argument("--closing_suffix", type=str, help="Closing prefix", default="")
    # Inference
    parser.add_argument("--pred-mode", type=str, help="ova or anns", default='ova')
    parser.add_argument("--save_n_epochs", type=int, help="Save every N epochs", default=-1)
    parser.add_argument("--save-model", action='store_true', help="Should the model be saved")
    parser.add_argument("--avoid_save_hf_model", action=argparse.BooleanOptionalAction)
    # Evaluation
    parser.add_argument("-A", type=float, help="The propensity factor A" , default=0.6)
    parser.add_argument("-B", type=float, help="The propensity factor B", default=2.6)
    parser.add_argument("--filter-labels", type=str, help="filter labels at validation time", default="filter_labels_test.txt")
    parser.add_argument("--eval-interval", type=int, help="The numbers of epochs between acc evalulation", default=30)
    parser.add_argument("--k", type=int, help="k for recall", default=100)
    parser.add_argument("--ks", type=str, help="List of K values", default='1,2,3,4,5')
    return parser

def run(args) -> None:
    args2save = args.__dict__.copy()
    configure_paths(args)
    print(args.tokenization_folder)
    set_seed(args.seed)
    parse_curriculum(args)

    train_loader = prepare_data(args)
    args2save["n_labels"] = args.n_labels
    criterion = prepare_loss(args, train_loader)

    snet = prepare_network(
        args,
        train_loader.dataset.trn_embs,
        train_loader.dataset.trn_embs_mask,
        train_loader.dataset.lbl_embs,
        train_loader.dataset.lbl_embs_mask,
        train_loader.dataset.val_embs,
        train_loader.dataset.val_embs_mask,
    )

    optimizer, scheduler = prepare_optimizer_and_scheduler(args, snet, len(train_loader))

    if args.current_epoch > 0:
        load_model(args, snet)
        if args.prime:
            load_cnt_repr(args, snet)
            load_clusters(args, snet)
            load_scheduler_and_seeds(args, scheduler, optimizer)
    else:
        if args.prime:
            initialize_cnt_repr(args, snet, train_loader.dataset.labels, first_forward=False)
            initialize_clusters(args, snet, train_loader.dataset.labels, first_forward=False)

    train(args, snet, criterion, optimizer, scheduler, train_loader, args2save)

    if args.save_model:
        save_model(args, snet, scheduler, args2save, args.epochs)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.resume:
        args_file = os.path.join(args.work_dir, 'models' , "X-M1", args.dataset, args.version, "executed_script_args.txt")
        try:
            with open(args_file, 'r') as f:
                args2save = json.load(f)
            args.__dict__.update(args2save)
        except Exception:
            print(f"File {args_file} not found!")
            raise SystemExit(1)
        if args.current_epoch == args.epochs:
            print("Training done!")
            raise SystemExit(1)

    print(args)
    run(args)


if __name__ == "__main__":
    main()
