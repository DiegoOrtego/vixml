# Official code for ViXML paper @ AAAI 2026.

Code for "_Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework_" published in AAAI 2026

## Requirements

- Check requirements.txt file.

### Expected directory structure
Wherever you place a root folder the structure must be as follows:

```txt
+-- <root_dir>
|  +-- data
|    +-- <dataset_name>
|  +-- models
|  +-- results
```
Basically, you place all datasets under data folder and running the training code creates models and results folders (creating subfolders with the same <dataset_name>)


### Download data for ViXML

#### Textual data 
```txt
* Download the (zipped file) raw data from The XML repository [5].  
* Extract the zipped file into data directory.
* Process the following files using extract_text_and_labels.py
* The following files should be available in <work_dir>/data/<dataset> (create empty filter file if unavailable):
    - trn.raw.txt
    - tst.raw.txt
    - lbl.raw.txt
    - trn_X_Y.txt
    - tst_X_Y.txt
    - filter_labels_text.txt
```

#### Images
Download XXX.
You may use one of the following options:

    - We provide SIGLIP2 embeddings for the datasets used in the paper, so after downloading place embedding information (<split_name>_embs_imgs_3_1152_siglip2.npy and <split_name>_imgs_384_3_map.npy) in the corresponding <datase_name> folder.
    - We provide the image paths of LF-AmazonTitles datasets in the file img_urls.parquet. Run download_amazon_imgs.py (set internally the datapath to the corresponding <dataset_name>) to download the images (<split_name>_imgs_384_3.bin and <split_name>_imgs_384_3_map.npy). Then, run extract_img_embed.py to extract image embeddings with SIGLIP2, i.e. creating <split_name>_embs_imgs_3_1152_siglip2.npy. For MM-AmazonTitles-300K, you can get the urls from the official dataset and, then, create a img_urls.parquet file to proceed as in LF-AmazonTitles datasets.

All resulting files mentioned above must be placed at <dataset_name> folder.

### Run ViXML

We provide two scripts to run encoder and decoder alternatives in LF-AmazonTitles-131K. See run_vixml_miniLML3_amzTitles131K.sh and run_vixml_qwen25_3B_amzTitles131K.sh.

```bash
./run_vixml_qwen25_3B_amzTitles131K.sh <gpu_id> LF-AmazonTitles-131K
```

## Cite as

```bib
@InProceedings{Ortego26,
    author = "Ortego, D., Rodr{\'i}guez, M., Almagro, M., Dahiya, K., Jim{\'e}nez, D. and SanMiguel, J.C.",
    title = "Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework",
    booktitle = "Association for the Advancement of Artificial Intelligence Conference on Artificial Intelligence (AAAI)",
    year = 2026}
