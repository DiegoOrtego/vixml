import pandas as pd
import polars as pl
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import requests
from io import BytesIO
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

def getIds(path):
    ids = []
    with open(path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            current_line = line.strip().split('->')
            ids.append(current_line[0].strip())

    dataset = pl.DataFrame({'id': ids})
    print(f'Number of ids: {len(ids)}')
    return dataset

def getUrls(path):
    ids = []
    urls = []
    with open(path, 'r') as file:
        for line in file:
            current_line = line.strip().split('\t')
            ids.append(current_line[0].strip())
            urls.append(current_line[1].strip())

    urls_dataframe = pd.DataFrame({'id': ids, 'url': urls})
    print('Shape: ', urls_dataframe.shape)
    return urls_dataframe

def load_img(img, size=(384, 384), add_padding=True):
    img = img.convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    if add_padding:
        final_size = img.size
        delta_w = size[0] - final_size[0]
        delta_h = size[1] - final_size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        l_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
    else:
        l_img = img
    with BytesIO() as im_file:
        l_img.save(im_file, format="JPEG")
        value = base64.b64encode(im_file.getvalue()).decode('utf-8')
    return value

def download_image(url):
    if not url or not url.startswith(('http://', 'https://')):
        return None
    try:
        res = requests.get(url, stream=True)
        if res.status_code == 200:
            img = Image.open(BytesIO(res.content))
            return load_img(img)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None

def write_imgs_file(dataset, datapath, name, fixed_start_ids_file=None, prod_ids_split=None, num_threads=10):
    index = 0
    fixed_start_prod_id = 0
    all_ids = prod_ids_split["id"].to_numpy()
    ind_prods = np.empty(len(all_ids), dtype=object)  # To store lists of row indices or -1 for each product
    buffer = []
    f = open(f"{datapath}/{name}.bin", "ab")


    if fixed_start_ids_file is not None:
        last_prod_id, index = np.array(fixed_start_ids_file.split(".")[1].split("_")[2:]).astype(int)
        fixed_start_prod_id = last_prod_id + 1
        ind_prods = np.load(os.path.join(datapath, fixed_start_ids_file), allow_pickle=True)
    
    id_to_urls = {
        row['id']: row['url']
        for row in dataset.group_by('id', maintain_order=True).agg(pl.col('url')).to_dicts()
    }

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_prod = {
            executor.submit(download_image, url): (i_prod, url)
            for i_prod, prod_id in enumerate(tqdm(all_ids[fixed_start_prod_id:], desc="Submitting download tasks"))
            for url in id_to_urls[prod_id]
        }
        gc.collect()  # Force garbage collection to free up memory

        buffer = []
        for future in tqdm(as_completed(future_to_prod), total=len(future_to_prod), desc="Downloading images"):
            i_prod, url = future_to_prod[future]
            img = future.result()
            if img is not None:
                # Add the current index to the list of rows for the product
                if ind_prods[i_prod] is None or ind_prods[i_prod] == -1:
                    ind_prods[i_prod] = []
                ind_prods[i_prod].append(index)

                # Add the image to the buffer
                buffer.append(f"{img}\n")
                index += 1

                # Write buffer in larger chunks
                if len(buffer) >= 1000:
                    f.write("".join(buffer).encode('utf-8'))
                    buffer.clear()
            else:
                # If no image is downloaded for this product, set -1
                if ind_prods[i_prod] is None:
                    ind_prods[i_prod] = -1
            
            
        # Write remaining buffer
        if buffer:
            f.write("".join(buffer).encode('utf-8'))

    # Save the ind_prods array to a file
    np.save(f"{datapath}/{name}_map.npy", ind_prods)
    f.close()

    print(f"failed Prods: {len(np.where(ind_prods == -1)[0])}")


if __name__ == '__main__':
    img_size = (384, 384)
    add_padding = True # Adds white pixels to images to have the desired dimension without changing the aspect ratio
    num_imgs_per_prod = 3
    datapath = '/rnd_ai_datasets4/dataFAIR/xmc/data/LF-AmazonTitles-131K/'
    # datapath = '/rnd_ai_datasets4/dataFAIR/xmc/data/LF-AmazonTitles-1.3M/'
    # datapath = '/rnd_ai_datasets4/dataFAIR/xmc/data/MM-AmazonTitles-300K/'

    training_dataset = getIds(os.path.join(datapath, 'raw_data/train.raw.txt'))
    testing_dataset = getIds(os.path.join(datapath, 'raw_data/test.raw.txt'))
    labels_dataset = getIds(os.path.join(datapath, 'raw_data/label.raw.txt'))
    
    urls_dataframe = pl.read_parquet(os.path.join(datapath, 'raw_data', 'img_urls.parquet'))
    
    train_dataset = training_dataset.join(urls_dataframe, on='id', how='left')
    test_dataset = testing_dataset.join(urls_dataframe, on='id', how='left')
    label_dataset = labels_dataset.join(urls_dataframe, on='id', how='left')
    
    print(f'image_size: {img_size}')
    print(f'num_imgs_per_prod: {num_imgs_per_prod}')
    print(f'add_padding: {add_padding}')

    files_and_contents = [
        (train_dataset, datapath, f'train_imgs_{img_size[0]}_{num_imgs_per_prod}', None, training_dataset),
        (label_dataset, datapath, f'labels_imgs_{img_size[0]}_{num_imgs_per_prod}', None, labels_dataset),
        (test_dataset, datapath, f'test_imgs_{img_size[0]}_{num_imgs_per_prod}', None, testing_dataset),
    ]

    for dataset, datapath, file, fixed_start_ids_file, prod_ids_split in files_and_contents:
        write_imgs_file(dataset, datapath, file, fixed_start_ids_file, prod_ids_split, num_threads=16)


