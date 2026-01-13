import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import torch
import os
import numpy as np
from io import BytesIO
from PIL import Image
import base64

min_pixels = 188*28*28
max_pixels = 188*28*28

class DatasetBase(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir,
                 f_features,
                 iterate_over_features = True,
                 transform = None,
                 *args, **kwargs):

        self.data_dir = data_dir
        self.f_features = f_features
        self.iterate_over_features = iterate_over_features

        self.transform = transform
        self.pos_imgs_features = self.create_index(os.path.join(data_dir, f_features))


    def update_iterate_over_features(self, iterate_over_features: bool):
        self.iterate_over_features = iterate_over_features
        
    def create_index(self, fname):
        """Creates an Index of the position that each line starts
        with the aim of make faster lectures to the .bin files.
        """ 
        positions = []
        with open(fname, 'r') as file:
            pos = file.tell()
            while file.readline():
                positions.append(pos)
                pos = file.tell()
        return np.array(positions)
  
    def read_img_bin(self, dat: str):
        """Decode from base 64 and open an Image object
        """
        return Image.open(BytesIO(base64.b64decode(dat)))
    
    def load_image(self, type: str, index: int):
        """Read the image from .bin file and returns the image
        either for datapoits or labels.
        """
        line_positions = list()
        fname = ''

        if type == 'labels':
            line_positions = self.pos_imgs_labels[index]
            fname = os.path.join(self.data_dir, self.f_label_features)  
        elif type == 'datapoints':
            line_positions = self.pos_imgs_features[index]
            fname = os.path.join(self.data_dir, self.f_features) 
        else:
            raise Exception("")

        with open(fname, 'r') as file:
            file.seek(line_positions)
            line = file.readline().strip()
        img = self.read_img_bin(line)
        if hasattr(self.transform, "max_pixels"):
            images_kwargs = {'return_tensors': 'pt'}
            return self.transform(images=img, **images_kwargs)
        else:
            # For other models, we just return the pixel values
            return self.transform(images=img, return_tensors="pt")
    
    def __len__(self):
        if self.iterate_over_features:
            return len(self.pos_imgs_features)
        else:
            return len(self.pos_imgs_labels)
        
    def __getitem__(self, index):
        img = self.load_image(type='datapoints', index = index)
        return img, index

def collate_fn(batch):
    all_doc_fts, indices = zip(*batch)
    doc_fts = []
    grid = []
    for i in range(len(all_doc_fts)):
        doc_fts.append(all_doc_fts[i]["pixel_values"])
        if "image_grid_thw" in all_doc_fts[i]:
            grid.append(all_doc_fts[i]["image_grid_thw"])
    data = {}
    data['X'] = torch.stack(doc_fts)
    data['grid'] = torch.stack(grid) if grid else None
    data['batch_size'] = torch.tensor(len(batch), dtype=torch.int32) 
    data['indices'] = torch.LongTensor([item[-1] for item in batch])
    return data

def get_embeddings(encoder, dataset, embedding_dim, batch_size = 550, num_workers = 12, device = 'cuda', model_name= None):
    dt_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=4
    )
    torch.set_grad_enabled(False)
    embeddings = np.zeros((
        dt_loader.dataset.__len__(),
        embedding_dim),
        dtype='float32')
    
    count = 0
    for batch_data in tqdm(dt_loader, desc="Computing Embeddings"):
        batch_size = batch_data['batch_size']
        with autocast(device_type='cuda', dtype=torch.float16):
            out_ans = encoder.get_image_features(pixel_values=batch_data['X'].squeeze().to(encoder.device))
            embeddings[count:count+batch_size, :] = out_ans.detach().cpu().numpy()
        count += batch_size
    torch.cuda.empty_cache()
    return embeddings

if __name__ == '__main__':

    data_dir = '/rnd_ai_datasets4/dataFAIR/xmc/data/'
    dataset = 'LF-AmazonTitles-131K'
    # dataset = 'LF-AmazonTitles-1.3M'
    model_name = "google/siglip2-so400m-patch14-384"
    bsz = 32
    
    emb_size = 1152
    tst = {
        'input_ft':'test_imgs_384_3.bin',
        'output_embs':'tst_embs_imgs_3_1152_siglip2.npy'
    }
    trn = {
        'input_ft':'train_imgs_384_3.bin',
        'output_embs':'trn_embs_imgs_3_1152_siglip2.npy'
    }
    lbl = {
        'input_ft':'labels_imgs_384_3.bin',
        'output_embs':'lbl_embs_imgs_3_1152_siglip2.npy'
    }
        
    
    # At the time of working on this, there was an error loading the processor for google/siglip2-so400m-patch14-384, 
    # so we take the previous generation processor. https://arxiv.org/pdf/2502.14786
    # The paper does not mention updates in the processor 
    processor_name = "google/siglip-so400m-patch14-384"
    model = AutoModel.from_pretrained(model_name, device_map="auto").eval()    
    processor = AutoProcessor.from_pretrained(processor_name)
    print(model)

    # Load Test Dataset
    db = DatasetBase(
        data_dir=os.path.join(data_dir, dataset),
        f_features=tst['input_ft'],
        transform=processor,
        iterate_over_features = True,
    )

    # # Get the embeddings for the test data points
    embeddings = get_embeddings(model, db, embedding_dim=emb_size, batch_size = bsz, model_name=model_name) 
    np.save(os.path.join(data_dir, dataset, tst['output_embs']), embeddings)
    
    
    # Load Label Dataset
    db = DatasetBase(
        data_dir=os.path.join(data_dir, dataset),
        f_features=lbl['input_ft'],
        transform=processor,
        iterate_over_features = True,
    )

    # # Get the embeddings for the test data points
    embeddings = get_embeddings(model, db, embedding_dim=emb_size, batch_size = bsz) 
    np.save(os.path.join(data_dir, dataset, lbl['output_embs']), embeddings)

    # Load Train Dataset
    db = DatasetBase(
        data_dir=os.path.join(data_dir, dataset),
        f_features=trn['input_ft'],
        transform=processor,
        iterate_over_features = True,
    )

    # Get the embeddings for the train data points
    embeddings = get_embeddings(model, db, embedding_dim=emb_size, batch_size = bsz )
    np.save(os.path.join(data_dir, dataset, trn['output_embs']), embeddings)


