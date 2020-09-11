import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import string
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import json


class Dataset(data.Dataset):
    def __init__(self, anno_path, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """

        self.anno_path = anno_path
        self.vocab = vocab
        self.transform = transform
        self.annodata = json.load(open(anno_path,'r'))

    def __getitem__(self, index):
        """Returns one data pair ( image, caption, image_id )."""
        anno_path = self.anno_path
        vocab = self.vocab
        anno_data = self.annodata
        caption = anno_data['annotations'][index]
        img_id = anno_data['imgid'][index]
        attr_lst = anno_data['new_attributes'][index]
        filename = anno_data['filename'][index]
        
        if 'train' in filename:
            train_path = os.path.join('/home/guoyun/resized/train2014',filename)
            image = Image.open( train_path ).convert('RGB')
        elif 'val' in filename:
        	val_path = os.path.join('/home/guoyun/resized/val2014',filename)
        	image = Image.open( val_path ).convert('RGB')
            
        if self.transform is not None:
            image = self.transform( image )
            
		# Convert attributes (list) to word ids
        attr = []
        attr.extend([vocab(att) for i,att in enumerate(attr_lst) if i < 5])
        if len(attr_lst) < 5:
        	for i in range(5-len(attr_lst)):
        		attr.append(vocab('<unk>'))
        attr = torch.Tensor(attr)

        # Convert caption (string) to word ids.
        tokens = str( caption ).lower().translate( str.maketrans('','', string.punctuation) ).strip().split()
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        
        return image, attr, target, img_id

    def __len__(self):
        anno_data = self.annodata
        return len( anno_data['annotations'] )
        
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        attributes: torch tensor of shape (batch_size, 5).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
        img_ids: image ids in COCO dataset, for evaluation purpose
        filenames: image filenames in COCO dataset, for evaluation purpose
    """

    # Sort a data list by caption length (descending order).
    data.sort( key=lambda x: len( x[1] ), reverse=True )
    images, raw_attrs, captions, img_ids = list(zip( *data )) # unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    img_ids = list( img_ids )
    
    # Merge attributes (from tuple of 1D tensor to 2D tensor)
    lengths = [len(att) for att in raw_attrs]
    attrs = torch.zeros(len(raw_attrs), max(lengths)).long()
    for i, att in enumerate(raw_attrs):
    	end = lengths[i]
    	attrs[i, :end] = att[:end]

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, attrs, targets, lengths, img_ids


def get_loader(anno_path, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    fashion = Dataset(anno_path=anno_path,
                       vocab=vocab,
                       transform=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=fashion, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader