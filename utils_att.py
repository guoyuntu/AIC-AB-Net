import json
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pickle
from build_vocab import Vocabulary
from torch.autograd import Variable 
from torchvision import transforms, datasets
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
from aic_data_loader_coco import *
from metrics import *
import subprocess

#GPU memory query
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    return result

# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile )

# Show multiple images and caption words
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
            
    Adapted from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """
    
    assert(( titles is None ) or (len( images ) == len( titles )))
    
    n_images = len( images )
    if titles is None: 
        titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        
    fig = plt.figure( figsize=( 15, 15 ) )
    for n, (image, title) in enumerate( zip(images, titles) ):
        
        a = fig.add_subplot( np.ceil( n_images/ float( cols ) ), cols, n+1 )
        if image.ndim == 2:
            plt.gray()
            
        plt.imshow( image )
        a.axis('off')
        a.set_title( title, fontsize=200 )
        
    fig.set_size_inches( np.array( fig.get_size_inches() ) * n_images )
    
    plt.tight_layout( pad=0.4, w_pad=0.5, h_pad=1.0 )
    plt.show()
    
# MS COCO evaluation data loader
class CocoEvalLoader( datasets.ImageFolder ):

    def __init__( self, root, ann_path, vocab, transform=None, target_transform=None, 
                 loader=datasets.folder.default_loader ):
        '''
        Customized COCO loader to get Image ids and Image Filenames
        root: path for images
        ann_path: path for the annotation file (e.g., caption_val2014.json)
        '''
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = json.load( open( ann_path, 'r' ) )['eval_images']
        self.vocab = vocab


    def __getitem__(self, index):

        filename = self.imgs[ index ]['filename']
        img_id = self.imgs[ index ]['image_id']
        attr_lst = self.imgs[ index ]['attributes']
        vocab = self.vocab
        
        # Filename for the image
        if 'val' in filename.lower():
            path = os.path.join( self.root, 'val2014' , filename )
        else:
            path = os.path.join( self.root, 'train2014', filename )

        img = self.loader( path )
        if self.transform is not None:
            img = self.transform( img )
            
        # Convert attributes (list) to word ids
        attr = []
        attr.extend([vocab(att) for i,att in enumerate(attr_lst) if i < 5])
        if len(attr_lst) < 5:
        	for i in range(5-len(attr_lst)):
        		attr.append(vocab('<unk>'))
        attr = torch.Tensor(attr)

        return img, attr, img_id
        
    def __len__(self):
        return len(self.imgs)

def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort( key=lambda x: len( x[1] ), reverse=True )
    images, raw_attrs, img_ids = list(zip( *data )) # unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    img_ids = list( img_ids )
    
    #Merge attributes (from tuple of 1D tensor to 2D tensor)
    lengths = [len(att) for att in raw_attrs]
    attrs = torch.zeros(len(raw_attrs), max(lengths)).long()
    for i, att in enumerate(raw_attrs):
    	end = lengths[i]
    	attrs[i, :end] = att[:end]
     
    return images, attrs, img_ids, lengths

# MSCOCO Evaluation function
def data_eval( model, args, epoch ):
    
    '''
    model: trained model to be evaluated
    args: pre-set parameters
    epoch: epoch #, for disp purpose
    '''
    
    model.eval()
    
    # Validation images are required to be resized to 224x224 already
    transform = transforms.Compose([ 
        transforms.Scale( (args.crop_size, args.crop_size) ),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load the vocabulary
    with open( args.vocab_path, 'rb' ) as f:
         vocab = pickle.load( f )
    
    # Wrapper the COCO VAL dataset
    eval_data_loader = torch.utils.data.DataLoader( 
        CocoEvalLoader( args.image_dir, args.caption_val_path, vocab, transform ), 
        batch_size = args.eval_size, 
        shuffle = False, num_workers = args.num_workers,
        drop_last = False,
        collate_fn=collate_fn )
        
    val_data_loader = get_loader( args.caption_val_path, vocab, 
                    	transform, args.eval_size,
                        shuffle=True, num_workers=args.num_workers ) 
    
    # Generated captions to be compared with GT
    results = []
    print('---------------------Start evaluation on validation set-----------------------')
    with torch.no_grad():
        for i, (images, attr, image_ids, _ ) in enumerate( eval_data_loader ):
        #for i, (images, attr, captions, lengths, _ ) in enumerate( val_data_loader ):
        
            images = to_var( images )
            attr = to_var( attr )
            generated_captions, _, _ = model.sampler( images, attr )
        
            if torch.cuda.is_available():
                captions = generated_captions.cpu().data.numpy()
            else:
                captions = generated_captions.data.numpy()
        
            # Build caption based on Vocabulary and the '<end>' token
            for image_idx in range( captions.shape[0] ):
            
                sampled_ids = captions[ image_idx ]
                sampled_caption = []
            
                for word_id in sampled_ids:
                
                    word = vocab.idx2word[ word_id ]
                    if word == '<end>':
                        break
                    else:
                        sampled_caption.append( word )
            
                sentence = ' '.join( sampled_caption )
            
                temp = { 'image_id': int( image_ids[ image_idx ] ), 'caption': sentence }
                results.append( temp )
        
            # Disp evaluation process
            if (i+1) % 10 == 0:
                print('[%d/%d] generating captions'%( (i+1),len( eval_data_loader ) )) 
            
            
    print('------------------------Caption Generated-------------------------------------')
            
    # Evaluate the results based on the COCO API
    resFile = 'results/mixed-' + str( epoch ) + '.json'
    json.dump( results, open( resFile , 'w' ) )
    
    gts = standardize_caption(args.caption_val_path)
    rng = [x['image_id'] for x in gts ]
    metrics = calculate_metrics(rng,gts,results)
    
    cider = 0.
    print('-----------Evaluation performance on validation dataset for Epoch %d----------'%( epoch ))
    for metric, score in list(metrics.items()):
        
        print('%s: %.4f'%( metric, score ))
        if metric == 'CIDEr':
            cider = score
            
    return cider

def extract_attention( model, args, epoch ):
    
    '''
    model: trained model to be evaluated
    args: pre-set parameters
    epoch: epoch #, for disp purpose
    '''
    
    model.eval()
    
    # Validation images are required to be resized to 224x224 already
    transform = transforms.Compose([ 
        transforms.Scale( (args.crop_size, args.crop_size) ),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load the vocabulary
    with open( args.vocab_path, 'rb' ) as f:
         vocab = pickle.load( f )
    
    # Wrapper the COCO VAL dataset
    eval_data_loader = get_loader( args.caption_val_path, vocab, 
                              transform, args.eval_size,
                              shuffle=True, num_workers=args.num_workers ) 
    
    attention_results = {'imgid':[],'attention':[],'beta':[],'captions':[]}
    # Generated captions to be compared with GT
    results = []
    print('---------------------Start evaluation on validation set-----------------------')
    with torch.no_grad():
        for i, (images, attr, captions, _, imgid ) in enumerate( eval_data_loader ):
        
            images = to_var( images )
            attr = to_var( attr )
            generated_captions, attention, beta = model.sampler( images, attr )
        
            if torch.cuda.is_available():
                attention = attention.cpu().data.numpy()
                beta = beta.cpu().data.numpy()
            else:
                captions = generated_captions.data.numpy()
                beta = beta.data.numpy()
                
            for j in range(len(imgid)):
            	attention_results['imgid'].append(imgid[j])
            	attention_results['attention'].append(attention[j])
            	attention_results['beta'].append(beta[j])
            	attention_results['captions'].append(captions[j])
        
            # Disp evaluation process
            if (i+1) % 10 == 0:
                print('[%d/%d] extracted attention'%( (i+1),len( eval_data_loader ) )) 
            output = open('results/attention_coco.pkl', 'wb')
            pickle.dump(attention_results, output)
            output.close()
            
    return 0