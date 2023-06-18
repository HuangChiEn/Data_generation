# python std-pkg
import os
import math
import random
import warnings
from pathlib import Path
from typing import Callable
from functools import partial

# 3rd pkgs
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

# import guard, we only expose the following functions!!
__all__ = ['load_data', 'collate_fn'] 

global DS_DICT  # hoisting.. see javascript hoisting

def load_data(
    data_dir:str = 'None', 
    cache_dir:str = 'None', 
    cache_file_callbk:Callable = None, 
    resize_size:tuple = None, 
    subset_type:str = 'train', 
    ret_dataset:bool = True, 
    random_flip:bool = True, 
    data_ld_kwargs:dict = None
):
    if 'cityscape' in data_dir.lower():
        ds_mode = 'cityscape' 
    elif 'kitti' in data_dir.lower(): 
        ds_mode = 'kitti'
    else:
        raise ValueError(f"Currently we don't support dataset by given {data_dir}")

    # given dir chking..
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    # cache mode, early return !!
    if cache_dir.exists():
        warnings.warn(f"Apply cache file, cache file only return train subset.")
        return DS_DICT['cache'][ds_mode](cache_dir, cache_file_callbk)

    if not data_dir.exists():
        raise ValueError(f"data_dir '{data_dir}' is not exists!")

    if not subset_type in ['train', 'val', 'all']:
        raise ValueError(f"Doesn't support subset_type mode {subset_type}, it should be one of it ('train' or 'val' or 'all')!!")
    
    subset_type = ['train', 'val'] if subset_type == 'all' else [subset_type]
    all_ds = {}
    for subset in subset_type:
        all_ds[subset] = DS_DICT['ds'][ds_mode](data_dir, subset, resize_size, random_flip)

    # wrap with torch dataloader ~
    if not ret_dataset:
        for subset in all_ds.keys():
            all_ds[subset] = DataLoader(all_ds[subset], collate_fn=collate_fn, **data_ld_kwargs)
    
    # integrate with previous version, ret dict while enable 'all' subset_type
    return all_ds[subset_type[0]] if len(subset_type) == 1 else all_ds

class Cityscape_ds(Dataset):
    def __init__(self, data_dir, subset, resize_size, random_flip=True, rnd_rng=True):
        super().__init__()
        self.resize_size = resize_size
        self.random_flip = random_flip
        self.rnd_rng = rnd_rng
        # Path object glob method return iterator, so we immediately turn it into list
        self.imgs_path = list( (data_dir / 'leftImg8bit' / subset).glob('*/*.png') )
        # get corresponding label(s)
        self.classes, self.instances, self.clr_instances = [], [], []
        for lab_path in (data_dir / 'gtFine' / subset).glob('*/*.png'):
            if str(lab_path).endswith('_labelIds.png'):
                self.classes.append(lab_path)
            elif str(lab_path).endswith('_instanceIds.png'):
                self.instances.append(lab_path)
            elif str(lab_path).endswith('_color.png'):
                self.clr_instances.append(lab_path)
            else:
                warnings.warn(f"Unidentified file : {lab_path}")
        # sort lst to confirm the order
        self.imgs_path.sort() ; self.classes.sort() ; self.instances.sort() ; self.clr_instances.sort()


    def __pil_read(self, path, read_mode='RGB'): 
        with Image.open(path) as meta_im:
            meta_im.load()
        return meta_im.convert(read_mode)

    def __ratio_resize(self, pil_im, resample_method):
        min_len = min(*pil_im.size)
        # directly resize wrt. scale : "ratio is kept"  or "input square image, H == W"
        if (self.resize_size % min_len == 0) or len(set(pil_im.size)) == 1:
            scale = self.resize_size / min_len
            resiz_pil_im = pil_im.resize(
                tuple(int(x * scale) for x in pil_im.size), resample=Image.BOX
            )

        ## two-stage resize to mimic the distortion of resize
        # 1. stage : keep ratio downsampling (fast)
        while min(*pil_im.size) >= 2 * self.resize_size:
            pil_im = pil_im.resize(
                tuple(x // 2 for x in pil_im.size), resample=Image.BOX
            )
        # 2. stage : bicubic style, carefully resize without keeping ratio
        scale = self.resize_size / min(*pil_im.size)
        resiz_pil_im = pil_im.resize(
            tuple(round(x * scale) for x in pil_im.size), resample=resample_method
        )
        return resiz_pil_im

    # confirm: crop_size is the same for all of input tensor (img, inst_im, ..., etc.)
    def __center_crop(self, pil_im, crop_size, rnd_rng=False):
        arr_im = np.array(pil_im)
        if 'int' in str(type(rnd_rng)):
            crop_x = rnd_rng
            return arr_im[:, crop_x: crop_x + crop_size[1]]
        elif rnd_rng:
            samp_rng = (arr_im.shape[1] - crop_size[1])
            crop_x = np.random.randint(0, samp_rng)
            return (arr_im[:, crop_x: crop_x + crop_size[1]], crop_x)
        else:
            crop_x = (arr_im.shape[1] - crop_size[1]) // 2
            return arr_im[:, crop_x: crop_x + crop_size[1]]
        
    def __getitem__(self, idx):
        # read img from pil format
        im_path = self.imgs_path[idx]
        img = self.__pil_read(im_path)

        # read labels from pil format
        cls_path, inst_path, clr_msk_path = self.classes[idx], self.instances[idx], self.clr_instances[idx]
        clr_msk_im = self.__pil_read(clr_msk_path)
        # read cls and instance mask with binary mode 'L'!!
        cls_im = self.__pil_read(cls_path, read_mode='L')
        inst_im = self.__pil_read(inst_path, read_mode='L')
        
        # resize img & labels
        # dwn img with better quality : https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
        img = self.__ratio_resize(img, Image.BICUBIC)
        # labels resize with lower quality but better performance!
        clr_msk_im, cls_im, inst_im = \
            self.__ratio_resize(clr_msk_im, Image.NEAREST), self.__ratio_resize(cls_im, Image.NEAREST), self.__ratio_resize(inst_im, Image.NEAREST)
        
        # center-crop (hard-code 1080, 1440 currently..)
        sc = 1080 // self.resize_size  
        crop_size = ( self.resize_size, 1440//sc )

        if self.rnd_rng:
            img, crop_x = self.__center_crop(img, crop_size, self.rnd_rng)
            fix_crop = partial(self.__center_crop, rnd_rng=crop_x)
            clr_msk_im, cls_im, inst_im = \
                fix_crop(clr_msk_im, crop_size), fix_crop(cls_im, crop_size), fix_crop(inst_im, crop_size)
        else:
            img = self.__center_crop(img, crop_size, False)
            clr_msk_im, cls_im, inst_im = \
                self.__center_crop(clr_msk_im, crop_size), self.__center_crop(cls_im, crop_size), self.__center_crop(inst_im, crop_size)
        
        if self.random_flip and random.random() < 0.5:
            img = img[:, ::-1].copy()
            clr_msk_im = clr_msk_im[:, ::-1].copy()
            cls_im = cls_im[:, ::-1].copy() 
            inst_im = inst_im[:, ::-1].copy() 

        img = img.astype(np.float32) / 127.5 - 1
        
        out_dict = {'path':im_path, 'label_ori':cls_im, 'label':cls_im[None, ...], 
                    'instance':inst_im[None, ...], 'clr_instance':clr_msk_im[None, ...]}
                    
        # switch to channel first format due to torch tensor..
        return {"pixel_values":np.transpose(img, [2, 0, 1]), "label":out_dict}
        
    def __len__(self):
        return len(self.imgs_path)


class Cityscape_cache(Dataset):
    VAE_SCALE = 0.18215

    def __init__(self, cache_dir, cache_file_callbk=None):
        self.cache_path = list( (cache_dir).glob('train/*/*.pt') )
        self.cache_file_callbk = cache_file_callbk
        
    def __getitem__(self, idx):
        vae_cache = torch.load(self.cache_path[idx])
        # customized callback to deal with cache file
        if self.cache_file_callbk:
            return self.cache_file_callbk(vae_cache)
        
        # default procedure to load the cache file for VAE & VQVAE.. 
        if isinstance(vae_cache['x'], dict):
            mean, std = vae_cache['x']['mean'], vae_cache['x']['std']
            # normal_ make more efficient with std=1, mean=0 (exactly as randn) 
            # https://pytorch.org/docs/stable/generated/torch.randn.html
            sample = torch.cuda.FloatTensor(**mean.shape).normal_(mean=0, std=1)
            x = mean + std * sample
            x = x * Cityscape_cache.VAE_SCALE
        elif isinstance(vae_cache['x'], list):
            ret = random.randint(0, len(vae_cache['x'])-1)
            x = vae_cache['x'][ret] * Cityscape_cache.VAE_SCALE
            vae_cache['label']["segmap"] = vae_cache['label']["segmap"][ret]
        else:
            x = vae_cache['x']
            print(x.shape)
        return {"pixel_values": x, "label": vae_cache['label']}

    def __len__(self):
        return len(self.cache_path)

class Kitti_cache:
    VAE_SCALE = 0.18215

    def __init__(self, cache_dir, cache_file_callbk=None):
        self.cache_path = list( (cache_dir).glob('train/*/*.pt') )
        self.cache_file_callbk = cache_file_callbk
        
    def __getitem__(self, idx):
        vae_cache = torch.load(self.cache_path[idx])
        # customized callback to deal with cache file
        if self.cache_file_callbk:
            return self.cache_file_callbk(vae_cache)
        
        # default procedure to load the cache file for VAE & VQVAE.. 
        if isinstance(vae_cache['x'], dict):
            mean, std = vae_cache['x']['mean'], vae_cache['x']['std']
            # normal_ make more efficient with std=1, mean=0 (exactly as randn) 
            # https://pytorch.org/docs/stable/generated/torch.randn.html
            sample = torch.cuda.FloatTensor(**mean.shape).normal_(mean=0, std=1)
            x = mean + std * sample
            x = x * Kitti_cache.VAE_SCALE
        elif isinstance(vae_cache['x'], list):
            ret = random.randint(0, len(vae_cache['x'])-1)
            x = vae_cache['x'][ret] * Kitti_cache.VAE_SCALE
        else:
            x = vae_cache['x']
            print(x.shape)
        return {"pixel_values": x, "label": vae_cache['label']}

    def __len__(self):
        return len(self.cache_path)

class Kitti_ds:
    def __init__(self, data_dir, subset, resize_size, random_flip=True, rnd_rng=True, interval=15):
        
        def grap_path(data_dir):
            # Path object glob method return iterator, so we immediately turn it into list
            imgs_path = list( data_dir.rglob('**/**/image_02/data/*.png') )
            # sort lst to confirm the order
            imgs_path.sort()
            return imgs_path

        def sample_interval(imgs_path, interval):
            from os.path import sep
            path_lst = []
            prv_dr, cur_dr = '', ''
            cnt = 0
            for path in imgs_path:
                path = str(path)
                for dr in path.split(sep):
                    if 'drive' in dr: cur_dr = dr ; break
                
                if cur_dr != prv_dr:
                    cnt = 0 ; prv_dr = cur_dr

                if (cnt % interval) == 0:
                    path_lst.append(path)
                
                cnt += 1
            
            return path_lst 


        super().__init__()
        self.resize_size = resize_size
        self.random_flip = random_flip
        self.rnd_rng = rnd_rng
        self.subset = subset
        
        imgs_path = grap_path( Path(data_dir) )
        self.imgs_path = sample_interval(imgs_path, interval)

    def __pil_read(self, path, read_mode='RGB'): 
        with Image.open(path) as meta_im:
            meta_im.load()
        return meta_im.convert(read_mode)

    def __ratio_resize(self, pil_im, resample_method):
        min_len = min(*pil_im.size)
        # directly resize wrt. scale : "ratio is kept"  or "input square image, H == W"
        if (self.resize_size % min_len == 0) or len(set(pil_im.size)) == 1:
            scale = self.resize_size / min_len
            resiz_pil_im = pil_im.resize(
                tuple(int(x * scale) for x in pil_im.size), resample=Image.BOX
            )

        ## two-stage resize to mimic the distortion of resize
        # 1. stage : keep ratio downsampling (fast)
        while min(*pil_im.size) >= 2 * self.resize_size:
            pil_im = pil_im.resize(
                tuple(x // 2 for x in pil_im.size), resample=Image.BOX
            )
        # 2. stage : bicubic style, carefully resize without keeping ratio
        scale = self.resize_size / min(*pil_im.size)
        resiz_pil_im = pil_im.resize(
            tuple(round(x * scale) for x in pil_im.size), resample=resample_method
        )
        return resiz_pil_im

    # confirm: crop_size is the same for all of input tensor (img, inst_im, ..., etc.)
    def __center_crop(self, pil_im, crop_size, rnd_rng=False):
        arr_im = np.array(pil_im)
        if rnd_rng:
            samp_rng = (arr_im.shape[1] - crop_size[1])
            crop_x = np.random.randint(0, samp_rng)
        else:
            crop_x = (arr_im.shape[1] - crop_size[1]) // 2

        return arr_im[:, crop_x: crop_x + crop_size[1]]
        
    def __getitem__(self, idx):
        # read img from pil format
        im_path = self.imgs_path[idx]
        img = self.__pil_read(im_path)
        
        # resize img & labels
        # dwn img with better quality : https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
        img = self.__ratio_resize(img, Image.BICUBIC)

        # center-crop (hard-code 1080, 1440 currently..)
        sc = 1080 // self.resize_size  
        crop_size = ( self.resize_size, 1440//sc )

        if self.rnd_rng:
            img = self.__center_crop(img, crop_size, self.rnd_rng)
        else:
            img = self.__center_crop(img, crop_size, False)
        
        if self.random_flip and random.random() < 0.5:
            img = img[:, ::-1].copy()
        
        img = img.astype(np.float32) / 127.5 - 1
        # switch to channel first format due to torch tensor..
        return {"pixel_values":np.transpose(img, [2, 0, 1]), "label":{'path':im_path} }
        
    def __len__(self):
        return len(self.imgs_path)


DS_DICT = {
    'cache' : {
        'cityscape' : Cityscape_cache,
        'kitti' : Kitti_cache
    },
    'ds' : {
        'cityscape' : Cityscape_ds,
        'kitti' : Kitti_ds
    }
}

# default collate function
def collate_fn(examples):
    segmap = {}
    ret = random.randint(0, 1)
    for k in examples[0]["label"].keys():
        if k != 'path':
            if isinstance(examples[0]["label"][k], list):
                segmap[k] = torch.stack([torch.from_numpy(example["label"][k][ret]) for example in examples])
            else:
                segmap[k] = torch.stack([torch.from_numpy(example["label"][k]) for example in examples])
            segmap[k] = segmap[k].to(memory_format=torch.contiguous_format).float()

    pixel_values = torch.stack([torch.from_numpy(example["pixel_values"]) for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    filename_lst = [ os.path.join(example['label']['path']) for example in examples ]
    
    return {"pixel_values": pixel_values, "segmap": segmap, 'filename': filename_lst}

# unittest..
if __name__ == "__main__":
    from easy_configer.Configer import Configer
    cfger = Configer()
    cfger.cfg_from_str('''
    [ds1]
        data_dir = '/data/joseph/kitti_ds'
        resize_size = 540
        subset_type = 'train'

    [ds2]
        data_dir = '/data1/dataset/Cityscapes'
        resize_size = 540
        subset_type = 'train'

    [ld]
        batch_size = 8
        num_workers = 0
        shuffle = True
    ''')

    import torch
    from torch.utils.data import ConcatDataset
    import torchvision.transforms as T

    kitti_ds = load_data(**cfger.ds1)
    city_ds = load_data(**cfger.ds2)
    train_dataset = ConcatDataset([kitti_ds, city_ds])
    
    data_ld = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **cfger.ld
    )

    fn_lst = []
    tnsr2pil = T.ToPILImage()
    for idx, batch in enumerate(data_ld, 0):
        fn = batch['filename'][0]
        fn = os.path.basename(fn).split('.')[0]
        
        # Note : kitti_ds doesn't have mask..
        #clr_msk = [ clr_inst.permute(0, 3, 1, 2) / 255. for clr_inst in batch["segmap"]['clr_instance'] ][0]
        #tnsr2pil(clr_msk[0]).save(f'{fn}_clr.png')

        imgs = [((img+1)*127.5)/255. for img in batch["pixel_values"]]
        tnsr2pil(imgs[0]).save(f'tmp_im/{fn}_im.png')
        if idx >= 65:
            break  # yeah ~ take the break
    