# python std-pkg
import os
import math
import random
import warnings
from pathlib import Path
from typing import Callable

# 3rd pkgs
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import random

# import guard, we only expose the following functions!!
__all__ = ['load_data', 'collate_fn'] 

def load_data(
    data_dir:str = 'None', 
    cache_dir:str = 'None', 
    cache_file_callbk:Callable = None, 
    resize_size:tuple = None, 
    subset_type:str = 'train', 
    fn_qry:str = '*/*.png',
    random_flip:bool = False,
    ret_dataset:bool = True, 
    data_ld_kwargs:dict = None
):
    # given dir chking..
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    # cache mode, early return !!
    if cache_dir.exists():
        warnings.warn(f"Apply cache file, cache file only return train subset.")
        return Cityscape_cache(cache_dir, cache_file_callbk)

    if not data_dir.exists():
        raise ValueError(f"data_dir '{data_dir}' is not exists!")

    if not subset_type in ['train', 'val', 'all']:
        raise ValueError(f"Doesn't support subset_type mode {subset_type}, it should be one of it ('train' or 'val' or 'all')!!")
    
    subset_type = ['train', 'val'] if subset_type == 'all' else [subset_type]
    all_ds = {}.fromkeys(subset_type)
    for subset in subset_type:
        all_ds[subset] = Cityscape_ds(data_dir, subset, resize_size, random_flip, fn_qry)

    if ret_dataset:
        # integrate with previous version, ret dict while enable 'all' subset_type
        return all_ds[subset_type[0]] if len(subset_type) == 1 else all_ds

    # wrap with torch dataloader ~
    for subset in all_ds.keys():
        all_ds[subset] = DataLoader(all_ds[subset], collate_fn=collate_fn, **data_ld_kwargs)
    return all_ds

class Cityscape_ds(Dataset):   
    DEBUG = False
    def __init__(self, data_dir, subset, resize_size, random_flip, fn_qry):
        def get_tag(path):
            tmp = str(path).split('/')[-1].split('.')[-2].split('_')
            city, fid, bid, _ = tmp[0], tmp[1], tmp[2], tmp[3]
            return city, fid, bid

        super().__init__()
        self.resize_size = resize_size
        self.random_flip = random_flip
        # Path object glob method return iterator, so we immediately turn it into list
        self.imgs_path = list( (data_dir / 'leftImg8bit' / subset).glob(fn_qry) )
        # get corresponding label(s)
        self.classes, self.instances, self.clr_instances = [], [], []
        
        for lab_path in (data_dir / 'gtFine' / subset).glob(fn_qry):
            # Loading path for classes
            if str(lab_path).endswith('_labelIds.png'):
                self.classes.append(lab_path)
            # Loading path for instance
            elif str(lab_path).endswith('_edgeMaps.png'):
                self.instances.append(lab_path)
            elif str(lab_path).endswith('_crEdgeMaps.png'):
                continue
            elif str(lab_path).endswith('_instanceIds.png'):
                continue   
            # Loading path for color instance
            elif str(lab_path).endswith('_color.png'):
                self.clr_instances.append(lab_path)
            else:
                warnings.warn(f"Unidentified file : {lab_path}")
        
        # sort lst to confirm the order
        self.imgs_path.sort() ; self.clr_instances.sort()  # img-pairs
        self.classes.sort() ; self.instances.sort()  # instances is edge map
        
        if Cityscape_ds.DEBUG:
            for im, cs, ins in zip(self.imgs_path, self.classes, self.instances):
                im_city, im_fid, im_bid = get_tag(im)
                cs_city, cs_fid, cs_bid = get_tag(cs)
                ins_city, ins_fid, ins_bid = get_tag(ins)
                # assert img ids consistent
                assert im_city == cs_city == ins_city
                assert im_fid == cs_fid == ins_fid
                assert im_bid == cs_bid == ins_bid
            
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

    # commit the requirement for SRDC's resolution specification 
    def __center_crop(self, arr_im, crop_size, contro=None):
        crop_x=0
        if contro is not None and contro != "center":
            if contro == "right":
                crop_x = arr_im.shape[1] - crop_size[1]
            elif contro == "left":
                crop_x = 0
            else:
                print("error")
        else:
            crop_x = (arr_im.shape[1] - crop_size[1]) // 2
        return arr_im[:, crop_x: crop_x + crop_size[1]]

    def get_data(self, idx, contro = None):
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
            self.__ratio_resize(clr_msk_im, Image.NEAREST), self.__ratio_resize(cls_im,
                                                                                Image.NEAREST), self.__ratio_resize(
                inst_im, Image.NEAREST)

        # assert img shape consistent
        if Cityscape_ds.DEBUG:
            assert img.size == clr_msk_im.size == cls_im.size == inst_im.size

        # convert PIL2np.arr
        img = np.array(img) ; clr_msk_im = np.array(clr_msk_im) ; cls_im = np.array(cls_im) ; inst_im = np.array(inst_im)

        if self.resize_size % 256 != 0:
            # center-crop (hard-code 1080, 1440 currently..)
            sc = 1080 // self.resize_size
            crop_size = (self.resize_size, 1440 // sc)

            img = self.__center_crop(img, crop_size, contro)
            clr_msk_im, cls_im, inst_im = \
                self.__center_crop(clr_msk_im, crop_size, contro), self.__center_crop(cls_im, crop_size, contro), self.__center_crop(
                    inst_im, crop_size, contro)

        if self.random_flip and random.random() < 0.5:
            img = img[:, ::-1].copy()
            clr_msk_im = clr_msk_im[:, ::-1].copy()
            cls_im = cls_im[:, ::-1].copy()
            inst_im = inst_im[:, ::-1].copy()

        # assert img shape consistent
        if Cityscape_ds.DEBUG:
            assert img.size == clr_msk_im.size
            assert cls_im.size == inst_im.size

        # print(im_path)
        img = img.astype(np.float32) / 127.5 - 1
        out_dict = {'path': im_path, 'label_ori': cls_im, 'label': cls_im[None,],
                    'instance': inst_im[None,], 'clr_instance': clr_msk_im[None,]}

        # switch to channel first format due to torch tensor..
        return {"pixel_values": np.transpose(img, [2, 0, 1]), "label": out_dict}

    def __getitem__(self, idx):
        return self.get_data(idx, "center")
        
    def __len__(self):
        return len(self.imgs_path)


class Cityscape_cache(Dataset):
    VAE_SCALE = 0.18215
    #VAE_SCALE = 7.706491063029163
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
            print(len(vae_cache['x']))
            x = vae_cache['x'][ret] * Cityscape_cache.VAE_SCALE
            vae_cache['label']["segmap"] = vae_cache['label']["segmap"][ret]
        else:
            #print("error")
            x = vae_cache['x'] * Cityscape_cache.VAE_SCALE

        return {"pixel_values": x, "label": vae_cache['label']}

    def __len__(self):
        return len(self.cache_path)


def collate_fn(examples):
    #print(len(examples))
    segmap = {}
    ret = random.randint(0, 1)
    for k in examples[0]["label"].keys():
        if k != 'path':
            if isinstance(examples[0]["label"][k], list):
                segmap[k] = torch.stack([torch.from_numpy(example["label"][k][ret]) for example in examples])
            else:
                segmap[k] = torch.stack([torch.from_numpy(example["label"][k]) if isinstance(example["pixel_values"],np.ndarray) else example["pixel_values"] for example in examples])

            segmap[k] = segmap[k].to(memory_format=torch.contiguous_format).float()

    if isinstance(examples[0]["pixel_values"], list):
        pixel_values = torch.stack([torch.from_numpy(example["pixel_values"][ret]) for example in examples])
    else:
        pixel_values = torch.stack([torch.from_numpy(example["pixel_values"]) for example in examples])

    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    filename_lst = [os.path.join(example['label']['path']) for example in examples]
    # print(filename_lst)

    return {"pixel_values": pixel_values, "segmap": segmap, 'filename': filename_lst}


# unittest..
if __name__ == "__main__":
    from pathlib import Path

    ds = Cityscape_ds(Path('/data1/dataset/Cityscapes'), 'val', 512, True, '[a-z]*/*.png')  
    tmp = DataLoader(ds, collate_fn=collate_fn, batch_size=8)
    
    for kk in tmp:
        breakpoint()

    
