import os
import math
import random
import warnings

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

# import guard, we only expose the functions for loading data and setup transformation!!
__all__ = ['load_data', 'build_trfs'] 

def load_data(data_dir='', catch_dir='', resize_size=None, subset_type='train', ret_dataset=True, random_flip=True):
    # given dir chking..
    data_dir, catch_dir = Path(data_dir), Path(catch_dir)
    if not (data_dir.exists() or catch_dir.exsits()):
        raise ValueError("Plz at least setup one, either data_dir or catch_dir!!")

    # cache mode, early return !!
    if cache_dir:
        cache_path = list( (catch_path).glob('*.pt') )
        return Cityscape_cache(cache_path)

    if not subset_type in ['train', 'val', 'all']:
        raise ValueError(f"Doesn't support subset_type mode {subset_type}, it should be one of it ('train' or 'val' or 'all')!!")
    
    subset_type = ['train', 'val'] if subset_type == 'all' else [subset_type]
    all_ds = {}.from_keys(subset_type)
    for subset in subset_type:
        all_ds[subset] = Cityscape_ds(data_dir, subset, resize_size, random_flip)

    if ret_dataset:
        return all_ds

    # wrap with torch dataloader ~
    for subset in all_ds.keys():
        all_ds[subset] = DataLoader(all_ds[subset])
    return all_ds

class Cityscape_ds(Dataset):
    def __init__(self, data_dir, subset, resize_size, random_flip):
        super().__init__()
        self.resize_size = resize_size
        self.random_flip = random_flip
        # Path object glob method return iterator, so we immediately turn it into list
        self.imgs_path = list( (data_dir / 'leftImg8bit' / subset).glob('*/*.png') )
        # get corresponding label(s)
        self.classes, self.instances, self.clr_instances = [], [], []
        for lab_path in (data_dir / 'gtFine' / subset).glob('*/*.png'):
            if lab_path.endswith('_labelIds.png'):
                self.classes.append(lab_path)
            elif lab_path.endswith('_instanceIds.png'):
                self.instances.append(lab_path)
            elif lab_path.endswith('_color.png'):
                self.clr_instances.append(lab_path)
            else:
                warnings.warn(f"Unidentified file : {lab_path}")

    def __pil_read(self, path, read_mode='RGB'): 
        with Image.open(path) as meta_im:
            meta_im.load()
        return meta_im.convert(read_mode)

    # confirm: two stage for all of input tensor (img, inst_im, ..., etc.) to resize
    def __resize(self, pil_im, resample_method=Image.BICUBIC, keep_aspect=True):
        while min(*pil_im.size) >= 2 * self.resize_size:
        pil_im = pil_im.resize(
            tuple(x // 2 for x in pil_im.size), resample=resample_method
        )

        if keep_aspect:
            scale = image_size / min(*pil_im.size)
            resiz_pil_im = pil_im.resize(
                tuple(round(x * scale) for x in pil_im.size), resample=Image.BICUBIC
            )
        else:
            resiz_pil_im = pil_im.resize((self.resize_size, self.resize_size), resample=Image.BICUBIC)

        return resiz_pil_im

    # confirm: crop_size is the same for all of input tensor (img, inst_im, ..., etc.)
    def __center_crop(self, pil_im, crop_size):
        arr_im = np.array(pil_im)
        crop_x = (arr_im.shape[1] - crop_size[1]) // 2
        return arr_im[:, crop_x: crop_x + crop_size[1]]

    def __shape_cal(self, img, ):
        h, w = img.size
        

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
        img = self.__resize(img, Image.BOX)
        # labels resize with lower quality but better performance!
        clr_msk_im, cls_im, inst_im = \
            self.__resize(clr_msk_im, Image.NEAREST), self.__resize(cls_im, Image.NEAREST), self.__resize(inst_im, Image.NEAREST)
        
        # center-crop (hard-code 1080, 1440 currently..)
        sc = 1080 // self.resize_size  
        crop_size = ( resize_size, 1440//sc )

        img = self.__center_crop(img, crop_size)
        clr_msk_im, cls_im, inst_im = \
            self.__center_crop(clr_msk_im, crop_size), self.__center_crop(cls_im, crop_size), self.__center_crop(inst_im, crop_size)
        
        if self.random_flip and random.random() < 0.5:
            img = img[:, ::-1].copy()
            clr_msk_im = clr_msk_im[:, ::-1].copy()
            cls_im = cls_im[:, ::-1].copy() 
            inst_im = inst_im[:, ::-1].copy() 

        img = img.astype(np.float32) / 127.5 - 1
        out_dict = {'path':im_path, 'label_ori':cls_im, 'label':cls_im[None, ], 
                    'instance':inst_im[None, ], 'clr_instance':clr_msk_im[None, ]}
                    
        # switch to channel first format due to torch tensor..
        return {"pixel_values":np.transpose(img, [2, 0, 1]), "label":out_dict}
        
    def __len__(self):
        return len(self.all_files)


class Cityscape_cache(Dataset):
    def __init__(self):
        ...
    def __getitem__(self):
        ...
    def __len__(self):
        ...



class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_mode,
        resolution,
        image_paths,
        classes=None,
        instances=None,
        clr_instances=None,
        random_crop=False,
        random_flip=True,
        is_train=True,
        sub_size=None,
        use_vae=True,
        catch_path=False,
        mask_emb="resize"
    ):
        super().__init__()
        self.is_train = is_train
        self.catch_mode = catch_path is not None
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.sub_size = sub_size

        self.local_images = image_paths
        self.local_classes = None if classes is None else classes
        self.local_instances = None if instances is None else instances
        self.local_clr_instances = None if clr_instances is None else clr_instances

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.use_vae = use_vae
        self.mask_emb = mask_emb

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        ## Note that catch_mode haven't support cache RGB-mask!
        if self.catch_mode:
            if self.dataset_mode == 'cityscapes':
                file = torch.load(self.local_images[idx])
                mean = file['x']['mean']
                std = file['x']['std']
                sample = torch.randn(mean.shape)
                x = mean + std * sample
                x = x * 0.18215
                if self.mask_emb == "resize":
                    return {"pixel_values": x, "label": file['label']}
                    
                elif self.mask_emb == "vae_encode":
                    label_latent = []
                    mean = file['label']['mean']
                    std = file['label']['std']

                    for m,s in zip(mean, std):
                        sample = torch.randn(m.shape)
                        sample = m + s * sample
                        sample = sample * 0.18215
                        label_latent.append(sample)
                    y = torch.cat(label_latent, dim=0)
                    return {"pixel_values": x, "y": y}
                else:
                    return {"pixel_values": x, "label": file['label']}
        else:
            
            if self.dataset_mode == 'cityscapes':
                # if self.sub_size is not None:
                #
                #self.resolution = (self.resolution,self.resolution*2)
                # label don't need to resize 
                Label_size={270 : (33, 45), 540 : (67, 90), 1080 : (135, 180)}
                if self.resolution in [270, 540, 1080]:
                    sc = 1080//self.resolution
                    label_size = Label_size[self.resolution] if self.use_vae and self.mask_emb == "resize" else None
                    arr_image, arr_class, arr_instance, arr_clr_instance = resize_arr([pil_image, pil_class, pil_instance, pil_clr_instance], self.resolution, True, ( self.resolution, 1440//sc ), label_size)
                else:
                    arr_image, arr_class, arr_instance, arr_clr_instance = resize_arr([pil_image, pil_class, pil_instance, pil_clr_instance], self.resolution)

            
            if self.random_flip and random.random() < 0.5:
                arr_image = arr_image[:, ::-1].copy()
                arr_class = arr_class[:, ::-1].copy()
                arr_instance = arr_instance[:, ::-1].copy() if arr_instance is not None else None
                arr_clr_instance = arr_clr_instance[:, ::-1].copy() if arr_instance is not None else None

            arr_image = arr_image.astype(np.float32) / 127.5 - 1

            out_dict['path'] = path
            out_dict['label_ori'] = arr_class.copy()

            if self.dataset_mode == 'ade20k':
                arr_class = arr_class - 1
                arr_class[arr_class == 255] = 150
            elif self.dataset_mode == 'coco':
                arr_class[arr_class == 255] = 182

            out_dict['label'] = arr_class[None, ]

            if arr_instance is not None:
                out_dict['instance'] = arr_instance[None, ]
            if arr_clr_instance is not None:
                out_dict['clr_instance'] = arr_clr_instance[None, ]

            return {"pixel_values":np.transpose(arr_image, [2, 0, 1]), "label":out_dict}

# leave 
def resize_arr(pil_list, image_size, keep_aspect=True, crop_size=None, label_size = None):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance, pil_clr_instance = pil_list
    breakpoint()
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if keep_aspect:
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    else:
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)

    if label_size is not None:
        pil_class = pil_class.resize( (label_size[0] * 2,label_size[0]), resample=Image.NEAREST)
    else:
        pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)

    if pil_instance is not None:
        if label_size is not None:
            pil_instance = pil_instance.resize((label_size[0] * 2, label_size[0]), resample=Image.NEAREST)
        else:
            pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    if pil_clr_instance is not None:
        if label_size is not None:
            pil_clr_instance = pil_clr_instance.resize((label_size[0] * 2, label_size[0]), resample=Image.NEAREST)
        else:
            pil_clr_instance = pil_clr_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    arr_clr_instance = np.array(pil_clr_instance) if pil_clr_instance is not None else None

    # center crop.. convert to numpy arr 
    
    else:
        return arr_image, arr_class, arr_instance, arr_clr_instance


def center_crop_arr(pil_list, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = (arr_image.shape[0] - image_size) // 2
    crop_x = (arr_image.shape[1] - image_size) // 2
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None


def random_crop_arr(pil_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = random.randrange(arr_image.shape[0] - image_size + 1)
    crop_x = random.randrange(arr_image.shape[1] - image_size + 1)
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None

# default collate function
def collate_fn(examples):
    segmap = {}
    for k in examples[0]["label"].keys():
        if k != 'path':
            segmap[k] = torch.stack([torch.from_numpy(example["label"][k]) for example in examples])
            segmap[k] = segmap[k].to(memory_format=torch.contiguous_format).float()    

    pixel_values = torch.stack([torch.from_numpy(example["pixel_values"]) for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    filename_lst = [ os.path.basename(example['label']['path']) for example in examples ]
    
    return {"pixel_values": pixel_values, "segmap": segmap, 'filename': filename_lst}

# unittest..
if __name__ == "__main__":
    ...