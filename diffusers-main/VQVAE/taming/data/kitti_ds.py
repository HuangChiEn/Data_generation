from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class Kitti_ds(Dataset):

    def __init__(self, root_dir: Path, sub_dir: str = 'sematic_im', subset: str = 'training'):
        # Path object glob method return iterator, so we immediately turn it into list
        self.ds_dir = data_dir / sub_dir / subset
        self.imgs_path = list( self.ds_dir.glob('image_2/*.png') )
        self.classes = list( self.ds_dir.glob('sematic/*.png') )
        self.clr_instances = list( self.ds_dir.glob('sematic_rgb/*.png') )
        self.instances = list( self.ds_dir.glob('instance/*.png') )
        
        self.imgs_path.sort() ; self.classes.sort() ; self.clr_instances.sort() ; self.instances.sort()
        
        self.random_flip = True

    def __pil_read(self, path, read_mode='RGB'): 
        with Image.open(path) as meta_im:
            meta_im.load()
        return meta_im.convert(read_mode)

    def __getitem__(self, index: int) -> Any:
        im_path = self.imgs_path[idx]
        img = self.__pil_read(im_path)

        cls_path, inst_path, clr_msk_path = self.classes[idx], self.instances[idx], self.clr_instances[idx]
        clr_msk_im = self.__pil_read(clr_msk_path)
        cls_im = self.__pil_read(cls_path, read_mode='L')
        inst_im = self.__pil_read(inst_path, read_mode='L')
        
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

    def __len__(self) -> int:
        return len(self.imgs_path)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]    
























