import torch
from taming.data import vq_dataset, cityscape_ds_alpha
from tqdm import tqdm, trange
import torchvision as tv
import os
from taming.models.vqvae import VQSub
from diffusers import AutoencoderKL, DDPMScheduler, VQModel
import numpy as np
from pathlib import Path

def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()

def preprocess_input(data, drop_rate=0.0):
    # move to GPU and change data types

    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = 34
    input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        b_msk = (data['instance']!=0)
        # binary mask to gpu-device and turn type to float
        instance_edge_map = b_msk.to(data['instance'].device).float()
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

    if drop_rate > 0.0:
        mask = (torch.rand([input_semantics.shape[0], 1, 1, 1]) > drop_rate).float()
        input_semantics = input_semantics * mask

    cond = {key: value for key, value in data.items() if key not in ['label', 'instance', 'label_ori']}
    cond['y'] = input_semantics

    return cond['y'].cuda().half()

def main(use_fp16=True, data_dir='/data1/dataset/Cityscapes', batch_size=8, image_size=540, generate_catch_file=True):

    #vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = VQModel.from_pretrained("CompVis/ldm-super-resolution-4x-openimages", subfolder="vqvae")
    #vae = VQSub.from_pretrained("/data/harry/Data_generation/diffusers-main/VQVAE/VQ_model/70ep", subfolder="vqvae")
    #vae = VQSub.from_pretrained("/data/harry/Data_generation/diffusers-main/VQVAE/SPADE_VQ_model_V2/99ep", subfolder="vqvae")

    if use_fp16:
        vae.to(dtype=torch.float16)
        #vae_off.to(dtype=torch.float16)
    vae.requires_grad_(False)
    vae = vae.cuda()
    vae.eval()

    def collate_fn(examples):
        segmap = {}
        for k in examples[0]["label"].keys():
            if k != "path":
                segmap[k] = torch.stack( [torch.from_numpy(example["label"][k]) if isinstance(example["label"][k],np.ndarray) else example["label"][k] for example in examples])
                segmap[k] = segmap[k].to(memory_format=torch.contiguous_format).float()

        pixel_values = torch.stack([torch.from_numpy(example["pixel_values"]) if isinstance(example["pixel_values"],np.ndarray) else example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        filename_lst = [os.path.join(example['label']['path']) for example in examples]

        return {"pixel_values": pixel_values, "segmap": segmap, 'filename': filename_lst}

    dataset = cityscape_ds_alpha.load_data(
        data_dir,
        resize_size=image_size,
        subset_type='train',
        #cache_dir="/data/harry/Cityscape_catch/VQ_model_V2",
        #ret_dataset=False,
        #data_ld_kwargs={'batch_size':batch_size}
    )
    data = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=8,
    )

    cnt = 0
    sav_fd = '/data/harry/Cityscape_catch/VQ_model_official/train/'
    #sav_fd = "vqvae_ress"
    if not os.path.isdir(sav_fd):
        os.makedirs(sav_fd)
    import copy

    for batch in tqdm(data):
        images, cond = batch['pixel_values'], batch['segmap']
        segmap = preprocess_input(batch["segmap"]) if "segmap" not in batch["segmap"].keys() else batch["segmap"]["segmap"].cuda().half()
        images = images.cuda()

        #clr_msks = [clr_inst.permute(0, 3, 1, 2) / 255. for clr_inst in batch["segmap"]['clr_instance']]

        if generate_catch_file:
            segmap_flip = segmap.flip(3)
            quant_flip = vae.encode(images.type(torch.float16).flip(3)).latents
            quant = vae.encode(images.type(torch.float16)).latents
            for i, (q, seg, q_f, seg_f, name) in enumerate(zip(quant, segmap, quant_flip, segmap_flip, batch['filename'])):
                q = q.cpu()
                q_f = q_f.cpu()
                path = sav_fd + name.split('/')[-2] + "/" + name.split('/')[-1].replace(".png", ".pt")
                if Path(path).exists():
                    print(f"load pt file from {path}")
                    data_dic = torch.load(path)
                    # if len(data_dic["x"]) > 4:
                    #     print(f"replace")
                    #     data_dic["x"][4] = q
                    #     data_dic["x"][5] = q_f
                    #     data_dic["label"]["segmap"][4] = seg.cpu()
                    #     data_dic["label"]["segmap"][5] = seg_f.cpu()
                    # else:
                    data_dic["x"] += [q, q_f]
                    data_dic["label"]["segmap"] += [seg.cpu(), seg_f.cpu()]
                else:
                    data_dic = {"x": [q, q_f], "label": {"segmap": [seg.cpu(), seg_f.cpu()], "path": name}}
                    if not os.path.isdir(sav_fd + name.split('/')[-2]):
                        os.makedirs(sav_fd + name.split('/')[-2])
                torch.save(data_dic, path)
        else:
            #samples, _ = vae(images.type(torch.float16), segmap)
            samples = vae.decode(images.type(torch.float16)/0.18215, segmap)
            clr_msks = samples
            for idx, (im, samp, clr) in enumerate(zip(images, samples, clr_msks)):
                tv.utils.save_image((im + 1) / 2.0, f"{sav_fd}/gt{cnt + 1}_im{idx}.png")
                tv.utils.save_image((samp + 1) / 2.0, f"{sav_fd}/out{cnt + 1}_im{idx}.png")
                # tv.utils.save_image((fl + 1) / 2.0, f"{sav_fd}/flip{cnt+1}_im{idx}.png")
                tv.utils.save_image(clr, f"{sav_fd}/segmap{cnt + 1}_im{idx}.png")
            #loop-breaker
            if cnt>=0 :
                break
            cnt += 1

if __name__ == "__main__":
    main()
