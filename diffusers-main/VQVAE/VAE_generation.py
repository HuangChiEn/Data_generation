import torch
from taming.data import vq_dataset
from tqdm import tqdm, trange
import torchvision as tv
import os
from taming.models.vqvae import VQSub
from diffusers import AutoencoderKL, DDPMScheduler, VQModel

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
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

    if drop_rate > 0.0:
        mask = (torch.rand([input_semantics.shape[0], 1, 1, 1]) > drop_rate).float()
        input_semantics = input_semantics * mask

    cond = {key: value for key, value in data.items() if key not in ['label', 'instance', 'label_ori']}
    cond['y'] = input_semantics

    return cond['y'].cuda().half()

def main(use_fp16=True,data_dir='/data1/dataset/Cityscapes',batch_size=8,image_size=540, mask_emb="resize"):

    #vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    #vae_off = VQModel.from_pretrained("CompVis/ldm-super-resolution-4x-openimages", subfolder="vqvae")
    #vae = VQSub.from_pretrained("/data/harry/Data_generation/diffusers-main/VQVAE/VQ_model/70ep", subfolder="vqvae")
    vae = VQSub.from_pretrained("/data/harry/Data_generation/diffusers-main/VQVAE/NOSPADE_VQ_model/70ep", subfolder="vqvae")

    if use_fp16:
        vae.to(dtype=torch.float16)
        #vae_off.to(dtype=torch.float16)
    vae.requires_grad_(False)
    vae = vae.cuda()
    vae.eval()

    # vae_off.requires_grad_(False)
    # vae_off = vae_off.cuda()
    # vae_off.eval()

    # train_dataset = cityscape_ds.load_data(
    #     data_dir=data_dir,
    #     cache_dir="/data/harry/Cityscape_catch/our_VQVAE_540_resize"
    # )

    dataset = vq_dataset.load_data(
        data_dir,
        resize_size=image_size,
        subset_type='train',
        name_qry="DA_Data/*",
        #ret_dataset=False,
        #data_ld_kwargs={'batch_size':batch_size}

    )
    data = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=vq_dataset.collate_fn,
        batch_size=batch_size,
        num_workers=8,
    )


    cnt = 0
    #sav_fd = 'vqvae_ress'
    sav_fd = "/data/harry/Cityscape_catch/our_NOSPADEVQVAE_540_resize/train/"
    if not os.path.isdir(sav_fd):
        os.makedirs(sav_fd)
    import copy

    for batch in tqdm(data):

        images, cond = batch['pixel_values'], batch['segmap']
        segmap = preprocess_input(batch["segmap"])
        images = images.cuda()

        tem = copy.deepcopy(segmap[0])
        segmap[0] = segmap[1]
        segmap[1] = tem

        tem = copy.deepcopy(batch["segmap"]['clr_instance'][0])
        batch["segmap"]['clr_instance'][0] = batch["segmap"]['clr_instance'][1]
        batch["segmap"]['clr_instance'][1] = tem

        #clr_msks = [clr_inst.permute(0, 3, 1, 2) / 255. for clr_inst in batch["segmap"]['clr_instance']]

        #samples , _ = vae(images.type(torch.float16), segmap)

        # quant = vae.encode_latent(images.type(torch.float16))
        # segmap_flip = segmap.flip(3)
        # quant_flip = vae.encode_latent(images.type(torch.float16).flip(3))

        segmap_flip = segmap.flip(3)
        quant , _, _ = vae.encode(images.type(torch.float16))
        quant_flip , _, _ = vae.encode(images.type(torch.float16).flip(3))



        # laten = vae.encode(images.type(torch.float16)).latents
        # r = torch.randn((2,3,135,180))
        # r = vae.decode(r.cuda().type(torch.float16), segmap)

        #
        # for idx, (im, samp, clr) in enumerate(zip(images, samples, clr_msks)):
        #     tv.utils.save_image((im + 1) / 2.0, f"{sav_fd}/gt{cnt+1}_im{idx}.png")
        #     tv.utils.save_image((samp + 1) / 2.0, f"{sav_fd}/out{cnt+1}_im{idx}.png")
        #     #tv.utils.save_image((fl + 1) / 2.0, f"{sav_fd}/flip{cnt+1}_im{idx}.png")
        #     tv.utils.save_image(clr, f"{sav_fd}/segmap{cnt + 1}_im{idx}.png")

        for i, (q, seg, q_f, seg_f, name) in enumerate(zip(quant, segmap, quant_flip, segmap_flip, batch['filename'])):
            q = q.cpu()
            q_f = q_f.cpu()
            data_dic = {"x": [q, q_f], "label": {"segmap": [seg.cpu(), seg_f.cpu()] , "path": name}}
            path = sav_fd + name.split('/')[-2] + "/" + name.split('/')[-1].replace(".png", ".pt")
            if not os.path.isdir(sav_fd + name.split('/')[-2]):
                os.makedirs(sav_fd + name.split('/')[-2])
            torch.save(data_dic, path)

        # loop-breaker
        # if cnt>=2 : break
        # cnt += 1

if __name__ == "__main__":
    main()
