from vqvae.hf_vqvqe import VQModel
import torch
from guided_diffusion.image_datasets import load_data
import os
from tqdm import tqdm, trange
from guided_diffusion import dist_util
import torchvision as tv


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

    return cond

def main(use_fp16=True,data_dir='/data1/dataset/Cityscapes',batch_size=8,image_size=540, mask_emb="resize"):
    dist_util.setup_dist()
    #vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = VQModel.from_pretrained("CompVis/ldm-super-resolution-4x-openimages", subfolder="vqvae")
    if use_fp16:
        vae.to(dtype=torch.float16)
    vae.requires_grad_(False)
    vae = vae.cuda()
    vae.eval()

    data = load_data(
        dataset_mode="cityscapes",
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=True,
        is_train=False,
        use_vae=True,
        mask_emb=mask_emb
    )

    for images, cond in tqdm(data):
        images = images.cuda()
        latents = vae.encode(images.type(torch.float16)).latents # .to(torch.float16)
        latents = latents * vae.config.scaling_factor

        # mean_ = latent_dist.mean
        # std_ = latent_dist.std
        # sample = torch.randn(mean_.shape).cuda()
        # sample = mean_ + std_ * sample


        sample = vae.decode(latents.type(torch.float16)).sample
        tv.utils.save_image((images[0] + 1) / 2.0, "input_img.png")
        tv.utils.save_image((sample[0] + 1) / 2.0, "output_img.png")
        break

if __name__ == "__main__":
    main()
