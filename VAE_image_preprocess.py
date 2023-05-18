from diffusers import AutoencoderKL
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

def main(use_fp16=True,data_dir='/data1/dataset/Cityscapes',batch_size=16,image_size=540, mask_emb="resize"):
    dist_util.setup_dist()
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
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
        is_train=True,
        use_vae=True,
        mask_emb=mask_emb
    )

    catch_patch = "/data/harry/Cityscape_catch/VAE_540_label_encode/train/"
    if not os.path.isdir(catch_patch):
        os.makedirs(catch_patch)

    initial_count = 0
    for path in os.listdir(catch_patch):
        if os.path.isfile(os.path.join(catch_patch, path)):
            initial_count += 1
        else:
            catch_patch_ = os.path.join(catch_patch, path)
            for path_ in os.listdir(catch_patch_):
                if os.path.isfile(os.path.join(catch_patch_, path_)):
                    initial_count += 1
    print(initial_count)

    for images, cond in tqdm(data):
        # if initial_count >= 2975:
        #     break
        images = images.cuda()
        latent_dist = vae.encode(images.type(torch.float16)).latent_dist # .to(torch.float16)
        mean_ = latent_dist.mean
        std_ = latent_dist.std
        sample = torch.randn(mean_.shape).cuda()
        sample = mean_ + std_ * sample


        sample = vae.decode(sample.type(torch.float16)).sample
        tv.utils.save_image((images[0] + 1) / 2.0, "input_img.png")
        tv.utils.save_image((sample[0] + 1) / 2.0, "output_img.png")
        break

        if mask_emb == "vae_encode":
            cond = preprocess_input(cond)
            cond['y'] = cond['y'].cuda()
            label_mean = []
            label_std = []
            for i in range(cond['y'].shape[1]):
                latent_dist = vae.encode(cond['y'][:, i, :, :].unsqueeze(1).repeat(1,3,1,1).type(torch.float16)).latent_dist
                label_mean.append(latent_dist.mean.cpu())
                label_std.append(latent_dist.std.cpu())

        for i, (m, s, name) in enumerate(zip(mean_, std_, cond['path'])):
            path = catch_patch + name.split('/')[-2] + "/" + name.split('/')[-1].replace(".png", ".pt")
            if os.path.isfile(path):
                continue
            else:
                print(path)
                initial_count += 1
                m = m.cpu()
                s = s.cpu()


                #data_dic = {"x": {"mean": m, "std": s}, "label": {k: cond[k][i] for k in cond.keys() }}
                data_dic = {"x": {"mean": m, "std": s}, "label": {"mean": [l_mean[i] for l_mean in label_mean], "std": [l_std[i] for l_std in label_std]}}

                if not os.path.isdir(catch_patch + name.split('/')[-2]):
                    os.makedirs(catch_patch + name.split('/')[-2])
                torch.save(data_dic,path)



if __name__ == "__main__":
    main()
