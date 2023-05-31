from diffusers import AutoencoderKL
from vqvae.hf_vqvae import VQModel 
import torch
#from guided_diffusion.image_datasets import load_data
from cityscape_ds_alpha import load_data, collate_fn
import os
from tqdm import tqdm, trange
#from guided_diffusion import dist_util
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

    return cond['y'].cuda().half()

def main(use_fp16=True,data_dir='/data1/dataset/Cityscapes',batch_size=1,image_size=540, mask_emb="resize"):
    #dist_util.setup_dist()
    #vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = VQModel.from_pretrained("/data/joseph/Data_generation/diffusers-main/examples/VAESDM/Test/checkpoint-1000", subfolder="vqvae")
    if use_fp16:
        vae.to(dtype=torch.float16)
    vae.requires_grad_(False)
    vae = vae.cuda()
    vae.eval()

    dataset = load_data(
        data_dir,
        resize_size=image_size,
        subset_type='val',
        #ret_dataset=False,
        #data_ld_kwargs={'batch_size':batch_size}
    )
    data = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    '''
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
    '''

    catch_patch = "/data/harry/Cityscape_catch/VQVAE_540_resize/train/"
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
    cnt = 0
    sav_fd = 'vqvae_res'
    for batch in tqdm(data):
        images, cond = batch['pixel_values'], batch['segmap']
        segmap = preprocess_input(batch["segmap"])
        
        images = images.cuda()
        
        quant, diff, _ = vae.encode(images.type(torch.float16))
        samples = vae.decode(quant, segmap)

        for idx, (im, samp) in enumerate(zip(images, samples)):
            tv.utils.save_image((im + 1) / 2.0, f"{sav_fd}/gt{cnt+1}_im{idx}.png")
            tv.utils.save_image((samp + 1) / 2.0, f"{sav_fd}/out{cnt+1}_im{idx}.png")
            
        # loop-breaker
        if cnt>=2 : break
        cnt += 1

if __name__ == "__main__":
    main()
