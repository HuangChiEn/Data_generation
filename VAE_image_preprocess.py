from diffusers import AutoencoderKL
import torch
from guided_diffusion.image_datasets import load_data
import os
from tqdm import tqdm, trange

def main(use_fp16=True,data_dir='/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic',batch_size=1,image_size=1080):
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
        use_vae=True
    )

    for images, cond in tqdm(data):
        images = images.cuda()

        x = vae.encode(images.type(torch.float16)).latent_dist.sample() # .to(torch.float16)
        print(x.shape)
        #print(cond['path'])
        #print()
        for i, (image, name) in enumerate(zip(x, cond['path'])):
            image = image.cpu()
            data_dic = {"x": image, "label": {k : cond[k][i] for k in cond.keys() }}
            path = "./train1080/train/" + name.split('/')[-2] + "/" + name.split('/')[-1].replace(".png",".pt")
            if not os.path.isdir("./train1080/train/" + name.split('/')[-2]):
                os.makedirs("./train1080/train/" + name.split('/')[-2])
            torch.save(data_dic,path)
        # if(images.shape[0]<16):
        #     break


if __name__ == "__main__":
    main()
