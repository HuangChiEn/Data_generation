from os import path, makedirs
import torch
from Pipline import LDMPipeline, SDMLDMPipeline, SDMPipeline
from diffusers import AutoencoderKL, DDPMScheduler, VQModel
from model.unet_2d_sdm import SDMUNet2DModel
from Cityscapes import load_data, collate_fn
from scheduler_factory import scheduler_setup

Pipe_dispatcher = {
    'LDMPipeline' : LDMPipeline, 
    'SDMPipeline' : SDMPipeline,
    'SDMLDMPipeline' : SDMLDMPipeline
}

# TODO : this should be placed in cityscape dataset py file
def preprocess_input(data, num_classes):
    # utils to get the edge of image
    def get_edges(t):
        # zero tensor, prepare to fill with edge (1) and bg (0)
        edge = torch.ByteTensor(t.size()).zero_().to(t.device)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = torch.FloatTensor(bs, num_classes, h, w).zero_().to(data['label'].device)
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

    return input_semantics


def get_dataloader(data_dir, image_size, batch_size, num_workers):
    train_dataset = load_data(
        dataset_mode="cityscapes",
        data_dir=data_dir,
        image_size=image_size,
        random_crop=False,
        random_flip=False,
        is_train=False,
        use_vae=False,
        mask_emb="resize"
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )

def get_diffusion_modules(unet_path, numk_ckpt, vae_type=None):
    # Load pretrained unet from local..
    unet_path = path.join(unet_path, f"checkpoint-{numk_ckpt*1000}")
    unet = SDMUNet2DModel.from_pretrained(unet_path, subfolder='unet').to('cuda').to(torch.float16)
    if not vae_type:
        return unet

    # Load hugging face VAE on cuda with fp16..
    if vae_type == 'KL':
        vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to('cuda').to(torch.float16)
    elif vae_type == 'VQ':
        vae = VQModel.from_pretrained('CompVis/ldm-super-resolution-4x-openimages', subfolder='vqvae').to('cuda').to(torch.float16)
    else:
        raise ValueError(f"Unsupport VAE type {vae_type}")
    
    return unet, vae

def get_pipeline(pipe_type, pipe_path=None, unet=None, vae=None):
    pipe_builder = Pipe_dispatcher[pipe_type]
    if pipe_path:
        return pipe_cls.from_pretrained(pipe_path)
    
    if unet == None:
        raise ValueError(f"Plz give the unet model while disable the from_pretrained mode!!")

    return pipe_builder(
        unet=unet,
        vae=vae,
        # pre-load dummy-scheduler into pipe..
        scheduler=DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler"),
        torch_dtype=torch.float16
    )

def get_cfg_str():
    return '''
    seed = 42@int
    num_inference_steps = 20@int
    scheduler_type = UniPC@str

    save_dir = Gen_results@str
    num_save_im = 35@int
    
    [dataloader]
        data_dir = /data1/dataset/Cityscapes@str
        image_size = 512@int
        batch_size = 8@int
        num_workers = 1@int

    [diff_mod]
        unet_path = /data/harry/Data_generation/diffusers-main/examples/VAESDM/VQLDM-sdm540-model@str
        numk_ckpt = 62@int
        vae_type = VQ@str
    
    [pipe]
        pipe_type = SDMLDMPipeline@str
        pipe_path = @str
    '''

if __name__ == "__main__":
    from torchvision.utils import save_image
    from easy_configer.Configer import Configer
    cfger = Configer()
    cfger.cfg_from_str( get_cfg_str() )

    data_ld = get_dataloader(**cfger.dataloader)   
    unet, vae = get_diffusion_modules(**cfger.diff_mod)

    pipe = get_pipeline(**cfger.pipe, unet=unet, vae=vae)
    pipe = scheduler_setup(pipe, cfger.scheduler_type)
    pipe = pipe.to("cuda")

    makedirs(cfger.save_dir, exist_ok=True)
    num_itrs = (cfger.num_save_im // cfger.dataloader['batch_size']) + 1
    
    # Generate the image : 
    # we put it in main block, 
    #  so you only allow to generate image by 'directly' exec this py file! 
    img_lst, fn_lst, clr_msk_lst = [], [], []
    generator = torch.manual_seed(cfger.seed)

    for idx, batch in enumerate(data_ld, 0):
        fn_lst.extend(batch['filename'])
        clr_msks = [ clr_inst.permute(0, 3, 1, 2) / 255. for clr_inst in batch["segmap"]['clr_instance'] ]
        clr_msk_lst.extend(clr_msks)

        segmap = preprocess_input(batch["segmap"], num_classes=34)
        segmap = segmap.to("cuda").to(torch.float16)
        images = pipe(segmap=segmap, generator=generator, num_inference_steps=cfger.num_inference_steps).images
        img_lst.extend(images)

        if idx >= num_itrs:
            break
    
    for idx, (image, clr_msk, fn_w_ext) in enumerate( zip(img_lst, clr_msk_lst, fn_lst) ):
        if idx <= cfger.num_save_im:
            image.save(f"{cfger.save_dir}/gen_{fn_w_ext}")
            save_image(clr_msk, f"{cfger.save_dir}/msk_{fn_w_ext}")
            