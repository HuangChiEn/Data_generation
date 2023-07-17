from os import path, makedirs
import torch
from diffusion_module.utils.Pipline import LDMPipeline, SDMLDMPipeline, SDMPipeline
from diffusers import AutoencoderKL, DDPMScheduler, VQModel
from diffusion_module.unet_2d_sdm import SDMUNet2DModel
from diffusion_module.unet import UNetModel
from dataset.cityscape_ds import load_data, collate_fn
from diffusion_module.utils.scheduler_factory import scheduler_setup
from taming.models.vqvae import VQSub
import cv2
from PIL import Image
Pipe_dispatcher = {
    'LDMPipeline' : LDMPipeline, 
    'SDMPipeline' : SDMPipeline,
    'SDMLDMPipeline' : SDMLDMPipeline
}


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

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
        b_msk = (data['instance']!=0)
        # binary mask to gpu-device and turn type to float 
        instance_edge_map = b_msk.to(data['instance'].device).float()
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

    return input_semantics

def get_dataloader(data_dir, image_size, batch_size, num_workers, fn_qry, subset_type="train", num_processes=1):
    train_dataset = load_data(
        data_dir,
        resize_size=image_size,
        subset_type=subset_type,
        fn_qry=fn_qry
    )
    num_data_pre = int(len(train_dataset) / num_processes)
    dataset = torch.utils.data.random_split(train_dataset, [num_data_pre for _ in range(num_processes-1)] + [len(train_dataset) - num_data_pre*(num_processes-1)])
    print("split dataset size", [len(d) for d in dataset])
    #train_dataset1,  train_dataset2= torch.utils.data.random_split(train_dataset, [250, 250])
    return [ torch.utils.data.DataLoader(
        d, shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        ) for d in dataset]

def get_diffusion_modules(unet_path, numk_ckpt=0, vae_type=None):
    # Load pretrained unet from local..
    if ".pt" in unet_path:
        unet = UNetModel(
        image_size=(270, 360),
        in_channels=3,
        model_channels=256,
        out_channels=3*2,
        num_res_blocks=2,
        attention_resolutions=(8,16,32),
        dropout=0,
        channel_mult=(1, 1, 2, 4, 4),
        num_heads= 64,
        num_head_channels= -1,
        num_heads_upsample= -1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        num_classes=35,
        use_checkpoint=True,
        )
        unet.load_state_dict(torch.load(unet_path))
        unet = unet.to(torch.float16)
    elif numk_ckpt == 0:
        unet = UNetModel.from_pretrained(unet_path, subfolder='unet').to('cuda').to(torch.float16)
    else:
        unet_path = path.join(unet_path, f"checkpoint-{numk_ckpt*1000}")
        unet = UNetModel.from_pretrained(unet_path, subfolder='unet').to('cuda').to(torch.float16)

    if not vae_type:
        return unet, None

    # Load hugging face VAE on cuda with fp16..
    if vae_type == 'KL':
        vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to('cuda').to(torch.float16)
    elif vae_type == 'VQ':
        #vae = VQModel.from_pretrained('CompVis/ldm-super-resolution-4x-openimages', subfolder='vqvae').to('cuda').to(torch.float16)
        vae = VQSub.from_pretrained('/data/harry/Data_generation/diffusers-main/VQVAE/SPADE_VQ_model_V2/70ep', subfolder='vqvae').to('cuda').to(torch.float16)
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
    seed = 42
    num_inference_steps = 1000
    scheduler_type = 'DDPM'
    save_dir = 'Gen_results3'
    num_save_im = 500
    s = 1.2@float    # 1. equal @float
    
    [dataloader]
        data_dir = '/data1/dataset/Cityscapes'
        image_size = 540
        batch_size = 8
        num_workers = 8
        subset_type = 'val'
        fn_qry = '*/*.png'   # qry-syntax to skip DA_Data [a-z]*/*.png

    [diff_mod]
        unet_path = '/data/harry/Data_generation/diffusers-main/VAESDM/ourVQVAE-SDM-learnvar'
        #unet_path = '/data/harry/Data_generation/OUTPUT/Cityscapes270-SDM-256CH-500epoch/model120000.pt'
        numk_ckpt = 0
        vae_type = 'VQ'

    [pipe]
        pipe_type = 'SDMLDMPipeline'
        pipe_path = ''
    '''

def test_car_image():
    def get_edges(t):
        # zero tensor, prepare to fill with edge (1) and bg (0)
        edge = torch.ByteTensor(t.size()).zero_().to(t.device)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    mask = cv2.imread(r"/data/harry/Data_generation/diffusers-main/car/preproc_car/mask/Fiat$$500$$2013$$White$$28_2$$1361$$image_4.png", 0)
    h, w = mask.shape
    x = 50
    y = 50
    mask_ = torch.zeros(8, 35, 540, 720)
    #mask_[0][26][x:x+h, y:y+w] = torch.tensor(mask)
    #mask_[0][0] = 1-mask_[0][26]
    mask_[:, 7, 270:, :] = 1.0
    mask_[:, 23, :270, :] = 1.0
    return mask_

if __name__ == "__main__":
    from torchvision.utils import save_image
    from easy_configer.Configer import Configer
    from accelerate import PartialState  # Can also be Accelerator or AcceleratorStaet
    distributed_state = PartialState()
    cfger = Configer()
    cfger.cfg_from_str( get_cfg_str() )

    data_lds = get_dataloader(**cfger.dataloader)
    unet, vae = get_diffusion_modules(**cfger.diff_mod)

    #del vae.quantize
    del vae.encoder

    pipe = get_pipeline(**cfger.pipe, unet=unet, vae=vae)
    pipe = scheduler_setup(pipe, cfger.scheduler_type)#, from_config = "CompVis/stable-diffusion-v1-4")
    pipe.to(distributed_state.device)

    # Assume two processes
    data_ld = data_lds[distributed_state.process_index]


    pipe = pipe.to(distributed_state.device)

    makedirs(cfger.save_dir, exist_ok=True)
    makedirs(f"{cfger.save_dir}/mask", exist_ok=True)
    makedirs(f"{cfger.save_dir}/image", exist_ok=True)
    makedirs(f"{cfger.save_dir}/real", exist_ok=True)

    num_itrs = (cfger.num_save_im // cfger.dataloader['batch_size'])
    
    # Generate the image : 
    # we put it in main block, 
    #  so you only allow to generate image by 'directly' exec this py file! 
    img_lst, fn_lst, clr_msk_lst = [], [], []
    generator = torch.manual_seed(cfger.seed)

    for idx, batch in enumerate(data_ld, 0):
        # if idx >= num_itrs:
        #     break
        clr_msks = [ clr_inst.permute(0, 3, 1, 2) / 255. for clr_inst in batch["segmap"]['clr_instance'] ]
        real = batch['pixel_values']
        #real = [ clr_inst.permute(0, 3, 1, 2) / 255. for clr_inst in batch['pixel_values'] ]
        segmap = preprocess_input(batch["segmap"], num_classes=34)
        segmap = segmap.to(distributed_state.device).to(torch.float16)
        images = pipe(segmap=segmap, generator=generator, num_inference_steps=cfger.num_inference_steps, s = cfger.s, batch_size=segmap.shape[0]).images

        for x, image, clr_msk, fn_w_ext in zip(real, images, clr_msks, batch['filename']):
            ...
            #torchvision.utils.save_image(x,f"{cfger.save_dir}/real/{fn_w_ext}")
            # numpy_to_pil(x).save(f"{cfger.save_dir}/real/{fn_w_ext}")
            # print(x.shape)
            # [2, 0, 1]
            #save_image((x+1.)/2., f"{cfger.save_dir}/real/{fn_w_ext}")
            #image.save(f"{cfger.save_dir}/image/{fn_w_ext}")
            #save_image(clr_msk, f"{cfger.save_dir}/mask/{fn_w_ext}")

            
