import torch
from Pipline import LDMPipeline, SDMLDMPipeline, SDMPipeline
from diffusers import AutoencoderKL, DDPMScheduler
from unet_2d_sdm import SDMUNet2DModel
from Cityscapes import load_data
pipe = LDMPipeline.from_pretrained("/data/harry/Data_generation/diffusers-main/examples/VAESDM/LDM-uncondition-model", torch_dtype = torch.float16)
pipe = pipe.to("cuda")

images = pipe().images

for i, image in enumerate(images):
    image.save(f"generation_image{i}.png")

def preprocess_input(data, num_classes):
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

def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_().to(t.device)
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()

train_dataset = load_data(
    dataset_mode="cityscapes",
    data_dir="/data1/dataset/Cityscapes",
    image_size=512,
    random_crop=False,
    random_flip=False,
    is_train=True,
    use_vae=True,
    mask_emb="resize"
)


def collate_fn(examples):
    segmap = {}
    for k in examples[0]["label"].keys():
        if k != "path":
            segmap[k] = torch.stack([torch.from_numpy(example["label"][k]) for example in examples])
            segmap[k] = segmap[k].to(memory_format=torch.contiguous_format).float()

    pixel_values = torch.stack([torch.from_numpy(example["pixel_values"]) for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {"pixel_values": pixel_values, "segmap": segmap}


# DataLoaders creation:
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=8,
    num_workers=1,
)


noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda").to(torch.float16)
unet = SDMUNet2DModel.from_pretrained("/data/harry/Data_generation/diffusers-main/examples/VAESDM/LDM-sdm-model/checkpoint-46000", subfolder="unet").to("cuda").to(torch.float16)

pipe = SDMLDMPipeline(
    unet=unet,
    vqvae=vae,
    scheduler=noise_scheduler,
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")
images = None
for step, batch in enumerate(train_dataloader):
    segmap = preprocess_input(batch["segmap"], 34)
    images = pipe(segmap=segmap.to("cuda").to(torch.float16)).images
    break

for i, image in enumerate(images):
    image.save(f"generation_image{i}.png")