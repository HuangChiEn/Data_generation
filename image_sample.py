"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import torch as th
import torch.distributed as dist
import torchvision as tv

from guided_diffusion.image_datasets import load_data

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    print("start")
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion, vae = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    if vae is not None:
        vae.to(dist_util.dev())
        vae.eval()

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=False,
        use_vae=args.use_vae,
        mask_emb=args.mask_emb
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)

    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        image = ((batch + 1.0) / 2.0).cuda()
        label = (cond['label_ori'].float() / 255.0).cuda()
        model_kwargs = preprocess_input(cond, num_classes=args.num_classes)
        model_kwargs['y'] = model_kwargs['y'].cuda()
        if args.mask_emb == "vae_encode":
            label_latent = []
            for i in range(model_kwargs['y'].shape[1]):
                latent = vae.encode(
                    model_kwargs['y'][:, i, :, :].unsqueeze(1).repeat(1, 3, 1, 1).type(th.float16)).latent_dist.sample()
                label_latent.append(latent * 0.18215)
            model_kwargs['y'] = th.cat(label_latent, dim=1)

        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (args.batch_size, 3, image.shape[2], image.shape[3]) if not args.use_vae else (args.batch_size, 3, 135, 180),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            device= dist_util.dev()
        )


        if vae is not None:
            sample /= 0.18215
            sample = vae.decode(sample.type(th.float16), model_kwargs['y'].type(th.float16))#.sample

        sample = (sample + 1) / 2.0

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(sample.shape[0]):
            tv.utils.save_image(image[j], os.path.join(image_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(sample[j], os.path.join(sample_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(label[j], os.path.join(label_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

        if len(all_samples) * args.batch_size > args.num_samples:
            break

    dist.barrier()
    logger.log("sampling complete")


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        s=1.0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
