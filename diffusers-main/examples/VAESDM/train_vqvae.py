import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

from diffusers.utils.import_utils import is_xformers_available

from cityscape_ds_alpha import load_data, collate_fn
from vqvae.hf_vqvae import VQModel
from vqvae.vqperceptual import VQLPIPSWithDiscriminator

from easy_configer.IO_Converter import IO_Converter

import math
import logging
logger = get_logger(__name__, log_level="INFO")

## disabled currently..
## TODO : chk that is the model need this to override this method ?
def enable_xformers(model):
    if is_xformers_available():
        import xformers
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        model.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

def get_dataloader(data_dir, image_size):
    all_ds = load_data(
        data_dir,
        resize_size=image_size,
        subset_type='all',
        ret_dataset=True
    )
    return all_ds['train'], all_ds['val']

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


## TODO : enable logging mechanism
def main(cfger):
    ## Setup accelerator..
    logging_dir = os.path.join(cfger.output_dir, cfger.logging_dir)
    total_limit = cfger.checkpoints_total_limit if cfger.checkpoints_total_limit!=-1 else None
    accelerator = Accelerator(
        gradient_accumulation_steps=cfger.gradient_accumulation_steps,
        mixed_precision=cfger.mixed_precision,
        project_config=ProjectConfiguration(total_limit=total_limit),
        log_with=cfger.report_to,
        logging_dir=logging_dir,
    )

    ## Setup model
    vqvae = VQModel(**cfger.model)
    disc_mod = VQLPIPSWithDiscriminator(**cfger.loss)

    # accelerator plugins ~ enjoy the community
    if cfger.enable_xformers_memory_efficient_attention:
        enable_xformers(vqvae)

    if cfger.gradient_checkpointing:
        vqvae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfger.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    ## Setup optimizer
    if cfger.scale_lr:
        cfger.learning_rate = (
            cfger.learning_rate * cfger.gradient_accumulation_steps * cfger.train_batch_size * accelerator.num_processes
        )
    if cfger.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    vqvae_optim = optimizer_cls(
        vqvae.parameters(),
        **cfger.optimizer
    )
    # get the trainable params from disc_mod
    disc_optim = optimizer_cls(
        disc_mod.discriminator.parameters(),
        **cfger.optimizer
    )

    ## Setup dataloader 
    train_dataset, val_dataset = get_dataloader(
        cfger.data['data_dir'], 
        cfger.data['image_size']
    )
    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=collate_fn, 
        batch_size=cfger.data['batch_size'], 
        num_workers=cfger.data['num_workers']
    )
    val_dataloader = DataLoader(
        val_dataset, 
        collate_fn=collate_fn, 
        batch_size=cfger.data['batch_size'], 
        num_workers=cfger.data['num_workers']
    )
    
    # Prepare everything with our `accelerator`.
    vqvae, vqvae_optim, disc_mod, disc_optim, train_dataloader = accelerator.prepare(
        vqvae, vqvae_optim, disc_mod, disc_optim, train_dataloader
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # Move vae to gpu and cast to weight_dtype
    vqvae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfger.gradient_accumulation_steps)
    if cfger.max_train_steps == -1:
        cfger.max_train_steps = cfger.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    # Afterwards we recalculate our number of training epochs
    cfger.num_train_epochs = math.ceil(cfger.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator can not record dict obj or even list obj, so we disable it!
        tracker_config = {}
        accelerator.init_trackers(cfger.tracker_project_name, tracker_config)

#-----------------------------------------------------------------------------------------
    ## Training-loop begin..
    total_batch_size = cfger.train_batch_size * accelerator.num_processes * cfger.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfger.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfger.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfger.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfger.max_train_steps}")
    
    global_step, first_epoch = 0, 0
    if cfger.resume_from_checkpoint:
        if cfger.resume_from_checkpoint != "latest":
            path = os.path.basename(cfger.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfger.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfger.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfger.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfger.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * cfger.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * cfger.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, cfger.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, cfger.num_train_epochs):
        vqvae.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if cfger.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % cfger.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(vqvae):
                imgs = batch["pixel_values"].to(weight_dtype)
                segmap = preprocess_input(batch["segmap"], cfger.segmap_channels)
                xrec, qloss = vqvae(imgs, segmap)

                # Train VQ-VAE, opt_idx == 0
                vqvae.zero_grad()                   
                aeloss, log_dict_ae = discr_mod(qloss, imgs, xrec, 0, global_step,
                                                last_layer=vqvae.get_last_layer(), split="train")

                accelerator.log({"train/aeloss": aeloss}, step=global_step)
                accelerator.log({"log_dict_ae": log_dict_ae}, step=global_step)
                accelerator.backward(aeloss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vqvae.parameters(), cfger.max_grad_norm)
                vqvae_optim.step()
                
                # Train discriminator, opt_idx == 1
                discloss, log_dict_disc = discr_mod(qloss, imgs, xrec.detach(), 1, global_step,
                                                last_layer=self.get_last_layer(), split="train")
                
                accelerator.log({"train/discloss": discloss}, step=global_step)
                accelerator.log({"log_dict_ae": log_dict_disc}, step=global_step)
                accelerator.backward(discloss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discr_mod.discriminator.parameters(), cfger.max_grad_norm)
                disc_optim.step()

                ## loss 計算和處理的template..
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(aeloss.repeat(cfger.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfger.gradient_accumulation_steps

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfger.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(cfger.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            logs = {"ae_step_loss": aeloss.detach().item()}   # , "lr": lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(**logs)

            if global_step >= cfger.max_train_steps:
                break

        if accelerator.is_main_process:
            if cfger.validation_step and epoch % cfger.validation_epochs == 0:
                # disable validation currently..
                logger.info("Running validation... with one step!!")
                batch = next(val_dataloader)
                imgs = batch["pixel_values"].to(weight_dtype)
                segmap = preprocess_input(batch["segmap"], cfger.segmap_channels)
                breakpoint()
                xrec, qloss = vqvae(imgs, segmap)
                aeloss, log_dict_ae = discr_mod(qloss, imgs, xrec, 0, global_step,
                                                    last_layer=vqvae.get_last_layer(), split="val")

                discloss, log_dict_disc = discr_mod(qloss, imgs, xrec, 1, global_step,
                                                    last_layer=vqvae.get_last_layer(), split="val")
                rec_loss = log_dict_ae["val/rec_loss"]
                accelerator.log({"val/rec_loss": rec_loss}, step=global_step)
                accelerator.log({"val/aeloss": aeloss}, step=global_step)
                accelerator.log({"val/log_dict_ae": log_dict_ae}, step=global_step)
                accelerator.log({"val/log_dict_disc": log_dict_disc}, step=global_step)


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vqvae = accelerator.unwrap_model(vqvae)
        ## TODO : build pipeline or directly save it ?? how to save it ?
        '''
        pipeline = VQVAEPipeline(
            vqvae=vqvae,
            torch_dtype=weight_dtype
        )
        pipeline.save_pretrained(cfger.output_dir)
        '''
    
    # end of record for report_to
    accelerator.end_training()


## TODO: cp & past cfg from https://raw.githubusercontent.com/CompVis/taming-transformers/master/configs/imagenet_vqgan.yaml
def get_cfg_str():
    return '''
    seed = 42@int

    # efficient tricks 
        use_8bit_adam = False@bool
        enable_xformers_memory_efficient_attention = False@bool
        gradient_checkpointing = False@bool
        allow_tf32 = False@bool
        scale_lr = False@bool

    # acceleratot setup
        mixed_precision = fp16@str
        gradient_accumulation_steps = 1@int
        checkpoints_total_limit = -1@int
        report_to = tensorboard@str
        logging_dir = logs@str
        output_dir = Test@str
        tracker_project_name = VQVAE_train@str

    # train-loop
        train_batch_size = 12@int
        max_train_steps = -1@int
        num_train_epochs = 1000@int
        resume_from_checkpoint = False@bool
        validation_step = False@bool
        segmap_channels = 34@int

    [model]  
        n_embed = 1024@int
        embed_dim = 256@int
        [model.ddconfig]
            double_z = False@bool
            z_channels = 256@int
            resolution = 256@int
            in_channels = 3@int
            out_ch = 3@int
            ch = 128@int
            ch_mult = [1, 1, 2, 2, 4]@list  # num_down = len(ch_mult)-1
            num_res_blocks = 2@int
            attn_resolutions = [16]@list
            dropout = 0.0@float
            segmap_channels = $segmap_channels
            use_SPADE = True@bool
    
    [optimizer]
        lr = 4.5e-6@float
        # adam beta1 and beta2..
        betas = [0.9, 0.999]@list
        weight_decay = 1e-2@float
        eps = 1e-08@float

    [loss]
        disc_conditional = False@bool
        disc_in_channels = 3@int
        disc_start = 250001@int
        disc_weight = 0.8@float
        codebook_weight = 1.0@float
    
    [data]
        data_dir = /data1/dataset/Cityscapes@str
        image_size = 270@int
        batch_size = $train_batch_size
        num_workers = 1@int
    '''


if __name__ == "__main__":
    import os
    from easy_configer.Configer import Configer
    cfger = Configer(cmd_args=True)
    cfger.cfg_from_str(get_cfg_str())
    
    # If passed along, set the training seed now.
    if cfger.seed != -1:
        set_seed(cfger.seed)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != cfger.local_rank:
        cfger.local_rank = env_local_rank

    # Sanity checks
    if cfger.data['data_dir'] is None:
        raise ValueError("Need either a dataset name or a training folder.")

    main(cfger)
