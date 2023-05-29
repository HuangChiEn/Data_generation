import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

from diffusers.utils.import_utils import is_xformers_available

from cityscape_ds_alpha import load_data, collate_fn
from vqvae.hf_vqvae import VQModel

import logging
logger = get_logger(__name__, log_level="INFO")

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


def get_dataloader(data_dir, image_size, batch_size, num_workers):
    train_dataset = load_data(
        data_dir,
        resize_size=image_size,
        subset_type='train'
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size
        num_workers=num_workers
    )

## TODO : implement resume mechanism..
def resume_mechanism():
    raise NotImplementedError


## TODO : enable logging mechanism
def main(cfger):
    ## Setup accelerator..
    accelerator_project_config = ProjectConfiguration(total_limit=cfger.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfger.gradient_accumulation_steps,
        mixed_precision=cfger.mixed_precision,
        log_with=cfger.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    ## Setup model
    vqvae = VQModel(**cfger.model)

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

    optimizer = optimizer_cls(
        vqvae.parameters(),
        lr=cfger.learning_rate,
        betas=(cfger.adam_beta1, cfger.adam_beta2),
        weight_decay=cfger.adam_weight_decay,
        eps=cfger.adam_epsilon,
    )

    ## Setup dataloader 
    train_dataloader = get_dataloader(**cfger.data)
    
    # Prepare everything with our `accelerator`.
    vqvae, optimizer, train_dataloader = accelerator.prepare(
        vqvae, optimizer, train_dataloader
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
    if cfger.max_train_steps is None:
        cfger.max_train_steps = cfger.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    # Afterwards we recalculate our number of training epochs
    cfger.num_train_epochs = math.ceil(cfger.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(cfger))
        tracker_config.pop("validation_prompts")
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
        resume_mechanism()

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

            with accelerator.accumulate(unet):
                # TODO : insert training step content..
                
                # input 素材, 如何串接到 training step ?? 
                # im = batch["pixel_values"].to(weight_dtype)
                # segmap = preprocess_input(batch["segmap"], cfger.segmap_channels)

                ## loss 計算和處理的template..
                '''
                if cfger.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, cfger.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfger.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfger.gradient_accumulation_steps
                '''

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vqvae.parameters(), cfger.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

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
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfger.max_train_steps:
                break

        if accelerator.is_main_process:
            # TODO : insert validation_step..
            if cfger.validation_prompts is not None and epoch % cfger.validation_epochs == 0:
                log_validation(
                    vae,
                    unet,
                    noise_scheduler,
                    cfger,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                
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
    report_to = tensorboard@str
    train_batch_size = 8@int
    gradient_accumulation_steps = 4@int
    max_train_steps = 100000@int
    num_train_epochs = 100@int
    resume_from_checkpoint = False@bool

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
        [model.lossconfig]
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
    from easy_configer.Configer import Configer
    cfger = Configer()
    cfger.cfg_from_str( get_cfg_str() )

    # If passed along, set the training seed now.
    if cfger.seed != -1:
        set_seed(cfger.seed)

    main(cfger)
