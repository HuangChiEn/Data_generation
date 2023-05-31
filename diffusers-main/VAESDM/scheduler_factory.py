from functools import partial
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DPMSolverSinglestepScheduler
from diffusers.pipeline_utils import DiffusionPipeline

def build_proc(sch_cfg=None, _sch=None, **kwargs):
    if kwargs:
        return _sch(**kwargs)

    type_str = str(type(sch_cfg))
    if 'dict' in type_str:
        return _sch.from_config(**sch_cfg)
    return _sch.from_config(sch_cfg)

scheduler_factory = {
    'UniPC' : partial(build_proc, _sch=UniPCMultistepScheduler),
    # DPM family
    'DDPM' : partial(build_proc, _sch=DDPMScheduler),
    'DPMSolver' : partial(build_proc, _sch=DPMSolverMultistepScheduler, algorithm_type='dpmsolver'),
    'DPMSolver++' : partial(build_proc, _sch=DPMSolverMultistepScheduler),
    'DPMSolverSingleStep' : partial(build_proc, _sch=DPMSolverSinglestepScheduler)
}

def scheduler_setup(pipe : DiffusionPipeline = None, scheduler_type : str = 'UniPC', from_config=None, **kwargs):
    if not isinstance(pipe, DiffusionPipeline):
        raise TypeError(f'pipe should be DiffusionPipeline, but given {type(pipe)}\n')

    sch_cfg = from_config if from_config else pipe.scheduler.config    
    pipe.scheduler = scheduler_factory[scheduler_type](**kwargs) if kwargs \
                        else scheduler_factory[scheduler_type](sch_cfg)
    pipe.scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    pipe.scheduler.variance_type="learned"
    return pipe


# unittest of scheduler..
if __name__ == "__main__":
    def ld_mod():   
        noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda").to(torch.float16)
        unet = SDMUNet2DModel.from_pretrained("/data/harry/Data_generation/diffusers-main/examples/VAESDM/LDM-sdm-model/checkpoint-46000", subfolder="unet").to("cuda").to(torch.float16)
        return noise_scheduler, vae, unet

    from Pipline import SDMLDMPipeline
    from diffusers import StableDiffusionPipeline
    import torch

    path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
    
    # change scheduler 
    # customized args : once you customized, customize forever ~ no from_config
    #pipe = scheduler_setup(pipe, 'DPMSolver++', thresholding=True)
    # from_config
    pipe = scheduler_setup(pipe, 'DPMSolverSingleStep')

    pipe = pipe.to("cuda")
    prompt = "a highly realistic photo of green turtle"
    generator = torch.manual_seed(0)
    # only 15 steps are needed for good results => 2-4 seconds on GPU
    image = pipe(prompt, generator=generator, num_inference_steps=15).images[0]
    # save image
    image.save("turtle.png")

    '''
    # load & wrap submodules into pipe-API
    noise_scheduler, vae, unet = ld_mod()
    pipe = SDMLDMPipeline(
        unet=unet,
        vqvae=vae,
        scheduler=noise_scheduler,
        torch_dtype=torch.float16
    )

    # change scheduler 
    pipe = scheduler_setup(pipe, 'DPMSolverSingleStep')
    pipe = pipe.to("cuda")
    '''
