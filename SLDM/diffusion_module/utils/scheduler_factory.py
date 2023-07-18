from functools import partial
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DPMSolverSinglestepScheduler
from diffusers.pipeline_utils import DiffusionPipeline
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from typing import List, Optional, Tuple, Union
import numpy as np
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils import randn_tensor, BaseOutput


### Testing the DDPM Scheduler for Variant 
class ModifiedDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDPMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            print("Conidtion is trigger")
       
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
            # [2,3, 64, 128]
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise

            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise
        
        pred_prev_sample = pred_prev_sample + variance
        print(pred_prev_sample.shape)
        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
   

class ModifiedUniPCScheduler(UniPCMultistepScheduler):
    '''
    This is the modification of UniPCMultistepScheduler, which is the same as UniPCMultistepScheduler except for the _get_variance function.
    '''
    def __init__(self, variance_type: str = "fixed_small", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_timesteps = False
        self.variance_type=variance_type
        self.config.timestep_spacing="leading"
    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
            )
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps

        return prev_t
    
    def _get_variance(self, t, predicted_variance=None, variance_type="learned_range"):
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        if variance_type is None:
            variance_type = self.config.variance_type

        if variance_type == "fixed_small":
            variance = variance
        elif variance_type == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, return_dict: bool = True) -> Union[SchedulerOutput, Tuple]:
        
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            print("condition using predicted_variance is trigger")
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        super_output = super().step(model_output, timestep, sample, return_dict=False)
        prev_sample = super_output[0]
        # breakpoint()
        variance = 0
        if timestep > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=None, device=device, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(timestep, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                # breakpoint()
                variance = self._get_variance(timestep, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
                # breakpoint()
            else:
                variance = (self._get_variance(timestep, predicted_variance=predicted_variance) ** 0.5) * variance_noise

      
        # breakpoint()
        print("time step is ", timestep)
        prev_sample = prev_sample  + variance

        if not return_dict:
            return (prev_sample,)
        
        return DDPMSchedulerOutput(prev_sample=prev_sample,pred_original_sample=prev_sample) 

        #return SchedulerOutput(prev_sample=prev_sample)


def build_proc(sch_cfg=None, _sch=None, **kwargs):
    if kwargs:
        return _sch(**kwargs)

    type_str = str(type(sch_cfg))
    if 'dict' in type_str:
        return _sch.from_config(**sch_cfg)
    return _sch.from_config(sch_cfg, subfolder="scheduler")

scheduler_factory = {
    'UniPC' : partial(build_proc, _sch=UniPCMultistepScheduler),
    'modifiedUniPC' : partial(build_proc, _sch=ModifiedUniPCScheduler),
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
    #sch_cfg = diffusers.configuration_utils.FrozenDict({**sch_cfg, 'solver_order':3})  
    #pipe.scheduler = scheduler_factory[scheduler_type](**kwargs) if kwargs \
    #                    else scheduler_factory[scheduler_type](sch_cfg)
    
    # pipe.scheduler = DPMSolverSinglestepScheduler()
    # #pipe.scheduler = DDPMScheduler(beta_schedule="linear", variance_type="learned_range")
    # print(pipe.scheduler)
    print("Scheduler type in Scheduler_factory.py is Hard-coded to modifyUniPC, Please change it back to AutoDetect functionality if you want to change scheudler")
    pipe.scheduler = ModifiedUniPCScheduler(variance_type="learned_range", )
    # pipe.scheduler = ModifiedDDPMScheduler(beta_schedule="linear", variance_type="learned_range")
    
    #pipe.scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    #pipe.scheduler._get_variance = _get_variance
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
