from functools import partial
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DPMSolverSinglestepScheduler
from diffusers.pipeline_utils import DiffusionPipeline
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from typing import List, Optional, Tuple, Union
import numpy as np
from diffusers.schedulers.scheduling_utils import SchedulerOutput
import diffusers

class DPMSolverMultistepScheduler_variant(DPMSolverMultistepScheduler): 
    '''
    This Edit original  DPMSolverMultistepScheduler class adding a variant of the scheduler
    '''
    
    def __init__(self, variant=None, variant_type=None, **kwargs): 
        super().__init__()
        self._variant = variant
        self._variant_type = variant_type



    def _get_variance(self, predicted_variance: Union[torch.FloatTensor, np.ndarray], timestep: float) -> torch.FloatTensor:
        """
        Compute the variance at the previous timestep based on the predicted variance.

        Args:
            predicted_variance (Union[torch.FloatTensor, np.ndarray]): Variance predicted by the model.
            timestep (float): Current discrete timestep in the diffusion chain.

        Returns:
            torch.FloatTensor: Variance at the previous timestep.
        """
        step_index = self.index_for_timestep(timestep)
        if self.state_in_first_order:
            variance = predicted_variance[step_index - 1]
        else:
            variance = predicted_variance[step_index]

        return variance

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: Union[float, torch.FloatTensor],
        sample: Union[torch.FloatTensor, np.ndarray],
        predicted_variance: Union[torch.FloatTensor, np.ndarray],
        return_dict: bool = True,
        s_noise: float = 1.0,
    ):
        """
        Args:
            model_output (Union[torch.FloatTensor, np.ndarray]): Direct output from learned diffusion model.
            timestep (Union[float, torch.FloatTensor]): Current discrete timestep in the diffusion chain.
            sample (Union[torch.FloatTensor, np.ndarray]): Current instance of sample being created by diffusion process.
            predicted_variance (Union[torch.FloatTensor, np.ndarray]): Variance predicted by the model.
            return_dict (bool, optional): Option for returning tuple rather than SchedulerOutput class. Defaults to True.
            s_noise (float, optional): Scaling factor for the noise added to the sample. Defaults to 1.0.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        step_index = self.index_for_timestep(timestep)

        # Advance index counter by 1
        timestep_int = timestep.cpu().item() if torch.is_tensor(timestep) else timestep
        self._index_counter[timestep_int] += 1

        # Create a noise sampler if it hasn't been created yet
        if self.noise_sampler is None:
            min_sigma, max_sigma = self.sigmas[self.sigmas > 0].min(), self.sigmas.max()
            self.noise_sampler = self.BrownianTreeNoiseSampler(sample, min_sigma, max_sigma, self.noise_sampler_seed)

        # Define functions to compute sigma and t from each other
        def sigma_fn(_t: torch.FloatTensor) -> torch.FloatTensor:
            return _t.neg().exp()

        def t_fn(_sigma: torch.FloatTensor) -> torch.FloatTensor:
            return _sigma.log().neg()

        if self.state_in_first_order:
            sigma = self.sigmas[step_index]
            sigma_next = self.sigmas[step_index + 1]
        else:
            # 2nd order
            sigma = self.sigmas[step_index - 1]
            sigma_next = self.sigmas[step_index]

        # Set the midpoint and step size for the current step
        midpoint_ratio = 0.5
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        delta_time = t_next - t
        t_proposed = t + delta_time * midpoint_ratio

        # 1. Compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            sigma_input = sigma if self.state_in_first_order else sigma_fn(t_proposed)
            pred_original_sample = sample - sigma_input * model_output
        elif self.config.prediction_type == "v_prediction":
            sigma_input = sigma if self.state_in_first_order else sigma_fn(t_proposed)
            pred_original_sample = model_output * (-sigma_input / (sigma_input ** 2 + 1) ** 0.5) + (
                sample / (sigma_input ** 2 + 1)
            )
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        if sigma_next == 0:
            derivative = (sample - pred_original_sample) / sigma
            dt = sigma_next - sigma
            prev_sample = sample + derivative * dt
        else:
            if self.state_in_first_order:
                t_next = t_proposed
            else:
                sample = self.sample

            sigma_from = sigma_fn(t)
            sigma_to = sigma_fn(t_next)
            sigma_up = min(sigma_to, (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
            sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
            ancestral_t = t_fn(sigma_down)
            prev_sample = (sigma_fn(ancestral_t) / sigma_fn(t)) * sample - (
                t - ancestral_t
            ).expm1() * pred_original_sample
            prev_sample = prev_sample + self.noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * sigma_up

            if self.state_in_first_order:
                # Store for 2nd order step
                self.sample = sample
                self.mid_point_sigma = sigma_fn(t_next)
            else:
                # Free for "first order mode"
                self.sample = None
                self.mid_point_sigma = None

        # Compute variance at previous timestep
        variance = self._get_variance(predicted_variance, timestep)

        if not return_dict:
            return (prev_sample, variance)

        #return SchedulerOutput(prev_sample=prev_sample, variance=variance)

    # ... Rest of the class implementation ...

def build_proc(sch_cfg=None, _sch=None, **kwargs):
    if kwargs:
        return _sch(**kwargs)

    type_str = str(type(sch_cfg))
    if 'dict' in type_str:
        return _sch.from_config(**sch_cfg)
    return _sch.from_config(sch_cfg, subfolder="scheduler")

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
    #sch_cfg = diffusers.configuration_utils.FrozenDict({**sch_cfg, 'solver_order':3})  
    #pipe.scheduler = scheduler_factory[scheduler_type](**kwargs) if kwargs \
    #                    else scheduler_factory[scheduler_type](sch_cfg)
    
    pipe.scheduler = DPMSolverSinglestepScheduler()
    #pipe.scheduler = DDPMScheduler(beta_schedule="linear", variance_type="learned_range")
    print(pipe.scheduler)
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
