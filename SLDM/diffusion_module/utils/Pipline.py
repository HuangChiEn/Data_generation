import inspect
from typing import List, Optional, Tuple, Union

import torch

from diffusers.models import UNet2DModel, VQModel
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import randn_tensor
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
import copy

class LDMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vae ([`VQModel`]):
            Vector-quantized (VQ) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            [`DDIMScheduler`] is to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, vae: VQModel, unet: UNet2DModel, scheduler: DDIMScheduler, torch_dtype=torch.float16):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
        self.torch_dtype = torch_dtype

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 8,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.model.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        if not isinstance(self.unet.config.sample_size,tuple):
            self.unet.config.sample_size = (self.unet.config.sample_size,self.unet.config.sample_size)

        latents = randn_tensor(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size[0], self.unet.config.sample_size[1]),
            generator=generator,
        )
        latents = latents.to(self.device).type(self.torch_dtype)

        # scale the initial noise by the standard deviation required by the scheduler (need to check)
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, t).sample
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs).prev_sample

        # decode the image latents with the VAE
        latents /= self.vae.config.scaling_factor#(0.18215)
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class SDMLDMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vae ([`VQModel`]):
            Vector-quantized (VQ) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            [`DDIMScheduler`] is to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, vae: VQModel, unet: UNet2DModel, scheduler: DDIMScheduler, torch_dtype=torch.float16):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
        self.torch_dtype = torch_dtype

    @torch.no_grad()
    def __call__(
        self,
        segmap = None,
        batch_size: int = 8,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        every_step_save: int = None,
        s: int = 1,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.model.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        self.unet.config.sample_size = (64, 128) # (135,180)
        if not isinstance(self.unet.config.sample_size, tuple):
            self.unet.config.sample_size = (self.unet.config.sample_size, self.unet.config.sample_size)

        if segmap is None:
            print("Didn't inpute any segmap, use the empty as the input")
            segmap = torch.zeros(batch_size,self.unet.config.segmap_channels, self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        segmap = segmap.to(self.device).type(self.torch_dtype)
        latents = randn_tensor(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size[0], self.unet.config.sample_size[1]),
            generator=generator,
        )
        latents = latents.to(self.device).type(self.torch_dtype)

        # scale the initial noise by the standard deviation required by the scheduler (need to check)
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        step_latent = []
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
    
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, segmap, t).sample
            # compute the previous noisy sample x_t -> x_t-1

            if s > 1.0:
                model_output_zero = self.unet(latent_model_input, torch.zeros_like(segmap), t).sample
                noise_prediction[:, :3] = model_output_zero[:, :3] + s * (noise_prediction[:, :3] - model_output_zero[:, :3])

            # when apply different scheduler, mean only !!
            #latents = self.scheduler.step(noise_prediction[:, :3], t, latents, **extra_kwargs).prev_sample

            latents = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs).prev_sample

            if every_step_save is not None:
                if (i+1) % every_step_save == 0:
                    step_latent.append(copy.deepcopy(latents))

        # decode the image latents with the VAE
        if every_step_save is not None:
            image = []
            for i, l in enumerate(step_latent):
                l /= self.vae.config.scaling_factor  # (0.18215)
                #latents /= 7.706491063029163
                l = self.vae.decode(l, segmap)
                l = (l / 2 + 0.5).clamp(0, 1)
                l = l.cpu().permute(0, 2, 3, 1).numpy()
                if output_type == "pil":
                    l = self.numpy_to_pil(l)
                image.append(l)
        else:
            latents /= self.vae.config.scaling_factor#(0.18215)
            #latents /= 7.706491063029163
            image = self.vae.decode(latents, segmap).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == "pil":
                image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class SDMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vae ([`VQModel`]):
            Vector-quantized (VQ) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            [`DDIMScheduler`] is to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, unet: UNet2DModel, scheduler: DDIMScheduler, torch_dtype=torch.float16, vae=None):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.torch_dtype = torch_dtype

    @torch.no_grad()
    def __call__(
        self,
        segmap = None,
        batch_size: int = 8,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        s: int = 1,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.model.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        self.unet.config.sample_size = (270,360)
        if not isinstance(self.unet.config.sample_size, tuple):
            self.unet.config.sample_size = (self.unet.config.sample_size, self.unet.config.sample_size)

        if segmap is None:
            print("Didn't inpute any segmap, use the empty as the input")
            segmap = torch.zeros(batch_size,self.unet.config.segmap_channels, self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        segmap = segmap.to(self.device).type(self.torch_dtype)
        latents = randn_tensor(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size[0], self.unet.config.sample_size[1]),
            generator=generator,
        )

        latents = latents.to(self.device).type(self.torch_dtype)

        # scale the initial noise by the standard deviation required by the scheduler (need to check)
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, segmap, t).sample

            #noise_prediction = noise_prediction[]

            if s > 1.0:
                model_output_zero = self.unet(latent_model_input, torch.zeros_like(segmap), t).sample
                noise_prediction[:, :3] = model_output_zero[:, :3] + s * (noise_prediction[:, :3] - model_output_zero[:, :3])

            #noise_prediction = noise_prediction[:, :3]

            # compute the previous noisy sample x_t -> x_t-1
            #breakpoint()
            latents = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs).prev_sample

        # decode the image latents with the VAE
        # latents /= self.vae.config.scaling_factor#(0.18215)
        # image = self.vae.decode(latents).sample
        image = latents
        #image = (image + 1) / 2.0
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

