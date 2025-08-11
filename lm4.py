'''
LICENSE
Introduction:
Users are those who use it, others are developers.
You may use this code under all conditions, at your own risk, provided that:
1) ALL weights used are always freely downloadable and without limitations; 
    the model you use with this code cannot be sold.
2) There are no limitations on the use of the output, which belongs to the user, 
    and they can sell it without any compensation to the model manager, except for retraining, 
    to which these rules apply.
3) Content creators may charge for hardware rental, not for use of the model, which remains free.
4) The code is licensed under the AGPL.
5) Users who wish to train or retrain a new model using the output must comply with this license.
6) No NSFW restrictions for adult users who can generate any type of content, without censorship of any kind.
7) Content exclusively for minors: rules and checks on output must be implemented, which do not apply to adults.
8) This code cannot be used by anyone who wants to limit or censor AI 
    in any form or who uses bias to block responses.
8) It cannot be used to generate human deformities intended as NSFW countermeasures.
9) It can be used and adapted for the creation of alien, monsters, horror, zoobies, etc.
10) Share alike.
    File verion 1.0
'''

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from compel import Compel, ReturnedEmbeddingsType
from safetensors.torch import load_file
from models import TransparentVAEDecoder
from datasets import disable_caching
import random
import string
import warnings
from lpw_stable_diffusion_xl import SDXLLongPromptWeightingPipeline
from diffusers.loaders import StableDiffusionLoraLoaderMixin
# Sopprimi i warning LoRA
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers.models.lora")

disable_caching()

class CustomSDXLPipeline(SDXLLongPromptWeightingPipeline, StableDiffusionLoraLoaderMixin):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str = None,
        prompt_2: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
    ):
        # [Il resto del metodo __call__ rimane invariato, come nel codice precedente]
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt_embeds is not None and pooled_prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        elif prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Use provided embeddings if available, otherwise generate them
        if prompt_embeds is not None and pooled_prompt_embeds is not None:
            # Ensure negative embeddings are provided or create zero tensors for CFG
            if do_classifier_free_guidance:
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                if negative_pooled_prompt_embeds is None:
                    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        else:
            # Generate embeddings from text prompts
            negative_prompt = negative_prompt if negative_prompt is not None else ""
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.get_weighted_text_embeddings_sdxl(
                pipe=self, prompt=prompt, neg_prompt=negative_prompt
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if (
            denoising_end is not None
            and isinstance(denoising_end, float)
            and denoising_end > 0
            and denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

def generate_random_string():
    length = random.choice([6])
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def clean_memory():
    try:
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Memoria GPU pulita con successo")
    except Exception as e:
        print(f"Errore durante la pulizia della memoria GPU: {e}")

if __name__ == "__main__":
    clean_memory()

    try:
        transparent_vae = TransparentVAEDecoder.from_pretrained(
            "D:/LM/huggingface.co/madebyollin/sdxl-vae-fp16-fix/",
            torch_dtype=torch.float16
        )
        transparent_vae.config.force_upcast = False
        model_path = 'D:/ai/models/layerdiffusion-v1/vae_transparent_decoder.safetensors'
        transparent_vae.set_transparent_decoder(load_file(model_path))

        # Use the custom pipeline
        pipeline = CustomSDXLPipeline.from_pretrained(
            "D:/ai/test/fused-lumiya",   # use sdxl and "fuse lora", not other method to create trasparent png image this model is sdxl + lumiya lora this is not important for test
            vae=transparent_vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False
        ).to("cuda")

        pipeline.load_lora_weights(
            'D:/ai/models/diffuser_layerdiffuse',
            weight_name='diffuser_layer_xl_transparent_attn.safetensors',
            adapter_name="transparent_attn"
        )
        pipeline.set_adapters(["default_0"], adapter_weights=[1])
        pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.enable_model_cpu_offload()

        # Configura Compel
        compel = Compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        os.makedirs("./img4", exist_ok=True)

        # Prompt bilanciato
        prompt = (
            "4lb1n0 G0dd3ss, (albino woman:1.0), white hair, pale skin, seductive pose, photo-realistic, "
            "(fantasy landscape:3.0), majestic red white dragon flying over vibrant forest, starry sky, "
            "ancient ruins, glowing crystals in cliffs, mystical river, heroic knight on hilltop, "
            "(black swan in lake:3.0), cinematic lighting, 8k, transparent_background"
        )
        negative_prompt = "blurry:1.2, low quality:1.2, cartoonish:1.0, unrealistic:1.0, extra limbs:1.0, deformed:1.0"

        # Genera embeddings con Compel
        conditioning, pooled = compel(prompt)
        negative_conditioning, negative_pooled = compel(negative_prompt)

        # Verifica il numero di token (opzionale, per debug)
        tokenizer = pipeline.tokenizer
        tokens = tokenizer(prompt, return_tensors="pt", truncation=False, padding=True)
        print(f"Numero di token: {len(tokens['input_ids'][0])}")
        print(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))

        # Verifica le forme degli embedding
        print(f"prompt_embeds shape: {conditioning.shape}")
        print(f"pooled_prompt_embeds shape: {pooled.shape}")
        print(f"negative_prompt_embeds shape: {negative_conditioning.shape}")
        print(f"negative_pooled_prompt_embeds shape: {negative_pooled.shape}")

        # Genera le immagini usando gli embeddings
        seed = torch.randint(high=1000000, size=(1,)).item()
        images = pipeline(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_conditioning,
            negative_pooled_prompt_embeds=negative_pooled,
            num_inference_steps=20,
            guidance_scale=7,
            height=768,
            width=768,
            generator=torch.Generator(device='cuda').manual_seed(seed),
            num_images_per_prompt=1
        ).images

        for i, image in enumerate(images):
            random_string = generate_random_string()
            image.save(f"img4/img{i:03d}_{random_string}_{seed}.png")

    finally:
        clean_memory()