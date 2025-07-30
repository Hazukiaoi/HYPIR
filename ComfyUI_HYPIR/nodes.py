import torch
from .HYPIR.enhancer.sd2 import SD2Enhancer
import random
from accelerate.utils import set_seed

class HYPIRModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "runwayml/stable-diffusion-v1-5"}),
                "hypir_model_path": ("STRING", {"default": "path/to/HYPIR_sd2.pth"}),
                "lora_rank": ("INT", {"default": 256, "min": 1, "max": 1024}),
                "model_t": ("INT", {"default": 200, "min": 1, "max": 1000}),
                "coeff_t": ("INT", {"default": 200, "min": 1, "max": 1000}),
                "device": (["cuda", "cpu"],),
            }
        }

    RETURN_TYPES = ("HYPIR_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HYPIR"

    def load_model(self, base_model_path, hypir_model_path, lora_rank, model_t, coeff_t, device):
        lora_modules = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
            "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2",
            "ff.net.0.proj",
        ]

        # Always load to CPU first
        model = SD2Enhancer(
            base_model_path=base_model_path,
            weight_path=hypir_model_path,
            lora_modules=lora_modules,
            lora_rank=lora_rank,
            model_t=model_t,
            coeff_t=coeff_t,
            device="cpu",
        )
        model.init_models()

        # Store the target device for sampling
        model.target_device = device

        return (model,)

class HYPIRUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hypir_model": ("HYPIR_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "a high-quality photo"}),
                "upscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
                "unload_to_cpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "HYPIR"

    def upscale(self, hypir_model, image, prompt, upscale, seed, unload_to_cpu):
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        set_seed(seed)

        target_device = hypir_model.target_device

        # Move models to target device
        hypir_model.G.to(target_device)
        hypir_model.text_encoder.to(target_device)
        hypir_model.vae.to(target_device)
        hypir_model.device = target_device

        # from comfy format (B, H, W, C) to model format (B, C, H, W)
        image = image.permute(0, 3, 1, 2)

        enhanced_image = hypir_model.enhance(
            lq=image,
            prompt=prompt,
            upscale=upscale,
            return_type="pt",
        )

        # from model format (B, C, H, W) to comfy format (B, H, W, C)
        enhanced_image = enhanced_image.permute(0, 2, 3, 1)

        if unload_to_cpu:
            hypir_model.G.to("cpu")
            hypir_model.text_encoder.to("cpu")
            hypir_model.vae.to("cpu")
            hypir_model.device = "cpu"

        return (enhanced_image,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "HYPIRModelLoader": HYPIRModelLoader,
    "HYPIRUpscaler": HYPIRUpscaler
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYPIRModelLoader": "HYPIR Model Loader",
    "HYPIRUpscaler": "HYPIR Upscaler"
}
