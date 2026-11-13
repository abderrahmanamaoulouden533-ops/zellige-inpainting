import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

def load_model():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")
    return pipe

def restore_image(pipe, image_path, mask_path, prompt="Moroccan zellige pattern"):
    image = Image.open(image_path).resize((512, 512))
    mask = Image.open(mask_path).resize((512, 512)).convert("L")
    
    result = pipe(prompt=prompt, image=image, mask_image=mask, num_inference_steps=30).images[0]
    return result
