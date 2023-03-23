# !pip install opencv-python  accelerate
# pip install transformers
# pip install controlnet-aux==0.0.1
##
import sys 
sys.path.append('./src/')
sys.path.append('/home/ubuntu/controlnet_aux/src')
##
 

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from controlnet_aux import MLSDdetector
from annotator import MLSDdetector, UniformerDetector

from PIL import Image, ImageOps

# download an image
image_ref = load_image(
    "https://can01.anibis.ch/Genf-Maison-Prestige/?1024x768/3/60/anibis/071/478/047/zv2wrFS8JkCernwEpWoMeg_1.jpg"
)

image = np.array(image_ref)

# get hough image
model = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
hough_image = model(image)
H, W, C = np.array(hough_image).shape

# get segmentation
keep_labels = ['windowpane','sky', 'fireplace', 'stairs']
keep_from_line_labels = ['ceiling','wall','floor','door']

uniformer = UniformerDetector.from_pretrained("lllyasviel/ControlNet")
detected_map_seg, label_image = uniformer(image)

classes = uniformer.model.CLASSES
label_image = label_image[0]
keep_idxs = [classes.index(label) for label in keep_labels]
keep_from_line_idxs = [classes.index(label) for label in keep_from_line_labels]

line_mask = np.sum(np.array([label_image == idx for idx in keep_from_line_idxs]),axis=0)
mask_np = np.sum(np.array([label_image == idx for idx in keep_idxs]),axis=0)*255
mask = Image.fromarray(mask_np.astype('uint8'))
mask = ImageOps.invert(mask.convert("RGB"))

# mask_latent_np = cv2.resize(mask_np, (W // 8, H // 8), interpolation=cv2.INTER_NEAREST)
# mask_latent_np = np.expand_dims(mask_latent_np, axis=-1)
# mask_latent = array2tensor(mask_latent_np).cuda().repeat(num_samples, 1, 1, 1)
# ref = array2tensor(img_resized).float().cuda().repeat(num_samples, 1, 1, 1) / 255.
mask_image_exemple = load_image(
    "https://github.com/CompVis/latent-diffusion/raw/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
)

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16)

from examples.community.stable_diffusion_controlnet_inpaint_img2img import StableDiffusionControlNetInpaintImg2ImgPipeline

pipe = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                controlnet=controlnet, 
                safety_checker=None, 
                torch_dtype=torch.float16
            )

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# remove following line if xformers is not installed
# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)

image = pipe(
    "nice living room, high resolution",
    image_ref,
    mask,
    hough_image,
    num_inference_steps=20,
).images[0]

image.save("output_image.png")