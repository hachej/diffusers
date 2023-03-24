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


keep_labels = ['windowpane','sky', 'fireplace', 'stairs']
keep_from_line_labels = ['ceiling','wall','floor','door']
value_threshold = 0.61
distance_threshold = 6

# download an image
image_ref = load_image(
    #"https://media2.homegate.ch/f_auto/t_web_dp_fullscreen/listings/x379/3002675848/image/bfa446e81f1ab2c653df85902131f113.jpg"
    "https://media2.homegate.ch/f_auto/t_web_dp_fullscreen/listings/x379/3002675848/image/8e85d1109cc7644fc2ae3d823d9c100b.jpg"
)
image_ref.save("out_image_original.png")
max_width = 1024
scale = 1024 / image_ref.size[0]
image_ref = image_ref.resize((1024,int(image_ref.size[1]* scale)),Image.Resampling.NEAREST)

image = np.array(image_ref)
H, W, C = np.array(image).shape

# get segmentation
uniformer = UniformerDetector.from_pretrained("lllyasviel/ControlNet")
detected_map_seg, label_image = uniformer(image)

classes = uniformer.model.CLASSES
label_image = label_image[0]
keep_idxs = [classes.index(label) for label in keep_labels]
keep_from_line_idxs = [classes.index(label) for label in keep_from_line_labels]

mask_np = np.sum(np.array([label_image == idx for idx in keep_idxs]),axis=0)*255
mask_np = cv2.dilate(mask_np.astype('uint8'), np.ones((5, 5), np.uint8), iterations=1)
mask = Image.fromarray(mask_np.astype('uint8'))
mask = ImageOps.invert(mask.convert("RGB"))
mask = mask.resize((W,H),Image.Resampling.NEAREST)

mask.save("out_mask.png")

# get hough image
model = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
hough_image = model(image, value_threshold, distance_threshold)
line_mask = np.sum(np.array([label_image == idx for idx in keep_from_line_idxs]),axis=0)
hough_image = hough_image * np.expand_dims(line_mask,axis=-1).astype('uint8')
hough_image = Image.fromarray(hough_image.astype('uint8'))
hough_image = hough_image.resize((W,H),Image.Resampling.NEAREST)
hough_image.save("out_controlnet.png")

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16)

from examples.community.stable_diffusion_controlnet_inpaint_img2img import StableDiffusionControlNetInpaintImg2ImgPipeline

pipe = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained(
    #"runwayml/stable-diffusion-v1-5", -> generate error...
    "runwayml/stable-diffusion-inpainting", 
    controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)


pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# remove following line if xformers is not installed
# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)

image = pipe(
    "nice living room, high resolution, furnitures, couch",
    image_ref,
    mask,
    hough_image,
    num_inference_steps=40,
).images[0]

image.save("output_image.png")