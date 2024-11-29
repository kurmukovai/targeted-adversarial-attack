
import requests
import io
import json
import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path



def load_imagenet_classes() -> Dict[str, str]:
	"""Load ImageNet synset IDs and their descriptions."""
	synset_map = {}
	synset_file = Path(__file__).parent / "assets" /  "LOC_synset_mapping.txt"
	with open(synset_file, 'r') as f:
		for line in f:
		# Each line format: n02119789 kit_fox, red_fox, Vulpes_vulpes
			synset_id, description = line.strip().split(' ', 1)
			synset_map[synset_id] = description
	return synset_map


def load_imagenet_sample_names() -> Dict[str, str]:
	file = Path(__file__).parent / "assets" /  "imagenet_sample_1000.json"
	with open(file, "r") as f:
		file_names = json.load(f)
	return file_names


def retrieve_imagenet_image(synset_class: str):
	"""Downloads a single image from the ImageNet from `synset_class`."""
	file_names = load_imagenet_sample_names() 
	image_url = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/refs/heads/master/{file_names[synset_class]}.JPEG"
	img_data = requests.get(image_url).content
	img_pil = Image.open(io.BytesIO(img_data))
	return np.array(img_pil)


def normalize8(I):
  # https://stackoverflow.com/a/53236206
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn)/mx) * 255
	# im_dog_8 = normalize8(rearrange(transforms_timm(im_dog).detach().numpy(), "c h w -> h w c"))
    return I.astype(np.uint8)


def read_image(file: str):
	return torchvision.io.read_image(file) / 255
