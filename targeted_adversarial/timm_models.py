import timm
import torch
import numpy as np
from .io import load_imagenet_classes

def get_model(model_name: str='mobilenetv4_conv_small_050.e3000_r224_in1k'):
# https://huggingface.co/timm/mobilenetv4_conv_small_050.e3000_r224_in1k
	model = timm.create_model(
		model_name,
		pretrained=True,
	)
	data_config = timm.data.resolve_model_data_config(model)
	transforms_timm = timm.data.create_transform(**data_config, is_training=False)

	return model, transforms_timm


def predict_topk(image, model, transforms, k=5):
	output = model(transforms(image).unsqueeze(0))
	top_probabilities, top_class_indices = torch.topk(output.softmax(dim=1) * 100, k=k)
	return top_probabilities, top_class_indices