import timm 
import torch

# https://huggingface.co/timm/mobilenetv4_conv_small_050.e3000_r224_in1k

def predict_timm(X, model_name="mobilenetv4_conv_small_050.e3000_r224_in1k"):

	model = timm.create_model(
		model_name,
		pretrained=True,
	)
	model.eval();
	data_config = timm.data.resolve_model_data_config(model)
	transforms = timm.data.create_transform(**data_config, is_training=False)
	output = model(transforms(torch.rand((3,224,224))).unsqueeze(0))
	return output

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=10)

if __name__=="__main__":
	image = torch.rand((3,224,224))
	preds = predict_timm(image)
	top_probabilities, top_class_indices = torch.topk(preds.softmax(dim=1) * 100, k=10)

	# TODO class labels