import timm

def get_model(model_name: str='mobilenetv4_conv_small_050.e3000_r224_in1k'):
# https://huggingface.co/timm/mobilenetv4_conv_small_050.e3000_r224_in1k
	model = timm.create_model(
		model_name,
		pretrained=True,
	)
	data_config = timm.data.resolve_model_data_config(model)
	transforms_timm = timm.data.create_transform(**data_config, is_training=False)

	return model, transforms_timm