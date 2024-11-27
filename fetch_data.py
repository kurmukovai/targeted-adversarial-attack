import os
import kaggle
import json

def match_labels():
	labels = dict()
	with open("./data/tiny-imagenet-200/words.txt", 'r') as f:
		for line in f:
			if line.strip():
				key, value = line.strip().split('\t')
				labels[key] = value
	return labels


if __name__=="__main__":
  
	# Set Kaggle username and API key
	# Use ~/.kaggle/kaggle.json instead
	# os.environ["KAGGLE_USERNAME"] = ""
	# os.environ["KAGGLE_KEY"] = ""

	api = kaggle.KaggleApi()
	api.authenticate()

	# Select tiny ImageNet
	dataset_name = "akash2sharma/tiny-imagenet"
	api.dataset_download_files(dataset_name, path='./data', quiet=False, unzip=True)
    
	# Match and dump label names
	labels = match_labels()
	dataset_classes = os.listdir('./data/tiny-imagenet-200/train')
	labels_tiny = {key: labels[key] for key in dataset_classes}
	with open('./data/tiny-imagenet-200/labels_tiny_map.json', 'w') as file:
		json.dump(labels_tiny, file)


# TODO: pass folder as param 