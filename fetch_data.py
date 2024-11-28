import json

def match_labels_timm():
  labels = dict()
  with open("LOC_synset_mapping.txt", 'r') as f:
    for line in f:
      if line.strip():
        key, value = line[:9].strip(), line[9:].strip()
        labels[key] = value
  return labels

if __name__=="__main__":
	labels_timm_trained_model = match_labels_timm()
	with open('./data/tiny-imagenet-200/labels_timm_trained_model.json', 'w') as file:
		json.dump(labels_timm_trained_model, file)
