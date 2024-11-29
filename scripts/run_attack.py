import typer
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from targeted_adversarial.io import load_imagenet_classes, read_image, retrieve_imagenet_image, save_tensor_as_jpeg
from targeted_adversarial.timm_models import get_model, predict_topk
from targeted_adversarial import pgd_linf_targ

app = typer.Typer(
    name="targeted-adversarial",
    help="Generate targeted adversarial examples for image classification models",
    no_args_is_help=True,
)

def validate_target_class(value: str) -> str:
	"""Validate that the target class is a valid ImageNet synset ID."""
	synset_map = load_imagenet_classes()
	if value not in synset_map:
		valid_examples = list(synset_map.keys())[:3]
		raise typer.BadParameter(
			f"Invalid ImageNet synset ID. Must be one of the ImageNet class IDs.\n"
			f"Example valid IDs: {', '.join(valid_examples)}\n"
			f"Use --list-classes to see all available classes."
		)
	return value

def validate_epsilon(value: float) -> float:
	"""Validate that epsilon is between 0 and 1."""
	if not 0 <= value <= 1:
		raise typer.BadParameter("Epsilon must be between 0 and 1")
	return value

@app.command()
def list_classes():
	"""List all available ImageNet classes and their synset IDs."""
	synset_map = load_imagenet_classes()

	typer.echo("Available ImageNet classes:")
	for synset_id, description in synset_map.items():
		typer.echo(f"{synset_id} {description}")

@app.command()
def run_attack(
	path_to_source_image: Path = typer.Argument(
		...,
		help="Path to the source image file, should be an RGB jpeg file",
		exists=True,
		file_okay=True,
		dir_okay=False,
		resolve_path=True,
	),
	target_class: str = typer.Argument(
		...,
		help="Target ImageNet synset ID (e.g., n03063689). Use --list-classes to see available IDs",
		callback=validate_target_class,
	),
	epsilon: float = typer.Option(
		0.1,
		"--epsilon", "-e",
		help="Maximum perturbation size (L infinity norm)",
		callback=validate_epsilon,
		show_default=True,
	),
	alpha: float = typer.Option(
		3e-2,
		"--alpha", "-a",
		help="Step size for each iteration",
		show_default=True,
	),
	num_iter: int = typer.Option(
		40,
		"--num-iter", "-n",
		help="Number of PGD iterations",
		min=1,
		show_default=True,
	),
	output_path: Optional[Path] = typer.Option(
		None,
		"--output", "-o",
		help="Path to save the adversarial example (defaults to source_adversarial.png)",
		dir_okay=False,
		resolve_path=True,
	),
	device: str = typer.Option(
		"cuda",
		"--device", "-d",
		help="Inference device (cuda or cpu)",
		show_default=True,
	),
) -> None:
	"""
	Generate a targeted adversarial example using PGD attack.

	This command takes a source image and generates an adversarial example
	that aims to be classified as the target ImageNet class while maintaining
	visual similarity to the source image.

	Example:
		$ targeted-adversarial run-attack cat.jpg n02123045 --epsilon 0.1 --num-iter 50
		(n02123045 is the synset ID for 'tabby cat')

	Use the --list-classes command to see all available ImageNet classes:
		$ targeted-adversarial list-classes

	The attack uses Projected Gradient Descent (PGD) with L∞ norm constraint.
	The parameters epsilon, alpha, and num_iter control the strength and
	precision of the attack.
	"""
	synset_map = load_imagenet_classes()
	values_synset_map = np.array(list(synset_map.values()))
	target_description = synset_map[target_class]

	# setup model and inputs transforms
	model, transforms = get_model(model_name='mobilenetv4_conv_small_050.e3000_r224_in1k')

	# Set default output path if not provided
	if output_path is None:
		output_path = path_to_source_image.parent / f"{path_to_source_image.stem}_adversarial_{target_description[:5]}.jpg"

	typer.echo(f"Loading source image from {path_to_source_image}")
	source_image = read_image(str(path_to_source_image))
	# print(source_image.shape)

	top_probabilities, top_class_indices = predict_topk(source_image, model, transforms)
	class_names = values_synset_map[top_class_indices.cpu().detach().numpy()]

	# print(class_names.shape)
	# print(top_probabilities.size())

	typer.echo("Source image top predicted classes are:")
	for c, p in zip(class_names.reshape(-1), top_probabilities.cpu().detach().numpy().flatten()):
		print(c, p)
		
	typer.echo(f"\nTarget class: {target_class} ({target_description}), pulling sample image of target class...")
	target_class_image = retrieve_imagenet_image(target_class)
	print(target_class_image.size())

	top_probabilities, top_class_indices = predict_topk(target_class_image, model, transforms)
	class_names = values_synset_map[top_class_indices.cpu().detach().numpy()]
    
	typer.echo("Target class image top predicted classes are:")
	for c, p in zip(class_names.reshape(-1), top_probabilities.cpu().detach().numpy().flatten()):
		print(c, p)

	typer.echo(f"Use {class_names.reshape(-1)[0]} as target class...")

	X = torch.cat([transforms(source_image.unsqueeze(0)), transforms(target_class_image.unsqueeze(0))])

	output = model(X)
	top_p, top_c = torch.topk(output.softmax(dim=1) * 100, k=1)

	y_true = top_c
	y_target = y_true.detach().clone()
	y_target = y_target.roll(1)

	try:
		delta = pgd_linf_targ(
		model=model,
		X=X,
		y=y_true.squeeze(),
		epsilon=epsilon,
		alpha=alpha,
		num_iter=num_iter,
		y_targ=y_target.squeeze(),
		)

		# Save the result
		typer.echo(f"Saving adversarial example to {output_path}")

		adversarial_X = X.to(device) + delta.to(device)
		save_tensor_as_jpeg(adversarial_X[0], str(output_path))

		typer.echo("Adversarial image top predicted classes are:")
		output = model(adversarial_X)
		top_p, top_c = torch.topk(output.softmax(dim=1) * 100, k=5)
		class_names = values_synset_map[top_c[0].cpu().detach().numpy()]
    
		for c, p in zip(class_names.reshape(-1), top_p[0].cpu().detach().numpy().flatten()):
			print(c, p)

	except Exception as e:
		typer.secho(f"Error during attack: {str(e)}", fg=typer.colors.RED)
		raise typer.Exit(1)

	typer.secho(
	f"Successfully generated adversarial example: {output_path}",
	fg=typer.colors.GREEN,
	)

if __name__ == "__main__":
	app()