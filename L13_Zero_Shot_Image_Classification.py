from transformers.utils import logging
from transformers import CLIPModel
from transformers import AutoProcessor
from PIL import Image


logging.set_verbosity_error()

## ------------------------------------------------------##
model = CLIPModel.from_pretrained("./models/openai/clip-vit-large-patch14")

## ------------------------------------------------------##
processor = AutoProcessor.from_pretrained("./models/openai/clip-vit-large-patch14")

## ------------------------------------------------------##
image = Image.open("./kittens.jpeg")

## ------------------------------------------------------##
image

## ------------------------------------------------------##
labels = ["a photo of a cat", "a photo of a dog"]

## ------------------------------------------------------##
inputs = processor(text = labels, images = image, return_tensors = "pt", padding = True)

## ------------------------------------------------------##
outputs = model(**inputs)

## ------------------------------------------------------##
outputs

## ------------------------------------------------------##
outputs.logits_per_image

## ------------------------------------------------------##
probs = outputs.logits_per_image.softmax(dim = 1)[0]

## ------------------------------------------------------##
probs

## ------------------------------------------------------##
probs = list(probs)

for i in range(len(labels)):
  print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")
