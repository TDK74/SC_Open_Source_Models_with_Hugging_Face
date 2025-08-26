import requests
import torch

from PIL import Image
from transformers.utils import logging
from transformers import BlipForImageTextRetrieval
from transformers import AutoProcessor


logging.set_verbosity_error()

## ------------------------------------------------------##
model = BlipForImageTextRetrieval.from_pretrained("./models/Salesforce/blip-itm-base-coco")

## ------------------------------------------------------##
processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-itm-base-coco")

## ------------------------------------------------------##
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

## ------------------------------------------------------##
raw_image =  Image.open(requests.get(img_url, stream = True).raw).convert('RGB')

## ------------------------------------------------------##
raw_image

## ------------------------------------------------------##
text = "an image of a woman and a dog on the beach"

## ------------------------------------------------------##
inputs = processor(images = raw_image, text = text, return_tensors = "pt")

## ------------------------------------------------------##
inputs

## ------------------------------------------------------##
itm_scores = model(**inputs)[0]

## ------------------------------------------------------##
itm_scores

## ------------------------------------------------------##
itm_score = torch.nn.functional.softmax(itm_scores, dim = 1)

## ------------------------------------------------------##
print(f"""The image and text are matched with a probability of {itm_score[0][1]:.4f}""")
