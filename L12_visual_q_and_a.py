import warnings

from transformers.utils import logging
from transformers import BlipForQuestionAnswering
from transformers import AutoProcessor
from PIL import Image


logging.set_verbosity_error()
warnings.filterwarnings("ignore", message = "Using the model-agnostic default `max_length`")

## ------------------------------------------------------##
model = BlipForQuestionAnswering.from_pretrained("./models/Salesforce/blip-vqa-base")

## ------------------------------------------------------##
processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-vqa-base")

## ------------------------------------------------------##
image = Image.open("./beach.jpeg")

## ------------------------------------------------------##
image

## ------------------------------------------------------##
question = "how many dogs are in the picture?"

## ------------------------------------------------------##
inputs = processor(image, question, return_tensors = "pt")

## ------------------------------------------------------##
out = model.generate(**inputs)

## ------------------------------------------------------##
print(processor.decode(out[0], skip_special_tokens = True))
