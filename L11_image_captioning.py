import warnings

from transformers import BlipForConditionalGeneration
from transformers.utils import logging
from transformers import AutoProcessor
from PIL import Image


logging.set_verbosity_error()
warnings.filterwarnings("ignore", message = "Using the model-agnostic default `max_length`")

## ------------------------------------------------------##
model = BlipForConditionalGeneration.from_pretrained(
                                                "./models/Salesforce/blip-image-captioning-base")

## ------------------------------------------------------##
processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-image-captioning-base")

## ------------------------------------------------------##
image = Image.open("./beach.jpeg")

## ------------------------------------------------------##
image

## ------------------------------------------------------##
text = "a photograph of"
inputs = processor(image, text, return_tensors = "pt")

## ------------------------------------------------------##
inputs

## ------------------------------------------------------##
out = model.generate(**inputs)

## ------------------------------------------------------##
out

## ------------------------------------------------------##
print(processor.decode(out[0], skip_special_tokens = True))

## ------------------------------------------------------##
inputs = processor(image,return_tensors = "pt")

## ------------------------------------------------------##
out = model.generate(**inputs)

## ------------------------------------------------------##
print(processor.decode(out[0], skip_special_tokens = True))
