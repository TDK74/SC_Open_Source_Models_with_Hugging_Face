import os
import gradio as gr

from helper import load_image_from_url, render_results_in_image
from helper import ignore_warnings
from helper import summarize_predictions_natural_language
from IPython.display import Audio as IPythonAudio
from PIL import Image
from transformers import pipeline
from transformers.utils import logging

## ------------------------------------------------------##
logging.set_verbosity_error()
ignore_warnings()

## ------------------------------------------------------##
od_pipe = pipeline("object-detection", "./models/facebook/detr-resnet-50")

## ------------------------------------------------------##
raw_image = Image.open('huggingface_friends.jpg')
raw_image.resize((569, 491))

## ------------------------------------------------------##
pipeline_output = od_pipe(raw_image)

## ------------------------------------------------------##
processed_image = render_results_in_image(raw_image, pipeline_output)

## ------------------------------------------------------##
def get_pipeline_prediction(pil_image):

    pipeline_output = od_pipe(pil_image)

    processed_image = render_results_in_image(pil_image, pipeline_output)

    return processed_image

## ------------------------------------------------------##
demo = gr.Interface(
                  fn = get_pipeline_prediction,
                  inputs = gr.Image(label = "Input image",
                                    type = "pil"),
                  outputs = gr.Image(label = "Output image with predicted instances",
                                     type = "pil")
                )

## ------------------------------------------------------##
demo.launch(share = True, server_port = int(os.environ['PORT1']))

## ------------------------------------------------------##
demo.close()

## ------------------------------------------------------##
pipeline_output

## ------------------------------------------------------##
od_pipe     # print(od_pipe)

## ------------------------------------------------------##
raw_image = Image.open('huggingface_friends.jpg')
raw_image.resize((284, 245))

## ------------------------------------------------------##
text = summarize_predictions_natural_language(pipeline_output)

## ------------------------------------------------------##
text    # print(text)

## ------------------------------------------------------##
tts_pipe = pipeline("text-to-speech", model = "./models/kakao-enterprise/vits-ljs")

## ------------------------------------------------------##
narrated_text = tts_pipe(text)

## ------------------------------------------------------##
IPythonAudio(narrated_text["audio"][0], rate = narrated_text["sampling_rate"])
