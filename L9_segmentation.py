import numpy as np
import torch
import os
import gradio as gr

from transformers.utils import logging
from transformers import pipeline
from transformers import SamModel, SamProcessor
from PIL import Image
from helper import show_pipe_masks_on_image
from helper import show_mask_on_image


logging.set_verbosity_error()

## ------------------------------------------------------##
sam_pipe = pipeline("mask-generation", "./models/Zigeng/SlimSAM-uniform-77")

## ------------------------------------------------------##
raw_image = Image.open('meta_llamas.jpg')
raw_image.resize((720, 375))

## ------------------------------------------------------##
output = sam_pipe(raw_image, points_per_batch = 32)

## ------------------------------------------------------##
show_pipe_masks_on_image(raw_image, output)

## ------------------------------------------------------##
model = SamModel.from_pretrained("./models/Zigeng/SlimSAM-uniform-77")

processor = SamProcessor.from_pretrained("./models/Zigeng/SlimSAM-uniform-77")

## ------------------------------------------------------##
raw_image.resize((720, 375))

## ------------------------------------------------------##
input_points = [[[1600, 700]]]

## ------------------------------------------------------##
inputs = processor(raw_image, input_points = input_points, return_tensors = "pt")

## ------------------------------------------------------##
with torch.no_grad():
    outputs = model(**inputs)

## ------------------------------------------------------##
predicted_masks = processor.image_processor.post_process_masks(
                                                            outputs.pred_masks,
                                                            inputs["original_sizes"],
                                                            inputs["reshaped_input_sizes"]
                                                            )

## ------------------------------------------------------##
len(predicted_masks)

## ------------------------------------------------------##
predicted_mask = predicted_masks[0]
predicted_mask.shape

## ------------------------------------------------------##
outputs.iou_scores

## ------------------------------------------------------##
for i in range(3):
    show_mask_on_image(raw_image, predicted_mask[ : , i])

## ------------------------------------------------------##
depth_estimator = pipeline(task = "depth-estimation",
                           model = "./models/Intel/dpt-hybrid-midas")

## ------------------------------------------------------##
raw_image = Image.open('gradio_tamagochi_vienna.png')
raw_image.resize((806, 621))

## ------------------------------------------------------##
output = depth_estimator(raw_image)

## ------------------------------------------------------##
output

## ------------------------------------------------------##
output["predicted_depth"].shape

## ------------------------------------------------------##
output["predicted_depth"].unsqueeze(1).shape

## ------------------------------------------------------##
prediction = torch.nn.functional.interpolate(
                                            output["predicted_depth"].unsqueeze(1),
                                            size = raw_image.size[ : : -1],
                                            mode = "bicubic",
                                            align_corners = False,
                                            )

## ------------------------------------------------------##
prediction.shape

## ------------------------------------------------------##
raw_image.size[ : : -1],

## ------------------------------------------------------##
prediction

## ------------------------------------------------------##
output = prediction.squeeze().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)

## ------------------------------------------------------##
depth

## ------------------------------------------------------##
def launch(input_image):
    out = depth_estimator(input_image)

    prediction = torch.nn.functional.interpolate(
                                                out["predicted_depth"].unsqueeze(1),
                                                size = input_image.size[ : : -1],
                                                mode = "bicubic",
                                                align_corners = False,
                                                )

    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    return depth

## ------------------------------------------------------##
iface = gr.Interface(launch,
                     inputs = gr.Image(type = 'pil'),
                     outputs = gr.Image(type = 'pil'))

## ------------------------------------------------------##
iface.launch(share = True, server_port = int(os.environ['PORT1']))

## ------------------------------------------------------##
iface.close()
