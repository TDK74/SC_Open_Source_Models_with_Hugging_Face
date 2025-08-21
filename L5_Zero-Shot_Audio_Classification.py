from transformers.utils import logging
from transformers import pipeline
from datasets import load_dataset, load_from_disk
from datasets import Audio
from IPython.display import Audio as IPythonAudio


logging.set_verbosity_error()

## ------------------------------------------------------##

# dataset = load_dataset("ashraq/esc50", split = "train[0 : 10]")
dataset = load_from_disk("./models/ashraq/esc50/train")

## ------------------------------------------------------##
audio_sample = dataset[0]

## ------------------------------------------------------##
audio_sample    # print(audio_sample)

## ------------------------------------------------------##
IPythonAudio(audio_sample["audio"]["array"],
             rate = audio_sample["audio"]["sampling_rate"])

## ------------------------------------------------------##
zero_shot_classifier = pipeline(task = "zero-shot-audio-classification",
                                model = "./models/laion/clap-htsat-unfused")

## ------------------------------------------------------##
zero_shot_classifier.feature_extractor.sampling_rate

## ------------------------------------------------------##
audio_sample["audio"]["sampling_rate"]

## ------------------------------------------------------##
dataset = dataset.cast_column("audio", Audio(sampling_rate = 48_000))

## ------------------------------------------------------##
audio_sample = dataset[0]

## ------------------------------------------------------##
audio_sample    # print(audio_sample)

## ------------------------------------------------------##
candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]

## ------------------------------------------------------##
candidate_labels = ["Sound of a child crying",
                    "Sound of vacuum cleaner",
                    "Sound of a bird singing",
                    "Sound of an airplane"]

## ------------------------------------------------------##
zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels = candidate_labels)
