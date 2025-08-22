import os
import gradio as gr
import soundfile as sf
import io
import numpy as np
import librosa

from datasets import load_dataset
from transformers.utils import logging
from transformers import pipeline
from IPython.display import Audio as IPythonAudio


logging.set_verbosity_error()

## ------------------------------------------------------##
dataset = load_dataset("librispeech_asr",
                       split = "train.clean.100",
                       streaming = True,
                       trust_remote_code = True)

## ------------------------------------------------------##
example = next(iter(dataset))

## ------------------------------------------------------##
dataset_head = dataset.take(5)
list(dataset_head)

## ------------------------------------------------------##
list(dataset_head)[2]

## ------------------------------------------------------##
example     # print(example)

## ------------------------------------------------------##
IPythonAudio(example["audio"]["array"],
             rate = example["audio"]["sampling_rate"])

## ------------------------------------------------------##
asr = pipeline(task = "automatic-speech-recognition",
               model = "distil-whisper/distil-small.en")

## ------------------------------------------------------##
asr.feature_extractor.sampling_rate

## ------------------------------------------------------##
asr(example["audio"]["array"])

## ------------------------------------------------------##
example["text"]     # print(example["text"])

## ------------------------------------------------------##
demo = gr.Blocks()

## ------------------------------------------------------##
def transcribe_speech(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")

        return ""

    output = asr(filepath)

    return output["text"]

## ------------------------------------------------------##
mic_transcribe = gr.Interface(
                            fn = transcribe_speech,
                            inputs = gr.Audio(sources = "microphone",
                                              type = "filepath"),
                            outputs = gr.Textbox(label = "Transcription",
                                                 lines = 3),
                            allow_flagging = "never")

## ------------------------------------------------------##
file_transcribe = gr.Interface(
                            fn = transcribe_speech,
                            inputs = gr.Audio(sources = "upload",
                                              type = "filepath"),
                            outputs = gr.Textbox(label = "Transcription",
                                                 lines = 3),
                            allow_flagging = "never",
                            )

## ------------------------------------------------------##
with demo:
    gr.TabbedInterface(
                        [mic_transcribe,
                         file_transcribe],
                        ["Transcribe Microphone",
                         "Transcribe Audio File"],
                        )

demo.launch(share = True,
            server_port = int(os.environ['PORT1']))

## ------------------------------------------------------##
demo.close()

## ------------------------------------------------------##
audio, sampling_rate = sf.read('narration_example.wav')

## ------------------------------------------------------##
sampling_rate   # print(sampling_rate)

## ------------------------------------------------------##
asr.feature_extractor.sampling_rate

## ------------------------------------------------------##
asr(audio)

## ------------------------------------------------------##
audio.shape

## ------------------------------------------------------##
audio_transposed = np.transpose(audio)

## ------------------------------------------------------##
audio_transposed.shape

## ------------------------------------------------------##
audio_mono = librosa.to_mono(audio_transposed)

## ------------------------------------------------------##
IPythonAudio(audio_mono, rate = sampling_rate)

## ------------------------------------------------------##
asr(audio_mono)

## ------------------------------------------------------##
sampling_rate   # print(sampling_rate)

## ------------------------------------------------------##
asr.feature_extractor.sampling_rate

## ------------------------------------------------------##
audio_16KHz = librosa.resample(audio_mono,
                               orig_sr = sampling_rate,
                               target_sr = 16000)

## ------------------------------------------------------##
asr(
    audio_16KHz,
    chunk_length_s = 30,
    batch_size = 4,
    return_timestamps = True,
    )["chunks"]

## ------------------------------------------------------##
demo = gr.Blocks()

## ------------------------------------------------------##
def transcribe_long_form(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")

        return ""

    output = asr(
                  filepath,
                  max_new_tokens = 256,
                  chunk_length_s = 30,
                  batch_size = 8,
                )

    return output["text"]

## ------------------------------------------------------##
mic_transcribe = gr.Interface(
                            fn = transcribe_long_form,
                            inputs = gr.Audio(sources = "microphone",
                                              type = "filepath"),
                            outputs=gr.Textbox(label = "Transcription",
                                               lines = 3),
                            allow_flagging = "never")

file_transcribe = gr.Interface(
                                fn = transcribe_long_form,
                                inputs = gr.Audio(sources = "upload",
                                                  type = "filepath"),
                                outputs=gr.Textbox(label = "Transcription",
                                                   lines = 3),
                                allow_flagging = "never",
                                )

## ------------------------------------------------------##
with demo:
    gr.TabbedInterface(
                        [mic_transcribe,
                         file_transcribe],
                        ["Transcribe Microphone",
                         "Transcribe Audio File"],
                        )

demo.launch(share = True,
            server_port = int(os.environ['PORT1']))

## ------------------------------------------------------##
demo.close()

## ------------------------------------------------------##
audio, sampling_rate = sf.read('narration_example.wav')

## ------------------------------------------------------##
sampling_rate   # print(sampling_rate)

## ------------------------------------------------------##
asr.feature_extractor.sampling_rate
