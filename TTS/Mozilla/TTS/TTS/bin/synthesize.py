#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import string
import json
from argparse import RawTextHelpFormatter
# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # pylint: disable=bad-continuation
    parser = argparse.ArgumentParser(description='''Synthesize speech on command line.\n\n'''

    '''You can either use your trained model or choose a model from the provided list.\n'''\

    '''
    Example runs:

    # list provided models
    ./TTS/bin/synthesize.py --list_models

    # run a model from the list
    ./TTS/bin/synthesize.py --text "Text for TTS" --model_name "<language>/<dataset>/<model_name>" --vocoder_name "<language>/<dataset>/<model_name>" --output_path

    # run your own TTS model (Using Griffin-Lim Vocoder)
    ./TTS/bin/synthesize.py --text "Text for TTS" --model_path path/to/model.pth.tar --config_path path/to/config.json --out_path output/path/speech.wav

    # run your own TTS and Vocoder models
    ./TTS/bin/synthesize.py --text "Text for TTS" --model_path path/to/config.json --config_path path/to/model.pth.tar --out_path output/path/speech.wav
        --vocoder_path path/to/vocoder.pth.tar --vocoder_config_path path/to/vocoder_config.json

    ''',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--list_models',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='list available pre-trained tts and vocoder models.'
        )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Text to generate speech.'
        )
    parser.add_argument(
        '--text_file',
        type=str,
        default=None,
        help='Text file to generate speech.'
    )

    # Args for running pre-trained TTS models.
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help=
        'Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>'
    )
    parser.add_argument(
        '--vocoder_name',
        type=str,
        default=None,
        help=
        'Name of one of the pre-trained  vocoder models in format <language>/<dataset>/<model_name>'
    )

    # Args for running custom models
    parser.add_argument(
        '--config_path',
        default=None,
        type=str,
        help='Path to model config file.'
        )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model file.',
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default=Path(__file__).resolve().parent,
        help='Path to save final wav file. Wav file will be named as the given text.',
    )
    parser.add_argument(
        '--use_cuda',
        type=bool,
        help='Run model on CUDA.',
        default=True
        )
    parser.add_argument(
        '--vocoder_path',
        type=str,
        help=
        'Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).',
        default=None,
    )
    parser.add_argument(
        '--vocoder_config_path',
        type=str,
        help='Path to vocoder model config file.',
        default=None)

    # args for multi-speaker synthesis
    parser.add_argument(
        '--speakers_json',
        type=str,
        help="JSON file for multi-speaker model.",
        default=None)
    parser.add_argument(
        '--speaker_idx',
        type=str,
        help="if the tts model is trained with x-vectors, then speaker_idx is a file present in speakers.json else speaker_idx is the speaker id corresponding to a speaker in the speaker embedding layer.",
        default=None)
    parser.add_argument(
        '--gst_style',
        help="Wav path file for GST stylereference.",
        default=None)

    # aux args
    parser.add_argument(
        '--save_spectogram',
        type=bool,
        help="If true save raw spectogram for further (vocoder) processing in out_path.",
        default=False)

    args = parser.parse_args()

    # load model manager
    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path)

    model_path = None
    config_path = None
    vocoder_path = None
    vocoder_config_path = None

    # CASE1: list pre-trained TTS models
    if args.list_models:
        manager.list_models()
        sys.exit()

    # CASE2: load pre-trained models
    if args.model_name is not None:
        model_path, config_path = manager.download_model(args.model_name)

    if args.vocoder_name is not None:
        vocoder_path, vocoder_config_path = manager.download_model(args.vocoder_name)

    # CASE3: load custome models
    if args.model_path is not None:
        model_path = args.model_path
        config_path = args.config_path

    if args.vocoder_path is not None:
        vocoder_path = args.vocoder_path
        vocoder_config_path = args.vocoder_config_path

    # RUN THE SYNTHESIS
    # load models
    synthesizer = Synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, args.use_cuda)

    use_griffin_lim = vocoder_path is None
    print(" > Text_file: {}".format(args.text_file))

    with open(args.text_file, encoding="utf-8")as f:
        # lines = f.read().splitlines()
        lines = json.load(f)

    # # handle multi-speaker setting
    # if not model_config.use_external_speaker_embedding_file and args.speaker_idx is not None:
    #     if args.speaker_idx.isdigit():
    #         args.speaker_idx = int(args.speaker_idx)
    #     else:
    #         args.speaker_idx = None
    # else:
    #     args.speaker_idx = None

    # if args.gst_style is None:
    #     if 'gst' in model_config.keys() and model_config.gst['gst_style_input'] is not None:
    #         gst_style = model_config.gst['gst_style_input']
    #     else:
    #         gst_style = None
    # else:
    #     # check if gst_style string is a dict, if is dict convert  else use string
    #     try:
    #         gst_style = json.loads(args.gst_style)
    #         if max(map(int, gst_style.keys())) >= model_config.gst['gst_style_tokens']:
    #             raise RuntimeError("The highest value of the gst_style dictionary key must be less than the number of GST Tokens, \n Highest dictionary key value: {} \n Number of GST tokens: {}".format(max(map(int, gst_style.keys())), model_config.gst['gst_style_tokens']))
    #     except ValueError:
    #         gst_style = args.gst_style

    # kick it
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    for id, line in enumerate(lines):
        # file_name, text, _ = json.loads(line).values()
        text, _, _, _ = line.values()
        # wav = synthesizer.tts(text)
        # if text[-1] != "." and text[-1] != "\"\"" and text[-1] != "\'\'" and text[-1] != "\"" and text[-1] != "\'":
        #     text += "."
        if text[-1] == ";":
            text = text[:-1]
        if "-LCB-" in text:
            text.remove("-LCB-")
        wav = synthesizer.tts(" ".join(text))

        # save the results
        file_name = args.text_file.split("/")[-1].split(".")[0] + "-" + str(id) + '.wav'
        out_path = os.path.join(args.out_path, file_name)
        print(" > Saving output to {}".format(out_path))
        synthesizer.save_wav(wav, out_path)


if __name__ == "__main__":
    main()