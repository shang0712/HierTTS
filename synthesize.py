import matplotlib.pyplot as plt
import IPython.display as ipd
import argparse

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import numpy as np
from scipy.io.wavfile import write
from text.symbols import is_language

from preprocessors.utils import language_mapping
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def get_model(hps, args):
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
        config=hps.config)#.cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.checkpoint_path, net_g, None)
    return net_g

language_id = {"zh":0, "en":1, "uy":2, "es":3,"de":4}
speaker_info = {"ch-DB-4":"zh",
"kf-DB-2":"zh",
"jiangqiuzai":"zh",
"dongbei":"zh",
"baker": "zh",
"wangzihan_ch": "zh",
"ljspeech": "en",
"p251": "en",
"bz2021": "es",
"german_a": "de"}


def synthesize(args, model, hps):
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for i, spk in enumerate(open(hps.data.speakerlist,'r').readlines()):
        selected_style = {-1:0,0:3,1:5,4:6}
        with open(args.test_set,'r') as f:
            for j, line in enumerate(f):
                parts = line.strip().split('|')
                cs, phone, ipa, ipa_lang = parts[0], parts[3], parts[4], parts[5]

                text = phone
                x = get_text(text, hps).unsqueeze(0)
                x_lengths = torch.LongTensor([x.size(1)])
                selected_style[language_id[speaker_info[spk.strip()]]]=i
                selected_style[-1]=i
                print(x)
                print([selected_style[is_language(_x)] for _x in x[0]])
                style_id = torch.LongTensor(np.asarray([selected_style[is_language(_x)] for _x in x[0]])).unsqueeze(0)
                sid = torch.LongTensor([i])
                audio = model.infer(x, x_lengths, sid_src=style_id, sid_tgt=sid)[0,0].data.cpu().float().numpy()
                #audio = model.recon(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1, spec=spec, spec_lengths=spec_lengths, src_spec=src_spec, src_spec_lengths=src_spec_lengths)[0][0,0].data.cpu().float().numpy()
                data = (audio+1)/2*65525. - 32768
                data = data.astype(np.int16)
                write(os.path.join(save_path,'codeswitch.tim_{}_utt_{}.wav'.format(str(i),str(j))), hps.data.sampling_rate, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='exp_stylespeech_wav2vec/ckpt/checkpoint_480000.pth.tar',
        help="Path to the pretrained model")
    parser.add_argument('--config', default='configs/config_wav2vec.json')
    parser.add_argument("--save_path", type=str, default='results/')
    parser.add_argument("--test_set", type=str, default='../DataProcess/codeswitch_set/cs.txt')
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)


    # Get model
    model = get_model(hps, args)
    print('model is prepared')


    # Synthesize
    synthesize(args, model, hps)
