#import matplotlib.pyplot as plt
#import IPython.display as ipd
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

#from text.codeswitch import CodeSwitchGen, SpanishEnglishG2P
from preprocessors.utils import language_mapping
from string import punctuation
import librosa
#from resemblyzer import preprocess_wav
from mel_processing import spectrogram_torch

from data_utils import (
  TextAudioSpeakerLoader,
  DistributedBucketSampler
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def preprocess_text(text, src_language='English', tgt_language='Spanish'):
    text = text.rstrip(punctuation)
    print('Loading resource...')
    Augment = CodeSwitchGen(0.0, src_language=src_language, tgt_language=tgt_language)
    print('Loading done!')

    aug_text, phones = Augment.cross_list(text, language_mapping[src_language], language_mapping[tgt_language])

    print("Raw Text Sequence: {}".format(text))
    print("Codeswitch Text Sequence: {}".format(aug_text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))

    return torch.from_numpy(sequence)#.to(device=device)

def preprocess_audio(audio_file, hps):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    if sample_rate != 16000:
        wav = librosa.resample(wav, sample_rate, 16000)
    wav = preprocess_wav(wav)
    spec = spectrogram_torch(torch.from_numpy(wav), hps.data.filter_length,
        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
        center=False)
    return spec#.to(device=device)

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

def synthesize(args, model, hps):
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    #for i in range(len(open(hps.data.speakerlist,'r').readlines()))[:1]:
    for i in range(1):
        eval_dataset = TextAudioSpeakerLoader(hps.data.data_path, args.txt, hps.data.speakerlist, hps.data)
        collate_fn = eval_dataset.collate_fn
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
            batch_size=1, pin_memory=True,
            drop_last=False, collate_fn=collate_fn)


        count = [0]*len(hps.data.speakerlist)
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, d_target_m, d_target_e, p_target, e_target, log_D, txt2sub, sub2phn_m, sub2phn_e, txt, txt_len, word2sub_m, word2sub_e, word_len, sub2sub, sub_len, _, sent2word_m, sent2word_e) in enumerate(eval_loader):
            '''
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speakers = speakers.cuda(0)
            d_target = d_target.cuda(0)
            p_target = p_target.cuda()
            e_target = e_target.cuda()
            '''

            if args.analysis:
                sid = torch.LongTensor([i])#.cuda()
                audios = model.analysis(x, x_lengths,  txt, txt_len, txt2sub, sid, d_target_m, d_target_e, sub2phn_m, sub2phn_e,  word2sub_m, word2sub_e, sent2word_m, sent2word_e, word_len, sub2sub, sub_len)
                for key in audios.keys():
                    audio = audios[key][0,0].data.float().numpy()
                    data = (audio+1)/2*65525. - 32768
                    data = data.astype(np.int16)
                    write(os.path.join(save_path,'transfer.tim_{}_sty_{}_utt_{}_key_{}.wav'.format(str(i),str(speakers[0].item()),count[speakers[0].item()], key)), hps.data.sampling_rate, data)
                count[speakers[0].item()]+=1
                break
            else:
                sid = torch.LongTensor([i])#.cuda()
                audio = model.infer(x, x_lengths,  txt, txt_len, txt2sub, sid, None, None, sub2phn_m, sub2phn_e,  word2sub_m, word2sub_e, sent2word_m, sent2word_e, word_len, sub2sub, sub_len)[0,0].data.float().numpy()

                data = (audio+1)/2*65525. - 32768
                data = data.astype(np.int16)
                write(os.path.join(save_path,'transfer.tim_{}_sty_{}_utt_{}.wav'.format(str(i),str(speakers[0].item()),count[speakers[0].item()])), hps.data.sampling_rate, data)
                count[speakers[0].item()]+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='exp_stylespeech_wav2vec/ckpt/checkpoint_480000.pth.tar',
        help="Path to the pretrained model")
    parser.add_argument('--config', default='configs/config_wav2vec.json')
    parser.add_argument("--save_path", type=str, default='results/')
    parser.add_argument("--analysis", type=bool, default=False,
        help="switch")
    parser.add_argument("--txt", type=str, default='sub1.txt',
        help="txt")
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)


    # Get model
    model = get_model(hps, args)
    print('model is prepared')


    # Synthesize
    synthesize(args, model, hps)
