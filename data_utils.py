import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from utils import pad_1D, pad_2D, process_meta_multi, ali_mask
from text import text_to_sequence
import json
from transformers import BertTokenizer

class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths

def replace_outlier(values, max_v, min_v):
    values = np.where(values<max_v, values, max_v)
    values = np.where(values>min_v, values, min_v)
    return values


def norm_mean_std(x, mean, std):
    x = (x - mean) / std
    return x


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, filename="train.txt.1", namelist=[], config=None, dataset='multispeaker'):
        self.dataset = dataset
        self.outlier = config.outlier if 'outlier' in config else None
        self.data_path, self.basename, self.text, self.ptext, self.mask, self.sid, self.lengths, self.clean_txt, self.txt2sub, self.sub2phn, self.encoding, self.word2sub, self.space, self.sub2sub = process_meta_multi([os.path.join(data_path, x.strip(), filename) for x in namelist], config.hop_length, self.outlier)
        self.sid_dict = self.create_speaker_table(self.sid)
        '''
        f0s = [np.load(os.path.join(data_path, x.strip(), 'f0.npy')) for x in open(namelist,'r').readlines()]
        f0 = np.concatenate(f0s,axis=0)
        energys = [np.load(os.path.join(data_path, x.strip(), 'energy.npy')) for x in open(namelist,'r').readlines()]
        energy = np.concatenate(energys,axis=0)

        f0_max = np.max(f0)
        f0_min = np.min(f0)
        f0_mean = np.mean(f0)
        f0_std = np.std(f0)
        energy_max = np.max(energy)
        energy_min = np.min(energy)
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
    
        stats_config = {
            "f0_stat": [f0_max.item(), f0_min.item(), f0_mean.item(), f0_std.item()],
            "energy_stat": [energy_max.item(), energy_min.item(), energy_mean.item(), energy_std.item()]
        }
        with open(os.path.join(data_path, 'stats.json'), 'w') as f:
            json.dump(stats_config, f)

        self.f0_stat = stats_config["f0_stat"] # max, min, mean, std
        self.energy_stat = stats_config["energy_stat"] # max, min, mean, std
        '''

        self.config = config

        self.create_sid_to_index()
        print('Speaker Num :{}'.format(len(self.sid_dict)))
        print('item Num :{}'.format(len(self.data_path)))

    def create_speaker_table(self, sids):
        speaker_ids = np.sort(np.unique(sids))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        return d

    def create_sid_to_index(self):
        _sid_to_indexes = {} 
        # for keeping instance indexes with the same speaker ids
        for i, sid in enumerate(self.sid):
            if sid in _sid_to_indexes:
                _sid_to_indexes[sid].append(i)
            else:
                _sid_to_indexes[sid] = [i]
        self.sid_to_indexes = _sid_to_indexes

    def __len__(self):
        return len(self.text)

    def get_audio(self, audio, spec_path):
        if os.path.exists(spec_path):
            spec = np.load(spec_path)
        else:
            #if not os.path.exists(os.path.dirname(spec_path)):
            #    os.mkdir(os.path.dirname(spec_path))
            spec = spectrogram_torch(torch.from_numpy(audio), self.config.filter_length,
                self.config.sampling_rate, self.config.hop_length, self.config.win_length,
                center=False)
            spec = torch.squeeze(spec, 0).numpy()
            np.save(spec_path, spec, allow_pickle=False)
        return spec

    def __getitem__(self, idx):
        basename = self.basename[idx]
        sid = self.sid_dict[self.sid[idx]]
        phone = np.array(text_to_sequence(self.text[idx], []))
        p_phone = np.array(text_to_sequence(self.ptext[idx], []))
        mask = [int(x) for x in self.mask[idx].split()]
        mask = np.array(mask)
        txt2sub = [int(x) for x in self.txt2sub[idx].split()]
        txt2sub = np.array(txt2sub)
        sub2phn = [int(x) for x in self.sub2phn[idx].split()]
        sub2phn = np.array(sub2phn)
        encoding = [int(x) for x in self.encoding[idx].split()]
        encoding = np.array(encoding)
        word2sub = [int(x) for x in self.word2sub[idx].split()]
        word2sub = np.array(word2sub)
        space = [int(x) for x in self.space[idx].split()]
        space = np.array(space)
        sub2sub = [int(x) for x in self.sub2sub[idx].split()]
        sub2sub = np.array(sub2sub)
        #mel_path = os.path.join(
        #    self.data_path[idx], "mel", "{}-mel-{}.npy".format(self.dataset,basename))
        #mel_target = np.load(mel_path)
        D_path = os.path.join(
                            self.data_path[idx], "alignment", "{}-ali-{}.npy".format(self.dataset,basename))
        D = np.load(D_path)
        assert len(phone)==len(D), print(phone,D)
        wav_mask = []
        for i in range(len(phone)):
            if phone[i] == 9:
                wav_mask += D[i]*self.config.hop_length*[0]
            else:    
                wav_mask += D[i]*self.config.hop_length*[1]
        audio_path = os.path.join(
            self.data_path[idx], "audio", "{}-audio-{}.npy".format(self.dataset,basename))
        audio = np.load(audio_path)*np.asarray(wav_mask)
        spec_path = os.path.join(
            self.data_path[idx], "spec", "{}-spec-{}.npy".format(self.dataset,basename))
        mel_target = self.get_audio(audio, spec_path).T
        D_path = os.path.join(
            self.data_path[idx], "alignment", "{}-ali-{}.npy".format(self.dataset,basename))
        D = np.load(D_path)
        assert len(phone)==len(D), print(phone,D)
        assert sum(D)==mel_target.shape[0], print(sum(D),mel_target.shape)
        f0_path = os.path.join(
            self.data_path[idx], "f0", "{}-f0-{}.npy".format(self.dataset,basename))
        f0 = np.load(f0_path)
        #f0 = replace_outlier(f0,  self.f0_stat[0], self.f0_stat[1])
        #f0 = norm_mean_std(f0, self.f0_stat[2], self.f0_stat[3])
        energy_path = os.path.join(
            self.data_path[idx], "energy", "{}-energy-{}.npy".format(self.dataset,basename))
        energy = np.load(energy_path)
        #energy = replace_outlier(energy, self.energy_stat[0], self.energy_stat[1])
        #energy = norm_mean_std(energy, self.energy_stat[2], self.energy_stat[3])

        sample = {"id": basename,
                "sid": sid,
                "text": phone,
                "ptext": p_phone,
                "mask": mask,
                "mel_target": mel_target,
                "D": D,
                "f0": f0,
                "energy": energy,
                "audio": audio,
                "txt2sub": txt2sub,
                "sub2phn": sub2phn,
                "encoding": encoding,
                "word2sub": word2sub,
                "space": space,
                "sub2sub": sub2sub}
                
        return sample

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        sids = [batch[ind]["sid"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        ptexts = [batch[ind]["ptext"] for ind in cut_list]
        masks = [batch[ind]["mask"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        audios = [batch[ind]["audio"] for ind in cut_list]
        txt2subs = [batch[ind]["txt2sub"] for ind in cut_list]
        sub2phns = [batch[ind]["sub2phn"] for ind in cut_list]
        encodings = [batch[ind]["encoding"] for ind in cut_list]
        word2subs = [batch[ind]["word2sub"] for ind in cut_list]
        spaces = [batch[ind]["space"] for ind in cut_list]
        sub2subs = [batch[ind]["sub2sub"] for ind in cut_list]

        length_audio = np.array(list())
        for audio in audios:
            length_audio = np.append(length_audio, audio.shape[0])

        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_ptext = np.array(list())
        for text in ptexts:
            length_ptext = np.append(length_ptext, text.shape[0])

        length_subword = np.array(list())
        for text in encodings:
            length_subword = np.append(length_subword, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        length_word = np.array(list())
        for w2s in word2subs:
            length_word = np.append(length_word, w2s.shape[0])


        subword_len = np.array(list())
        for s2s in sub2subs:
            subword_len = np.append(subword_len, s2s.shape[0])
        

        texts = pad_1D(texts)
        ptexts = pad_1D(ptexts)
        masks = pad_1D(masks)
        mel_targets = pad_2D(mel_targets)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(pad_1D(Ds) + 1.)
        audios = pad_1D(audios)

        #Ds = pad_1D(Ds)
        #sub2phns = pad_1D(sub2phns)
        #word2subs = pad_1D(word2subs)
        Ds_m, Ds_e = ali_mask(Ds)
        sub2phns_m, sub2phns_e = ali_mask(sub2phns)
        word2subs_m, word2subs_e = ali_mask(word2subs)
        assert sub2phns_m.shape[1]==word2subs_m.shape[2],print(sub2phns_m.shape,word2subs_m.shape, ids)
        sent2word_m, sent2word_e = ali_mask([[int(x)] for x in length_word])

        spaces = pad_1D(spaces,PAD=-1)
        sub2subs = pad_1D(sub2subs, PAD=-1)
        txt2subs = pad_1D(txt2subs)
        
        encodings = pad_1D(encodings)

        out = {"id": ids,
               "sid": np.array(sids),
               "text": texts,
               "ptext": ptexts,
               "mask": masks,
               "mel_target": mel_targets,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel,
               "audio": audios,
               "audio_len": length_audio,
               "txt2sub": txt2subs,
               "txt": encodings,
               "txt_len": length_subword,
               "word_len": length_word,
               "space": spaces,
               "sub2sub": sub2subs,
               "sub_len": subword_len,

               "D_m": Ds_m,
               "sub2phn_m": sub2phns_m,
               "word2sub_m": word2subs_m,
               "sent2word_m": sent2word_m,
               "D_e": Ds_e,
               "sub2phn_e": sub2phns_e,
               "word2sub_e": word2subs_e,
               "sent2word_e": sent2word_e}
        
        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["mel_target"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        output = self.reprocess(batch, index_arr)

        text_padded = torch.LongTensor(output["text"])
        text_lengths = torch.LongTensor(output["src_len"])
        spec_padded = torch.FloatTensor(output["mel_target"]).permute(0, 2, 1)
        spec_lengths = torch.LongTensor(output["mel_len"])
        wav_padded = torch.FloatTensor(output["audio"]).unsqueeze(1)
        wav_lengths = torch.LongTensor(output["audio_len"])
        sid = torch.LongTensor(output["sid"])
        p_target = torch.FloatTensor(output["f0"])
        e_target = torch.FloatTensor(output["energy"])
        log_D = torch.FloatTensor(output["log_D"])
        txt2sub = torch.LongTensor(output["txt2sub"])
        txt = torch.LongTensor(output["txt"])
        txt_len = torch.LongTensor(output["txt_len"])
        word_len = torch.LongTensor(output["word_len"])
        sub2sub = torch.LongTensor(output['sub2sub'])
        sub_len = torch.LongTensor(output['sub_len'])


        d_target_m = torch.BoolTensor(output["D_m"])
        sub2phn_m = torch.BoolTensor(output["sub2phn_m"])
        word2sub_m = torch.BoolTensor(output["word2sub_m"])
        sent2word_m = torch.BoolTensor(output["sent2word_m"])

        d_target_e = torch.LongTensor(output["D_e"])
        sub2phn_e = torch.LongTensor(output["sub2phn_e"])
        word2sub_e = torch.LongTensor(output["word2sub_e"])
        sent2word_e = torch.LongTensor(output["sent2word_e"])


        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, d_target_m, d_target_e, p_target, e_target, log_D, txt2sub, sub2phn_m, sub2phn_e, txt, txt_len, word2sub_m, word2sub_e, word_len, sub2sub, sub_len, output['id'], sent2word_m, sent2word_e


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size


class EvalLoader(torch.utils.data.Dataset):
    def __init__(self, filename="train.txt.1"):
 
        self.text = []
        self.basename = []
        self.ptext = []
        self.mask = []
        self.sid = []
        self.clean_txt = []
        self.txt2sub = []
        self.sub2phn = []
        self.encoding = []
        self.word2sub = []
        self.sub2sub = []
        self.space = []
        with open(filename, "r", encoding="utf-8") as f: 
            for line in f.readlines():
                n, t, pt, m, ipa, txt, t2s, s2p, enc, _, w2s, sp, s2s  = line.strip('\n').split('|')[:13]
                self.basename.append(n)
                self.text.append(t)
                self.ptext.append(pt)
                self.mask.append(m)
                self.sid.append(0)
                self.clean_txt.append(txt)
                self.txt2sub.append(t2s)
                self.sub2phn.append(s2p)
                self.encoding.append(enc)
                self.word2sub.append(w2s)
                self.space.append(sp)
                self.sub2sub.append(s2s)
        self.sid_dict = self.create_speaker_table(self.sid)
        


        self.create_sid_to_index()

    def create_speaker_table(self, sids):
        speaker_ids = np.sort(np.unique(sids))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        return d

    def create_sid_to_index(self):
        _sid_to_indexes = {} 
        # for keeping instance indexes with the same speaker ids
        for i, sid in enumerate(self.sid):
            if sid in _sid_to_indexes:
                _sid_to_indexes[sid].append(i)
            else:
                _sid_to_indexes[sid] = [i]
        self.sid_to_indexes = _sid_to_indexes

    def __len__(self):
        return len(self.text)


    def __getitem__(self, idx):
        basename = self.basename[idx]
        sid = self.sid_dict[self.sid[idx]]
        print(self.text[idx])
        phone = np.array(text_to_sequence(self.text[idx], []))
        p_phone = np.array(text_to_sequence(self.ptext[idx], []))
        mask = [int(x) for x in self.mask[idx].split()]
        mask = np.array(mask)
        txt2sub = [int(x) for x in self.txt2sub[idx].split()]
        txt2sub = np.array(txt2sub)
        sub2phn = [int(x) for x in self.sub2phn[idx].split()]
        sub2phn = np.array(sub2phn)
        encoding = [int(x) for x in self.encoding[idx].split()]
        encoding = np.array(encoding)
        word2sub = [int(x) for x in self.word2sub[idx].split()]
        word2sub = np.array(word2sub)
        space = [int(x) for x in self.space[idx].split()]
        space = np.array(space)
        sub2sub = [int(x) for x in self.sub2sub[idx].split()]
        sub2sub = np.array(sub2sub)
 
       

        sample = {"id": basename,
                "sid": sid,
                "text": phone,
                "ptext": p_phone,
                "mask": mask,
                "txt2sub": txt2sub,
                "sub2phn": sub2phn,
                "encoding": encoding,
                "word2sub": word2sub,
                "space": space,
                "sub2sub": sub2sub}
                
        return sample

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        sids = [batch[ind]["sid"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        ptexts = [batch[ind]["ptext"] for ind in cut_list]
        masks = [batch[ind]["mask"] for ind in cut_list]
        txt2subs = [batch[ind]["txt2sub"] for ind in cut_list]
        sub2phns = [batch[ind]["sub2phn"] for ind in cut_list]
        encodings = [batch[ind]["encoding"] for ind in cut_list]
        word2subs = [batch[ind]["word2sub"] for ind in cut_list]
        spaces = [batch[ind]["space"] for ind in cut_list]
        sub2subs = [batch[ind]["sub2sub"] for ind in cut_list]


        
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_ptext = np.array(list())
        for text in ptexts:
            length_ptext = np.append(length_ptext, text.shape[0])

        length_subword = np.array(list())
        for text in encodings:
            length_subword = np.append(length_subword, text.shape[0])


        length_word = np.array(list())
        for w2s in word2subs:
            length_word = np.append(length_word, w2s.shape[0])


        subword_len = np.array(list())
        for s2s in sub2subs:
            subword_len = np.append(subword_len, s2s.shape[0])
        

        texts = pad_1D(texts)
        ptexts = pad_1D(ptexts)
        masks = pad_1D(masks)

        sub2phns_m, sub2phns_e = ali_mask(sub2phns)
        word2subs_m, word2subs_e = ali_mask(word2subs)
        assert sub2phns_m.shape[1]==word2subs_m.shape[2],print(sub2phns_m.shape,word2subs_m.shape)
        sent2word_m, sent2word_e = ali_mask([[int(x)] for x in length_word])

        spaces = pad_1D(spaces,PAD=-1)
        sub2subs = pad_1D(sub2subs, PAD=-1)
        txt2subs = pad_1D(txt2subs)
        
        encodings = pad_1D(encodings)

        out = {"id": ids,
               "sid": np.array(sids),
               "text": texts,
               "ptext": ptexts,
               "mask": masks,
               "src_len": length_text,
               "txt2sub": txt2subs,
               "txt": encodings,
               "txt_len": length_subword,
               "word_len": length_word,
               "space": spaces,
               "sub2sub": sub2subs,
               "sub_len": subword_len,

               "sub2phn_m": sub2phns_m,
               "word2sub_m": word2subs_m,
               "sent2word_m": sent2word_m,
               "sub2phn_e": sub2phns_e,
               "word2sub_e": word2subs_e,
               "sent2word_e": sent2word_e}
        
        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        output = self.reprocess(batch, index_arr)

        text_padded = torch.LongTensor(output["text"])
        text_lengths = torch.LongTensor(output["src_len"])
        
        sid = torch.LongTensor(output["sid"])
        
        txt2sub = torch.LongTensor(output["txt2sub"])
        txt = torch.LongTensor(output["txt"])
        txt_len = torch.LongTensor(output["txt_len"])
        word_len = torch.LongTensor(output["word_len"])
        sub2sub = torch.LongTensor(output['sub2sub'])
        sub_len = torch.LongTensor(output['sub_len'])


       
        sub2phn_m = torch.BoolTensor(output["sub2phn_m"])
        word2sub_m = torch.BoolTensor(output["word2sub_m"])
        sent2word_m = torch.BoolTensor(output["sent2word_m"])

       
        sub2phn_e = torch.LongTensor(output["sub2phn_e"])
        word2sub_e = torch.LongTensor(output["word2sub_e"])
        sent2word_e = torch.LongTensor(output["sent2word_e"])


        return text_padded, text_lengths, sid, txt2sub, sub2phn_m, sub2phn_e, txt, txt_len, word2sub_m, word2sub_e, word_len, sub2sub, sub_len, output['id'], sent2word_m, sent2word_e
