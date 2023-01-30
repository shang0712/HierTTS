from email.policy import strict
import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
import torch.nn.functional as F

MATPLOTLIB_FLAG = False

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout)
logger = logging

def load_checkpoint_part(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    if k.split('.')[0]!='frame2wav':
      continue
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict, strict=False)
  else:
    model.load_state_dict(new_state_dict, strict=False)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    if True:
      new_state_dict[k] = saved_state_dict[k]
    else:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')
  
  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)
  
  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

from preprocessors.utils import language_mapping


def process_meta_multi(meta_path, hop_length, outlier_path=None):
    text = []
    if outlier_path is not None:
        outlier = [x.strip() for x in open(outlier_path,'r').readlines()]
    else:
        outlier = []
    name = []
    ptext = []
    mask = []
    identity = []
    data_path = []
    length = []
    clean_txt = []
    txt2sub = []
    sub2phn = []
    encoding = []
    word2sub = []
    sub2sub = []
    space = []
    for i in range(len(meta_path)):
        _meta_path = meta_path[i]
        with open(_meta_path, "r", encoding="utf-8") as f: 
            for line in f.readlines():
                n, t, pt, m, ipa, txt, t2s, s2p, enc, _, w2s, sp, s2s  = line.strip('\n').split('|')[:13]
                if n[0]=='e':
                    continue
                if '+zimu' in pt:
                    continue
                full_name = os.path.join(os.path.dirname(_meta_path),n)
                if full_name in outlier:
                    continue
                audio_path = os.path.join(os.path.split(_meta_path)[0], "audio", "{}-audio-{}.npy".format('multispeaker', n))
                audio = np.load(audio_path)
                fft_length = audio.shape[0]//hop_length
                if fft_length <120 or fft_length > 1000:
                  continue
                length.append(fft_length)
                name.append(n)
                text.append(t)
                ptext.append(pt)
                mask.append(m)
                identity.append(i)
                data_path.append(os.path.split(_meta_path)[0])
                clean_txt.append(txt)
                txt2sub.append(t2s)
                sub2phn.append(s2p)
                encoding.append(enc)
                word2sub.append(w2s)
                space.append(sp)
                sub2sub.append(s2s)

    return data_path, name, text, ptext, mask, identity, length, clean_txt, txt2sub, sub2phn, encoding, word2sub, space, sub2sub

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

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    fig.tight_layout()
    if titles is None:
        titles = [None for i in range(len(data))]
    for i in range(len(data)):
        spectrogram = data[i]
        axes[i][0].imshow(spectrogram, origin='lower')
        axes[i][0].set_aspect(2.5, adjustable='box')
        axes[i][0].set_ylim(0, 80)
        axes[i][0].set_title(titles[i], fontsize='medium')
        axes[i][0].tick_params(labelsize='x-small', left=False, labelleft=False) 
        axes[i][0].set_anchor('W')
    
    plt.savefig(filename, dpi=200)
    plt.close()


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).cuda()
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
    return mask

def ali_mask(inputs):
    B = len(inputs)
    t_len = max((len(x) for x in inputs))
    T_len = max([sum(x) for x in inputs])
    mask = np.ones((B, t_len, T_len))
    expand = np.zeros((B, T_len))
    for i in range(len(inputs)):
        cum=0
        cnt=1
        for j in range(len(inputs[i])):
            mask[i,j,cum:cum+inputs[i][j]]=0
            expand[i,cum:cum+inputs[i][j]]=cnt
            cum+=inputs[i][j]
            cnt+=1
    return mask, expand

def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
    

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, t_path)

def kl_coeff(step, total_step, constant_step, min_kl_coeff):
    return max(min((step - constant_step) / total_step, 1.0), min_kl_coeff)

def slope_inc(step, total_step, constant_step, min_end=-10, max_end=-0.5, num=5):
    rate =max((constant_step+total_step-step),0)/(total_step+constant_step)
    end=max_end-rate*(max_end-min_end)
    delta = (-end)/(num-1)
    return [end+i*delta for i in range(num)]
    

def kl_gain(kl, kl_ref):
    #return torch.abs(torch.max(kl, kl_ref.detach())-kl_ref.detach())
    return torch.abs(torch.max(kl, kl_ref.detach())-kl_ref.detach())# + kl_ref.detach()

def coeff(step, start, end, flat_ratio=0.5):
  slope_step = (end-start)*(1-flat_ratio)*0.5
  flat_step = (end-start)*flat_ratio
  if step< start or step>end:
    return 1
  elif  step> (start+slope_step) and step <(end-slope_step):
    return 0.1
  elif step<(start+slope_step):
    return  0.1+ 0.9*(1-(step-start)/slope_step)
  else:
    return 0.1 + 0.9*(step-(start+flat_step+slope_step))/slope_step

def coeff_adapt(kl_sent, kl_word, kl_subword, kl_phn, kl_frame, step):
  #word
  kl_sent = coeff(step, 7000, 8000, flat_ratio=0.5)*kl_sent
  kl_word = coeff(step, 5000, 6000, flat_ratio=0.5)*kl_word
  kl_subword = coeff(step, 3000, 4000, flat_ratio=0.5)*kl_subword
  kl_phn = coeff(step, 1000, 2000, flat_ratio=0.5)*kl_phn
  return kl_sent, kl_word, kl_subword, kl_phn, kl_frame
  
def kl_descent(kl, kl_rate):
    kl_cum = [None for _ in kl]
    kl_mat = [[0 for _ in kl] for _ in kl]
    #upper
    for i in range(len(kl)-1):
      for j in range(i+1, len(kl), 1):
        ref = 1.1*kl[i]*(kl_rate[i]/kl_rate[j])#.detach()
        delta = abs(max(kl[j],ref)-ref)/(len(kl)-1-i)
        if kl_cum[j] is None:
          kl_cum[j]=delta
        else:
          kl_cum[j]+=delta
        kl_mat[i][j]=delta.item()
    
    #lower
    for i in range(len(kl)-1,0,-1):
      for j in range(i-1, -1, -1):
        ref = 1.1*kl[i]*(kl_rate[i]/kl_rate[j])#.detach()
        delta = abs(max(kl[j],ref)-ref)
        if kl_cum[j] is None:
          kl_cum[j]=delta
        else:
          kl_cum[j]+=delta
        kl_mat[i][j]=delta.item()
    kl_mat = np.array(kl_mat)
    kl_all = None
    for i in range(len(kl_cum)):
      if kl_cum[i] is None:
        continue
      if kl_all is None:
        kl_all = kl_cum[i]
      else:
        kl_all += kl_cum[i]
    return kl_all
