from distutils.command.config import config
import os
import json
import argparse
import itertools
import math
from random import gauss
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import numpy as np

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  Discriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from stft_loss import MultiResolutionSTFTLoss


torch.backends.cudnn.benchmark = True
global_step = 0

def anneal_weight(start_step, end_step, start_w, end_w, current_step):
  if current_step<start_step:
    return start_w
  elif current_step<end_step:
    return ((end_w-start_w)*(current_step-start_step)/(end_step-start_step))+start_w
  else:
    return end_w

def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2

def free_bit(kl, rate):
  #return torch.max(kl-rate, torch.zeros_like(kl))
  return torch.max(kl, torch.ones_like(kl)*rate)

def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80000'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method=hps.dist_url, world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerLoader(hps.data.data_path, 'train.txt', hps.data.speakerlist, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = train_dataset.collate_fn
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.data_path, 'val.txt', hps.data.speakerlist, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model,
      config=hps.config).cuda(rank)
  net_d = Discriminator(hps.discriminator).cuda(rank)
  resolutions = eval(hps.discriminator.mrd.resolutions)
  stft_criterion = MultiResolutionSTFTLoss(torch.device('cuda', rank), resolutions)

  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      #filter(lambda p: p.requires_grad, net_g.parameters()), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

  #_, _, _, epoch_str = utils.load_checkpoint_part(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, None)
  #_, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)

  #epoch_str = 1
  #global_step = 0

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    print(epoch_str)
    print('1111111111111111111111')
    #epoch_str = 1
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    print('222222222222')
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d, stft_criterion], [optim_g, optim_d], [scheduler_g,scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d, stft_criterion], [optim_g, optim_d], [scheduler_g,scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d, stft_criterion = nets
  optim_g, optim_d = optims
  scheduler_g,scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, d_target_m, d_target_e, p_target, e_target, log_D, txt2sub, sub2phn_m, sub2phn_e, txt, txt_len, word2sub_m, word2sub_e, word_len, sub2sub, sub_len, names, sent2word_m, sent2word_e) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    txt, txt_lengths =  txt.cuda(rank, non_blocking=True), txt_len.cuda(rank, non_blocking=True)
    txt2sub = txt2sub.cuda(rank, non_blocking=True)
    word_lengths = word_len.cuda(rank, non_blocking=True)
    sub2sub=sub2sub.cuda(rank, non_blocking=True)
    sub_len = sub_len.cuda(rank, non_blocking=True)

    speakers = speakers.cuda(rank, non_blocking=True)

    p_target = p_target.cuda(rank, non_blocking=True)
    e_target = e_target.cuda(rank, non_blocking=True)
    log_D = log_D.cuda(rank, non_blocking=True)

    d_target_m= d_target_m.cuda(rank, non_blocking=True)
    sub2phn_m = sub2phn_m.cuda(rank, non_blocking=True)
    word2sub_m = word2sub_m.cuda(rank, non_blocking=True)
    sent2word_m = sent2word_m.cuda(rank, non_blocking=True)

    d_target_e = d_target_e.cuda(rank, non_blocking=True)
    sub2phn_e = sub2phn_e.cuda(rank, non_blocking=True)
    word2sub_e = word2sub_e.cuda(rank, non_blocking=True)
    sent2word_e = sent2word_e.cuda(rank, non_blocking=True)



    kl_coeff = utils.kl_coeff(global_step, hps.train.total_step, hps.train.constant_step, hps.train.kl_const_coeff)
    kl_coeff = 1

    #frame_kl = anneal_weight(55000, 60000, hps.train.frame_kl, hps.train.frame_kl_e, global_step)
    #phn_kl = anneal_weight(60000, 63000, hps.train.phn_kl, hps.train.phn_kl_e, global_step)
    #subword_kl = anneal_weight(63000, 66000, hps.train.subword_kl, hps.train.subword_kl_e, global_step)
    #word_kl = anneal_weight(66000, 68000, hps.train.word_kl, hps.train.word_kl_e, global_step)
    #sent_kl = anneal_weight(68000, 70000, hps.train.sent_kl, hps.train.sent_kl_e, global_step)
    frame_kl = hps.train.frame_kl
    phn_kl = hps.train.phn_kl
    subword_kl = hps.train.subword_kl
    word_kl = hps.train.word_kl
    sent_kl = hps.train.sent_kl
    #print(x)
    #print(names)
    with autocast(enabled=hps.train.fp16_run):
      y_hat, ids_slice, kl_sent, kl_word, kl_subword, kl_phn, kl_frame, D_loss = \
        net_g.forward(x, x_lengths, spec, spec_lengths, txt, txt_lengths, txt2sub, speakers, d_target_m, d_target_e, sub2phn_m, sub2phn_e, word2sub_m, word2sub_e,sent2word_m, sent2word_e, word_len, sub2sub, sub_len, log_D,step=global_step)
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      #Mel loss
      mel = spec_to_mel_torch(
          spec,
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.mel_fmin,
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1).to(torch.float32),
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.hop_length,
          hps.data.win_length,
          hps.data.mel_fmin,
          hps.data.mel_fmax
      )

      loss_mel = F.l1_loss(y_mel, y_hat_mel)# * hps.train.c_mel
      sc_loss, mag_loss = stft_criterion(y_hat.squeeze(1).to(torch.float32), y.squeeze(1))
      loss_stft = (sc_loss + mag_loss) * hps.train.stft_lamb
      res_fake, period_fake = net_d(y_hat)
      with autocast(enabled=False):
          score_loss = 0.0
          for (_, score_fake) in res_fake + period_fake:
              score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
          loss_gen = score_loss / len(res_fake + period_fake)
          loss_gen_all = loss_gen + loss_stft + D_loss
          loss_gen_all = loss_gen_all + kl_coeff*(free_bit(kl_sent, 0.0)*sent_kl + \
                                                    free_bit(kl_word, 0.0)*word_kl + \
                                                    free_bit(kl_subword,0.0)*subword_kl+ \
                                                    free_bit(kl_phn,0.0)*phn_kl + \
                                                    free_bit(kl_frame,0.0)*frame_kl)
    optim_g.zero_grad()
    if not torch.any(torch.isnan(loss_gen_all)):
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
    else:
        print('jump opt')

    with autocast(enabled=hps.train.fp16_run):
      # Discriminator
      res_fake, period_fake = net_d(y_hat.detach())
      res_real, period_real = net_d(y)

      with autocast(enabled=False):
        loss_d = 0.0
        for (_, score_fake), (_, score_real) in zip(res_fake + period_fake, res_real + period_real):
            loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
            loss_d += torch.mean(torch.pow(score_fake, 2))

        loss_disc_all = loss_d / len(res_fake + period_fake)

    optim_d.zero_grad()
    if not torch.any(torch.isnan(loss_disc_all)):
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
    else:
        print('jump opt')

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc_all, loss_gen, loss_stft,loss_mel, D_loss]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [kl_coeff, global_step, lr])
        kls = [kl_sent, kl_word, kl_subword, kl_phn, kl_frame]
        logger.info([x.item() for x in kls])

        logger.info([sent_kl, word_kl, subword_kl, phn_kl, frame_kl ])


      if global_step % hps.train.eval_interval == 0:
        #evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, d_target, p_target, e_target, log_D) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)
        d_target = d_target.cuda()
        p_target = p_target.cuda()
        e_target = e_target.cuda()
        log_D = log_D.cuda()

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        speakers = speakers[:1]
        break
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len=1000, spec=spec, spec_lengths=spec_lengths)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }

    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
