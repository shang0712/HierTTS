import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from text import symbols
from submodules import get_sinusoid_encoding_table, ConvNorm, LinearNorm, MultiHeadAttention, Mish
import Constants as Constants
import torch.distributions as D
import torchaudio
#from hier_att import AttentionRNN_bkp as AttentionRNN
from hier_att import AttentionRNN
from transformers import BertModel
from layers import PQMF,RepConv

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
    return mask

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

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)



def get_attn_key_pad_mask(seq_k, seq_q, sep=None, mem_len=None, cache_bias=None):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    if cache_bias is not None:
        len_q += cache_bias
    if mem_len is not None:
        mem_mask = torch.ones((seq_k.size(0),mem_len), dtype=seq_k.dtype, device=seq_k.device)
        seq_k = torch.cat([mem_mask, seq_k],-1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    if sep is not None:
        B = padding_mask.shape[0]
        padding_mask = padding_mask.clone()
        sep=F.pad(torch.cumsum(sep,-1),[1,0,0,0],mode='constant', value=0)
        for i in range(B):
            padding_mask[i,:sep[i][-1],:]=1
            for start, end in zip(sep[i][:-1],sep[i][1:]):
                padding_mask[i,start:end,start:end]=0

    return padding_mask

class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model,d_inner,
                    n_head, d_k, d_v, fft_conv1d_kernel_size, dropout, angle=True,causal=False,deconline=False):

        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, angle=angle)
        #self.slf_attn = AFTLocalAutoregressive(d_model, 32, True)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, fft_conv1d_kernel_size = fft_conv1d_kernel_size, causal=causal, deconline=deconline)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None, mems=None, cache=None, bias0=0, bias1=0, lookahead=0):
        if mems is not None:
            c = torch.cat([mems, enc_input], 1)
            q, k, v = enc_input, c, c
            #end = max(0,mems.size(1)-4)
            #q = torch.cat([mems[:,end:,:], enc_input],1)
            #cache_bias = mems.size(1)-end
            cache_bias = 0 
        else:
            q, k, v = enc_input, enc_input, enc_input
            cache_bias = 0
           
        enc_output, enc_slf_attn = self.slf_attn(
            q, k, v, mask=slf_attn_mask, bias0=bias0-cache_bias, bias1=bias1)
        enc_output = enc_output[:,cache_bias:,:]
        enc_output *= non_pad_mask
        
        enc_output, cache = self.pos_ffn(enc_output, cache=cache, lookahead=lookahead)
        enc_output *= non_pad_mask

        if cache is not None:
            output = {'0':enc_output,'1':cache}
        else:
            output = enc_output
        return output, enc_slf_attn



class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, fft_conv1d_kernel_size, dropout=0.1, causal=False, deconline=False):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.mish = Mish()
        self.causal = causal
        self.deconline = deconline
        if causal or deconline:
            self.lorder0=fft_conv1d_kernel_size[0]-1
            self.lorder1=fft_conv1d_kernel_size[1]-1
            if deconline and not causal:
                self.lorder0=self.lorder0//2
                self.lorder1=self.lorder1//2
            self.w_1 = nn.Conv1d(
                    d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0], padding=0)
            # position-wise
            self.w_2 = nn.Conv1d(
                        d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1], padding=0)
        else:
            self.w_1 = nn.Conv1d(
                    d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0], padding=(fft_conv1d_kernel_size[0]-1)//2)
            # position-wise
            self.w_2 = nn.Conv1d(
                        d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1], padding=(fft_conv1d_kernel_size[1]-1)//2)


        self.layer_norm = nn.LayerNorm(d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cache=None, lookahead=0):
        if self.causal or self.deconline:                
            new_cache = list()
        else:
            new_cache = None

        if cache is not None:
            cache_0=cache[:,:,:self.lorder0]
            #cache_1=cache[:,:,self.lorder0:self.lorder0+self.lorder1]

        residual = x
        output = x.transpose(1, 2)
        if self.causal or self.deconline:
            if cache is None:
                output = nn.functional.pad(output, (self.lorder0, 0), 'constant', 0.0)
            else:
                output = torch.cat([cache_0,output],dim=2)
            if not self.causal:
                output = nn.functional.pad(output, (0, self.lorder0), 'constant', 0.0)
            new_cache.append(output[:,:,output.size(2)-self.lorder0-lookahead:output.size(2)-lookahead])
        output = F.relu(self.w_1(output))
        output = self.w_2(output)
        output = output.transpose(1, 2)
        output = self.dropout(output)


        output = self.layer_norm(output + residual)
   
        if new_cache is not None:
            new_cache = torch.cat(new_cache, -1)

        return output, new_cache

class Prenet(nn.Module):
    ''' Prenet '''
    def __init__(self, hidden_dim, out_dim, dropout):
        super(Prenet, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
        )
        self.fc = LinearNorm(hidden_dim, out_dim)

    def forward(self, input, mask=None):
        residual = input
        # convs
        output = input.transpose(1,2)
        output = self.convs(output)
        output = output.transpose(1,2)
        # fc & residual
        output = self.fc(output) + residual

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output

class FrameEnc(nn.Module):
  def __init__(self, input_size, filter_size, kernel_size, n_layers, dropout) -> None:
    super().__init__()
    convs = []
    convs.append(ConvNorm(input_size, filter_size, kernel_size))
    for _ in range(n_layers-1):
      convs.append(ConvNorm(filter_size, filter_size, kernel_size))
    self.convs = nn.ModuleList(convs)
    self.norm_layer = nn.LayerNorm(filter_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, t=None):
    if t is not None:
        x = torch.cat((x, t.unsqueeze(-1).repeat(1,1,x.size(2))),1)

    for conv in self.convs:
      x = conv(x)
      x = self.relu(x)
      x = self.norm_layer(x.permute(0,2,1)).permute(0,2,1)
      x = self.dropout(x)
    return x


class Phnembedding(nn.Module):
    def __init__(self, config, n_src_vocab=len(symbols)+1):
        super(Phnembedding, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.d_model = config.encoder_hidden
        self.dropout = config.dropout

        self.src_word_emb = nn.Embedding(n_src_vocab, self.d_model, padding_idx=Constants.PAD)
        #self.prenet = Prenet(self.d_model, self.d_model, self.dropout)

    def forward(self, src_seq, x_lengths):
        mask = get_mask_from_lengths(x_lengths, torch.max(x_lengths))
      
        # -- Forward
        # word embedding
        src_embedded = self.src_word_emb(src_seq)
        # prenet
        output = src_embedded #self.prenet(src_embedded, mask)
        
        return output.transpose(1,2)


class FFTmodule(nn.Module):
    """ Decoder """

    def __init__(self, config,input_dim, decoder_layer, decoder_hidden, decoder_head, fft_conv1d_filter_size, fft_conv1d_kernel_size, dropout, max_seq_len,deconline = False):
        super(FFTmodule, self).__init__()
        self.n_layers = decoder_layer
        self.d_model = decoder_hidden
        self.n_head = decoder_head
        self.d_k = decoder_hidden // decoder_head
        self.d_v = decoder_hidden // decoder_head
        self.d_inner = fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = fft_conv1d_kernel_size
        self.dropout = dropout
        self.decoder_input_dim = input_dim
        self.deconline = deconline




        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, fft_conv1d_kernel_size = fft_conv1d_kernel_size,dropout=dropout, deconline=deconline) for _ in range(decoder_layer)])

        self.frames_history = config.history_decoder
        self.frames_current = config.current_decoder
        self.frames_overlap = config.overlap_decoder
        self.future=False
 

    def reset_stream(self, history, current, overlap, future=False):
        self.frames_history = history
        self.frames_current = current
        self.frames_overlap = overlap
        self.future = future

    def _update_mems(self, hids, mems, current):
        mem_len = mems[0].size(1) if mems is not None else 0
        hid_len = hids[0].size(1)
        with torch.no_grad():
            new_mems = []
            end_idx = mem_len + min(current, hid_len)
            beg_idx = max(0, end_idx-self.frames_history)
            for i in range(len(hids)):
                with torch.no_grad():
                    cat = torch.cat([mems[i], hids[i]],dim=1) if mems is not None else hids[i]
                    new_mems.append(cat[:,beg_idx:end_idx].detach())
        return new_mems

    def forward(self, enc_seq, enc_pos, firstpackage=0):
        dec_slf_attn_list = []
        enc_pos = (~enc_pos).long()

        # -- Forward
        
        #================== offine  or online mode ==================#
        if self.deconline:
            #chunknize the input
            max_len = enc_seq.size(1)
            mems = None
            B = enc_seq.size(0)
            caches = None
            result = list()
 
            for i in range(math.ceil((max_len-firstpackage)/self.frames_current)+int(firstpackage!=0)):
                if i==0 and firstpackage!=0:
                    cur_output = enc_seq[:,:firstpackage+self.frames_overlap,:]
                    cur_enc_pos = enc_pos[:,:firstpackage+self.frames_overlap]
                    current_length = firstpackage
                else:
                    cur_output = enc_seq[:,(i-int(firstpackage!=0))*self.frames_current+firstpackage:(i+1-int(firstpackage!=0))*self.frames_current+self.frames_overlap+firstpackage,:]
                    cur_enc_pos = enc_pos[:,(i-int(firstpackage!=0))*self.frames_current+firstpackage:(i+1-int(firstpackage!=0))*self.frames_current+self.frames_overlap+firstpackage]
                    current_length = self.frames_current
                mem_len = mems[0].size(1) if  mems is not None else 0

                #cache_bias = mem_len-max(0,mem_len-4)
                cache_bias = 0

                #cur_enc_pos = enc_pos[:,i*self.frames_current-mem_len:(i+1)*self.frames_current+self.frames_overlap]

                slf_attn_mask = get_attn_key_pad_mask(seq_k=cur_enc_pos, seq_q=cur_enc_pos, mem_len=mem_len, cache_bias=cache_bias)
                #slf_attn_mask = get_attn_key_pad_mask(seq_k=cur_enc_pos, seq_q=cur_enc_pos)
                non_pad_mask = get_non_pad_mask(cur_enc_pos)
                #utils._plot_data([slf_attn_mask[0],slf_attn_mask[0]], '1.png')

                hidden = []
                new_cache = list()
                for idx in range(len(self.layer_stack)):
                    hidden.append(cur_output)
                    dec_layer = self.layer_stack[idx]
                    mems_idx = mems[idx] if mems is not None else None
                    cache_idx = caches[idx] if caches is not None else None
                    bias= mems_idx.size(1) if mems_idx is not None else 0
                    #bias0=self.frames_current*i
                    #bias1=bias0-bias
                    lookahead = max(0,cur_output.size(1)-current_length)
                    cur_output, dec_slf_attn = dec_layer(
                        cur_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        mems=mems_idx,
                        cache=cache_idx,
                        bias0=bias,
                        lookahead=lookahead)
                        #bias1=bias1)
                    new_cache.append(cur_output['1'])
                    cur_output = cur_output['0']
         
                    #cur_output = cur_output[:,mem_len:]

                mems = self._update_mems(hidden, mems, current_length)
                result.append(cur_output[:,:current_length,:])
                caches = new_cache if len(new_cache) else None
                
            dec_output = torch.cat(result, dim=1)
        else:
            dec_output = enc_seq
            # -- Prepare masks
            slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
            non_pad_mask = get_non_pad_mask(enc_pos)
            caches = None
            new_cache = list()

            for i in range(len(self.layer_stack)):
                dec_layer = self.layer_stack[i]
                cache_idx = caches[i] if caches is not None else None
                if self.future and i==0:
                    dec_output, dec_slf_attn = dec_layer(
                        dec_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask_future,
                        cache=cache_idx)
                else:
                    dec_output, dec_slf_attn = dec_layer(
                        dec_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        cache=cache_idx)


        #============================ end ============================#


        return dec_output

    def inference(self, cur_enc_seq, cur_enc_pos, return_attns=False, mems=None, current_length=None, caches=None, frame_beg=None):
        dec_slf_attn_list = []

        # -- Forward
        dec_output = cur_enc_seq

        #================== offine  or online mode ==================#
        if self.deconline:
            if frame_beg is not None:
                begin = max(frame_beg-self.frames_history,0)
                mem_len = frame_beg-begin
            else:
                with torch.no_grad():
                    mems = torch.chunk(mems,len(self.layer_stack),dim=-1) if mems is not None else None
                cache_start = (self.fft_conv1d_kernel_size[0]-1)//2 
                if mems is not None:
                    caches = [x[:,-cache_start:,:] for x in mems]
                    mems = [x[:,:-cache_start,:] for x in mems]
                mem_len = mems[0].size(1) if  mems is not None else 0
            slf_attn_mask = get_attn_key_pad_mask(seq_k=cur_enc_pos, seq_q=cur_enc_pos, mem_len=mem_len)
            non_pad_mask = get_non_pad_mask(cur_enc_pos)
            hidden = []
            new_cache = list()
            for idx in range(len(self.layer_stack)):
                hidden.append(dec_output)
                if current_length is None:
                    current_length = self.frames_current 

                dec_layer = self.layer_stack[idx]
                if frame_beg is not None:
                    mems_idx = mems[idx][:,begin:frame_beg].detach() if mems is not None else None
                else:
                    mems_idx = mems[idx] if mems is not None else None
                cache_idx = caches[idx].transpose(1, 2) if mems is not None else torch.zeros((dec_output.size(0),cache_start,dec_output.size(2))).to(device).transpose(1, 2)
                bias= mems_idx.size(1) if mems_idx is not None else 0
                lookahead = max(0,dec_output.size(1)-current_length)

                dec_output, dec_slf_attn = dec_layer(
                        dec_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        mems=mems_idx,
                        cache=cache_idx,
                        bias0=bias,
                        lookahead=lookahead)
                new_cache.append(dec_output['1'].transpose(1, 2))
                dec_output = dec_output['0']

                if return_attns:
                    dec_slf_attn_list += [dec_slf_attn]

            if frame_beg is not None:
                with torch.no_grad():
                    for k in range(len(mems)):
                        mems[k][:,frame_beg:frame_beg+current_length]=hidden[k][:,:current_length]
                for k in range(len(caches)):
                    caches[k]=new_cache[k]
                dec_output = dec_output[:,:current_length,:]  
            else:
                mems = self._update_mems(hidden, mems, current_length)
                with torch.no_grad():
                    mems = torch.cat(mems,-1)
                caches = torch.cat(new_cache,-1) if len(new_cache) else None
                mems = torch.cat([mems,caches],1)
                dec_output = dec_output[:,:current_length,:]  
            del new_cache
            del hidden
            del cache_idx
            del mems_idx

        else:
            # -- Prepare masks
            slf_attn_mask = get_attn_key_pad_mask(seq_k=cur_enc_pos, seq_q=cur_enc_pos)
            non_pad_mask = get_non_pad_mask(cur_enc_pos)


            for i in range(len(self.layer_stack)):
                if adaconv:
                    gen_layer = self.layer_generated[i]
                    dec_output = gen_layer(spk_emb, dec_output)
                dec_layer = self.layer_stack[i]
                dec_output, dec_slf_attn = dec_layer(
                    dec_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask
                    )
                if return_attns:
                    dec_slf_attn_list += [dec_slf_attn]
        #============================ end ============================#
        if frame_beg is not None:
            return dec_output, None
        else:
            return dec_output, mems


class VariancePredictor(nn.Module):
    """ Variance Predictor """
    def __init__(self, input_size, filter_size, kernel_size, output_size=1, n_layers=2, dropout=0.5):
        super(VariancePredictor, self).__init__()

        convs = [ConvNorm(input_size, filter_size, kernel_size)]
        for _ in range(n_layers-1):
            convs.append(ConvNorm(filter_size, filter_size, kernel_size))
        self.convs = nn.ModuleList(convs)
        self.lns = nn.ModuleList([nn.LayerNorm(filter_size) for _ in range(n_layers)])
        self.linear_layer = nn.Linear(filter_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        for conv, ln in zip(self.convs, self.lns):
            x = x.transpose(1,2)
            x = self.relu(conv(x))
            x = x.transpose(1,2)
            x = ln(x)
            x = self.dropout(x)

        out = self.linear_layer(x)

        return out.squeeze(-1)


class AdaptFFT(nn.Module):
  def __init__(self, config, input_dim, decoder_layer, decoder_hidden, decoder_head, condition_dim=0, deconline = False, fft=False, outdim=None) -> None:
      super().__init__()
      self.fft = fft
      self.length_regulator = LengthRegulator(input_dim, config.max_seq_len)

      if self.fft:
        self.prelinear = nn.Linear(input_dim+condition_dim, decoder_hidden)
        self.nonlinear = FFTmodule(
                        config,
                        decoder_hidden, 
                        decoder_layer=decoder_layer, 
                        decoder_hidden=decoder_hidden, 
                        decoder_head=decoder_head, 
                        fft_conv1d_filter_size=config.fft_conv1d_filter_size, 
                        fft_conv1d_kernel_size=config.fft_conv1d_kernel_size, 
                        dropout=config.dropout, 
                        max_seq_len=config.max_seq_len,
                        deconline = deconline)
        self.postlinear = nn.Linear(decoder_hidden, 1) if outdim is not None else None
      else:
        self.nonlinear = nn.Sequential(
            nn.Conv1d(input_dim+condition_dim, decoder_hidden, 3, padding=1),
            Mish(),
            nn.Conv1d(decoder_hidden, decoder_hidden, 3, padding=1),
            Mish()
        )

  def forward(self, x, ali, max_len=None, condition=None, sub2phn=False):
      # Length regulate
      output, _, mel_len = self.length_regulator(x.transpose(1,2), ali, max_len, sub2phn=sub2phn)
      if condition is not None:
        output = torch.cat([output, condition.transpose(1,2)], 2)
      if self.fft:
        mel_mask = get_mask_from_lengths(mel_len)
        output = self.prelinear(output)
        output = self.nonlinear(output, mel_mask).transpose(1,2)
        if self.postlinear:
            output = self.postlinear(output.transpose(1,2)).transpose(1,2)
      else:
        output = self.nonlinear(output.transpose(1,2))
      return output
#   def infer(self,output,mel_len):
#       if self.fft:
#         mel_mask = get_mask_from_lengths(mel_len)
#         output = self.prelinear(output)
#         output = self.nonlinear(output, mel_mask).transpose(1,2)
#         if self.postlinear:
#             output = self.postlinear(output.transpose(1,2)).transpose(1,2)
#       else:
#         output = self.nonlinear(output.transpose(1,2))
#       return output
  def deconline_infer(self,output,mel_mask,mem):

      output = self.prelinear(output)
      output,mem = self.nonlinear.inference(output, mel_mask,mems = mem)
  
      return output,mem
  def infer(self, x, ali, mel_mask=None, condition=None, sub2phn=False):
      # Length regulate
      output = torch.matmul(x,ali).transpose(1,2)
      if condition is not None:
        output = torch.cat([output, condition.transpose(1,2)], 2)
      if self.fft:
        #mel_mask = get_mask_from_lengths(mel_len)
        output = self.prelinear(output)
        output = self.nonlinear(output, mel_mask).transpose(1,2)
        if self.postlinear:
            output = self.postlinear(output.transpose(1,2)).transpose(1,2)
      else:
        output = self.nonlinear(output.transpose(1,2))
      return output
#   def deconline_infer(self,x, ali, mel_len=None, condition=None, sub2phn=False):
#       output = torch.matmul(x,ali).transpose(1,2)
#       mel_mask = get_mask_from_lengths(mel_len)
#       if condition is not None:
#         output = torch.cat([output, condition.transpose(1,2)], 2)
#       output = self.prelinear(output)
#       output,mem = self.nonlinear.inference(output, mel_mask,mems = mem)
  
#       return output,mem

class LengthRegulator(nn.Module):
    """ Length Regulator """
    def __init__(self, hidden_size, max_pos):
        super(LengthRegulator, self).__init__()
        self.hidden_size = hidden_size

    def LR(self, h, ali, max_len):
        h = F.pad(h, [0, 0, 1, 0])
        ali_ = ali[..., None].repeat([1, 1, h.shape[-1]])
        h = torch.gather(h, 1, ali_)  # [B, T, H]

        mel_len = list()
        for ali_ in ali:
            ali_ = [x.item() for x in ali_]
            if 0 in ali_:
                len_ = ali_.index(0)
            else:
                len_ = len(ali_)
            mel_len.append(len_)

        return h, None, torch.LongTensor(mel_len).to(h.device)


    
    def AS(self, x, ali, max_len):
        output = list()
        mel_len = list()
        for batch, batch_idx in zip(x, ali):
            expanded = self.assign(batch, batch_idx)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)
        return output, None, torch.LongTensor(mel_len).to(output.device)

    def expand(self, batch, predicted):
        out = list()
        pos = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
            pos.append(self.position_enc[:int(expand_size), :])
        out = torch.cat(out, 0)
        pos = torch.cat(pos, 0)
        return out, pos

    def assign(self, batch, batch_idx):
        out = list()
        for idx in batch_idx:
            if idx==-1:
                out.append(torch.zeros(1,self.hidden_size).to(batch))
            else:
                out.append(batch[idx].unsqueeze(0))
        out = torch.cat(out, 0)
        return out


    def forward(self, x, duration, max_len=None, sub2phn=False):
        if sub2phn:
            output, position, mel_len = self.AS(x, duration, max_len)
        else:
            output, position, mel_len = self.LR(x, duration, max_len)
        return output, position, mel_len
        

class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x

from typing import List, Tuple
class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        in_d: int,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
            sample_rate=16000
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=1, bias=conv_bias, padding=(k-1)//2)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            def make_lowpass():
                lowpass = torchaudio.transforms.Resample(sample_rate, sample_rate//stride, resampling_method='sinc_interpolation')
                #for param in lowpass.parameters():
                #  param.requires_grad = False
                return lowpass

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), make_lowpass(), nn.GELU())

        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 4, "invalid conv definition: " + str(cl)
            (dim, k, stride, sample_rate) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                    sample_rate=sample_rate
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        for conv in self.conv_layers:
            x = conv(x)

        return x

def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)

  def forward(self, x, x_mask, g=None):
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    return x * x_mask

class GaussProj(nn.Module):
  def __init__(self, hidden_channels, out_channels) -> None:
    super().__init__()

    self.out_channels = out_channels
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.softplus = CustomSoftplus()

  def forward(self, x, x_mask=None, temp=1):
    stats = self.proj(x)
    m, logs = torch.split(stats, self.out_channels, dim=1)
    if x_mask is not None:
      return m*x_mask, self.softplus(logs)*x_mask
    else:
      return m, self.softplus(logs)
  def infer(self, x, x_mask=None, temp=1):
    stats = self.proj(x)
    m, logs = torch.split(stats, self.out_channels, dim=1)
    if x_mask is not None:   
        return m*x_mask, torch.log(1+torch.exp(logs))*x_mask
    else:     
        return m, torch.log(1+torch.exp(logs))

class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the location-variable convolutions'''
    def __init__(
            self,
            cond_channels,
            conv_in_channels,
            conv_out_channels,
            conv_layers,
            conv_kernel_size=3,
            kpnet_hidden_channels=64,
            kpnet_conv_size=3,
            kpnet_dropout=0.0,
            kpnet_nonlinear_activation="LeakyReLU",
            kpnet_nonlinear_activation_params={"negative_slope":0.1},
        ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int): number of layers
        '''
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        kpnet_kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers   # l_w
        kpnet_bias_channels = conv_out_channels * conv_layers                                           # l_b

        self.input_conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=2, bias=True)),
            getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.residual_convs = nn.ModuleList()
        padding = (kpnet_conv_size - 1) // 2
        for _ in range(3):
            self.residual_convs.append(
                nn.Sequential(
                    nn.Dropout(kpnet_dropout),
                    nn.utils.weight_norm(nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True)),
                    getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                    nn.utils.weight_norm(nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True)),
                    getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                )
            )
        self.kernel_conv = nn.utils.weight_norm(
            nn.Conv1d(kpnet_hidden_channels, kpnet_kernel_channels, kpnet_conv_size, padding=padding, bias=True))
        self.bias_conv = nn.utils.weight_norm(
            nn.Conv1d(kpnet_hidden_channels, kpnet_bias_channels, kpnet_conv_size, padding=padding, bias=True))
        
    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        '''
        batch, _, cond_length = c.shape
        c = self.input_conv(c)
        for residual_conv in self.residual_convs:
            residual_conv.to(c.device)
            c = c + residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = k.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            cond_length,
        )
        bias = b.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_out_channels,
            cond_length,
        )

        return kernels, bias

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv[0])
        nn.utils.remove_weight_norm(self.kernel_conv)
        nn.utils.remove_weight_norm(self.bias_conv)
        for block in self.residual_convs:
            nn.utils.remove_weight_norm(block[1])
            nn.utils.remove_weight_norm(block[3])

class LVCBlock(torch.nn.Module):
    '''the location-variable convolutions'''
    def __init__(
            self,
            in_channels,
            cond_channels,
            stride,
            dilations=[1, 3, 9, 27],
            lReLU_slope=0.2,
            conv_kernel_size=3,
            cond_hop_length=256,
            kpnet_hidden_channels=64,
            kpnet_conv_size=3,
            kpnet_dropout=0.0,
        ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels, 
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
            kpnet_nonlinear_activation_params={"negative_slope":lReLU_slope}
        )
        
        self.convt_pre = nn.Sequential(
            nn.LeakyReLU(lReLU_slope),
            nn.utils.weight_norm(nn.ConvTranspose1d(in_channels, in_channels, 2 * stride, stride=stride, padding=stride // 2 + stride % 2, output_padding=stride % 2)),
        )
        
        self.conv_blocks = nn.ModuleList()
        for dilation in dilations:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(lReLU_slope),
                    nn.utils.weight_norm(nn.Conv1d(in_channels, in_channels, conv_kernel_size, padding=dilation * (conv_kernel_size - 1) // 2, dilation=dilation)),
                    nn.LeakyReLU(lReLU_slope),
                )
            )

    def forward(self, x, c):
        ''' forward propagation of the location-variable convolutions.
        Args: 
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        
        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        ''' 
        _, in_channels, _ = x.shape         # (B, c_g, L')
        
        x = self.convt_pre(x)               # (B, c_g, stride * L')
        kernels, bias = self.kernel_predictor(c)

        for i, conv in enumerate(self.conv_blocks):
            output = conv(x)                # (B, c_g, stride * L')
            
            k = kernels[:, i, :, :, :, :]   # (B, 2 * c_g, c_g, kernel_size, cond_length)
            b = bias[:, i, :, :]            # (B, 2 * c_g, cond_length)

            output = self.location_variable_convolution(output, k, b, hop_size=self.cond_hop_length)    # (B, 2 * c_g, stride * L'): LVC
            x = x + torch.sigmoid(output[ :, :in_channels, :]) * torch.tanh(output[:, in_channels:, :]) # (B, c_g, stride * L'): GAU
        
        return x
    
    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernl. 
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100. 
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length). 
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length) 
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length) 
            dilation (int): the dilation of convolution. 
            hop_size (int): the hop_size of the conditioning sequence. 
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, _, in_length = x.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)     # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)   # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation, dilation)     # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]          
        x = x.transpose(3, 4)                   # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)  
        x = x.unfold(4, kernel_size, 1)         # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)

        return o

    def remove_weight_norm(self):
        self.kernel_predictor.remove_weight_norm()
        nn.utils.remove_weight_norm(self.convt_pre[1])
        for block in self.conv_blocks:
            nn.utils.remove_weight_norm(block[1])

MAX_WAV_VALUE = 32768.0

class Generator(nn.Module):
    """UnivNet Generator"""
    def __init__(self, inter_channel, gen, gin_channels=0):
        super(Generator, self).__init__()
        self.mel_channel = inter_channel
        self.noise_dim = gen.noise_dim
        self.hop_length = gen.hop_length
        channel_size = gen.channel_size
        kpnet_conv_size = gen.kpnet_conv_size
        gin_channels = gin_channels

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in gen.strides:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    inter_channel,
                    stride=stride,
                    dilations=gen.dilations,
                    lReLU_slope=gen.lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )
        
        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(gen.noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(gen.lReLU_slope),
            nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

        if gin_channels != 0:
          self.cond = nn.Conv1d(gin_channels, self.mel_channel, 1)

    def forward(self, c, g=None):
        '''
        Args: 
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length) 
            z (Tensor): the noise sequence (batch, noise_dim, in_length)
        
        '''
        z = torch.randn(c.size(0), self.noise_dim, c.size(2)).to(c.device)


        if g is not None:
          c = c + self.cond(g)

        z = self.conv_pre(z)                # (B, c_g, L)

        for res_block in self.res_stack:
            res_block.to(z.device)
            z = res_block(z, c)             # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)               # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, c, g=None):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        #zero = torch.full((1, self.mel_channel, 10), -11.5129).to(c.device)
        #mel = torch.cat((c, zero), dim=2)

        if g is not None:
          c = c + self.cond(g)

        audio = self.forward(c, g=g)
        '''
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(self.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        '''

        return audio


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class ResBlock3(nn.Module):
    def __init__(self, h, channels, kernel_sizes=3, dilations=(1, 3, 5)):
        super(ResBlock3, self).__init__()
        self.h = h
        self.act = nn.LeakyReLU(0.1)
        self.convs1 = nn.ModuleList()
        #self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(RepConv(channels, kernel_sizes, dilation=dilation))
            #self.convs2 += [RepConv(channels, kernel_sizes, dilation=1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            x = self.act(x)
            x = self.convs1[idx](x)
            #x = self.act(x)
            #x = self.convs2[idx](x)
        return x

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            x = self.act(x)
            x = self.convs1[idx].inference(x)
            #x = self.act(x)
            #x = self.convs2[idx].inference(x)
        return x
      
    def remove_weight_norm(self):
        for l in self.convs1:
            l.convert_weight_bias()
        #for l in self.convs2:
        #    l.convert_weight_bias()


class Generator_hifi(torch.nn.Module):
    def __init__(self, h):
        super(Generator_hifi, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(192, h.upsample_initial_channel, 7, 1, padding=3))
        if h.resblock == '1':
            resblock = ResBlock1
        elif h.resblock == '2':
            resblock = ResBlock2
        else:
            resblock = ResBlock3

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(u//2 + u%2), output_padding=u%2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 4, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upsample_rates = h.upsample_rates
        self.pqmf = PQMF()

    def forward(self, x, mapping_layers=[]):

        ret_acts = {}

        x = self.conv_pre(x)

        if 'pre' in mapping_layers:
            ret_acts['pre'] = x
        for i in range(self.num_upsamples):

            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                #xs_res = self.resblocks[i*self.num_kernels+j].inference(x)
                xs_res = self.resblocks[i*self.num_kernels+j](x)
                if xs is None:
                    xs = xs_res
                else:
                    xs += xs_res
                if 'res_{}_{}'.format(i,j) in mapping_layers:
                    ret_acts['res_{}_{}'.format(i,j)] = xs_res
                xs = F.leaky_relu(xs, LRELU_SLOPE)
                #xs = F.leaky_relu(xs)
            x = xs / self.num_kernels
            if 'up_{}'.format(i) in mapping_layers:
                ret_acts['up_{}'.format(i)] = xs
        #x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        x = self.pqmf.synthesis(x)
        if len(mapping_layers) == 0:
            return x
        else:
            return x, ret_acts

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
    def inference(self, c):
        audio = self.forward(c)
        return audio

class DiscriminatorR(torch.nn.Module):
    def __init__(self, hp, resolution):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        self.LRELU_SLOPE = hp.mpd.lReLU_slope

        norm_f = weight_norm if hp.mrd.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap,x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1).to(torch.float32)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False) #[B, F, TT, 2]
        mag = torch.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self, hp):
        super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = eval(hp.mrd.resolutions)
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(hp, resolution) for resolution in self.resolutions]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]

class DiscriminatorS(torch.nn.Module):
    def __init__(self, hp):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if hp.mpd.use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap,x

class DiscriminatorP(nn.Module):
    def __init__(self, hp, period):
        super(DiscriminatorP, self).__init__()

        self.LRELU_SLOPE = hp.mpd.lReLU_slope
        self.period = period

        kernel_size = hp.mpd.kernel_size
        stride = hp.mpd.stride
        norm_f = weight_norm if hp.mpd.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 64, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return  fmap,x


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, hp):
        super(MultiPeriodDiscriminator, self).__init__()

        #discs = [DiscriminatorS(hp)]
        discs = [DiscriminatorP(hp, period) for period in hp.mpd.periods]

        self.discriminators = nn.ModuleList(discs)

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]

class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)

    def forward(self, x):
        return self.MRD(x), self.MPD(x)

class EncCombinerCell(nn.Module):
    def __init__(self, Cin, Cout, type='default'):
        super(EncCombinerCell, self).__init__()
        # Cin = Cin1 + Cin2
        self.type = type
        if self.type=='default':
            self.conv = ConvNorm(Cin, Cout, 1)

    def forward(self, x1, x2):
        if self.type=='default':
            x2 = self.conv(x2)
            out = x1 + x2
        elif self.type=='nocomb':
            out = x1
        return out

# original combiner
class DecCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, method='concat'):
        super(DecCombinerCell, self).__init__()
        self.method = method
        if self.method=='nocomb':
          self.conv = ConvNorm(Cin2, Cout, kernel_size=1)
        elif self.method=='add':
          self.conv_0 = ConvNorm(Cin1, Cin2, kernel_size=1)
          self.conv = ConvNorm(Cin2, Cout, kernel_size=1)
        else:
          self.conv = ConvNorm(Cin1 + Cin2, Cout, kernel_size=1)

    def forward(self, x1, x2):
        if self.method=='nocomb':
          out = x2
        elif self.method=='add':
          out = self.conv_0(x1)+x2
        else:
          out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out

class Softplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result=torch.log(1+torch.exp(i))
        ctx.save_for_backward(i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*torch.sigmoid(ctx.saved_variables[0])

class CustomSoftplus(nn.Module):
    def forward(self, input_tensor):
        return Softplus.apply(input_tensor)

class Character(nn.Module):
    def __init__(self):
        super(Character, self).__init__()
        self.emb =  nn.Embedding(8021, 312) 

    def forward(self, x, mask):
        x = self.emb(x)
        return x, x

class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    gen,
    n_speakers=0,
    gin_channels=0,
    use_sdp=True,
    config=None,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gen = gen
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels
    self.dec_comb = config.dec_comb if config is not None and 'dec_comb' in config  else 'concat'
    self.enc_comb = config.enc_comb if config is not None and 'enc_comb' in config  else 'default'    
    self.use_sdp = use_sdp
    self.pooling = config.pooling if config is not None and 'pooling' in config else 'attention'
    self.residual = config.residual if config is not None and 'residual' in config else True
    self.bert_detach = config.bert_detach if config is not None and 'bert_detach' in config else True
    self.zero_txt = config.zero_txt if config is not None and 'zero_txt' in config else False
    self.straight_phn = config.straight_phn if config is not None and 'straight_phn' in config else True
    self.txt_cond = config.txt_cond if config is not None and 'txt_cond' in config else False
    self.bert = config.bert if config is not None and 'bert' in config else False
    self.punc_context = config.punc_context if config is not None and 'punc_context' in config else False



    # text unet
    self.txt_emb = Character() if self.bert else BertModel.from_pretrained(r"tiny_bert")
    if self.punc_context:
        self.txt_context = nn.Sequential(
                nn.Conv1d(312, 312, 3, padding=1),
                Mish(),
                nn.Conv1d(312, 312, 3, padding=1),
                Mish())

    self.subword_enc = nn.Conv1d(344, 128, 1)
    self.phn_enc = AttentionRNN(embed_size=config.encoder_hidden, gru_hidden=32, bidirectional=True, pooling_mode=self.pooling, linear=True)
    self.word_enc = AttentionRNN(embed_size=128, gru_hidden=128, bidirectional=True, pooling_mode=self.pooling, linear=True)
    self.sent_enc = AttentionRNN(embed_size=128, gru_hidden=128, bidirectional=True, pooling_mode=self.pooling, linear=True)
    self.length_regulator = LengthRegulator(312, config.max_seq_len)
    
    #wav2frame
    self.wav2frame = PosteriorEncoder(spec_channels, inter_channels.frame, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.wav2frame_comb = EncCombinerCell(config.decoder_hidden.phn2frame, hidden_channels, type=self.enc_comb)
    self.wav2frame_proj = GaussProj(hidden_channels, inter_channels.frame)
    self.frame2wav = Generator(inter_channels.frame, gen, gin_channels)
    #self.frame2wav = Generator_hifi(gen)
    self.frame2wav_comb = DecCombinerCell(config.decoder_hidden.phn2frame, inter_channels.frame, inter_channels.frame, self.dec_comb)
    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)
    
    #frame2phn
    self.phn_emb = Phnembedding(config)
    self.frame2phn_0 = FrameEnc(192, 256, 3, 3, 0.5)
    self.frame2phn_1 = AttentionRNN(embed_size=256, gru_hidden=128, bidirectional=True, pooling_mode=self.pooling)
    self.frame2phn_comb = EncCombinerCell(config.decoder_hidden.subword2phn, 256, type=self.enc_comb)
    self.frame2phn_proj = GaussProj(256, inter_channels.phn)
    self.phn2frame = AdaptFFT(config, inter_channels.phn, config.decoder_layer.phn2frame, config.decoder_hidden.phn2frame, config.decoder_head.phn2frame, deconline=config.deconline,fft=True)
    self.phn2frame_comb = DecCombinerCell(config.encoder_hidden, inter_channels.phn, inter_channels.phn, self.dec_comb) if self.straight_phn else DecCombinerCell(config.decoder_hidden.subword2phn, inter_channels.phn, inter_channels.phn, self.dec_comb) 
    self.phn2frame_proj = GaussProj(config.decoder_hidden.phn2frame, inter_channels.frame)
    self.out = nn.Linear(256,80)
    
    #phn2subword
    self.phn2subword = AttentionRNN(embed_size=256+config.encoder_hidden+1, gru_hidden=128, bidirectional=True, pooling_mode=self.pooling) if self.txt_cond else AttentionRNN(embed_size=256, gru_hidden=128, bidirectional=True, pooling_mode=self.pooling)
    self.phn2subword_comb = EncCombinerCell(config.decoder_hidden.word2subword, 256, type=self.enc_comb)
    self.phn2subword_proj = GaussProj(256, inter_channels.subword)
    self.subword2phn = AdaptFFT(config, inter_channels.subword, config.decoder_layer.subword2phn, config.decoder_hidden.subword2phn, config.decoder_head.subword2phn, condition_dim=config.encoder_hidden, fft=True)
    self.subword2phn_comb = DecCombinerCell(256, inter_channels.subword, inter_channels.subword, self.dec_comb)
    self.subword2phn_proj = GaussProj(config.decoder_hidden.subword2phn, inter_channels.phn)
    self.dur_linear = nn.Linear(config.decoder_hidden.subword2phn, 1)

    #subword2word
    self.subword2word = AttentionRNN(embed_size=256+128, gru_hidden=128, bidirectional=True, pooling_mode=self.pooling) if self.txt_cond else AttentionRNN(embed_size=256, gru_hidden=128, bidirectional=True, pooling_mode=self.pooling)
    self.subword2word_comb = EncCombinerCell(config.decoder_hidden.sent2word, 256, type=self.enc_comb)
    self.subword2word_proj = GaussProj(256, inter_channels.word)
    self.word2subword = AdaptFFT(config, inter_channels.word, config.decoder_layer.word2subword, config.decoder_hidden.word2subword, config.decoder_head.word2subword, condition_dim=128, fft=True)
    self.word2subword_comb = DecCombinerCell(config.decoder_hidden.sent2word, inter_channels.word, inter_channels.word, self.dec_comb)
    self.word2subword_proj = GaussProj(config.decoder_hidden.word2subword, inter_channels.subword)

    #word2sent
    self.word2sent = AttentionRNN(embed_size=256+128, gru_hidden=128, bidirectional=True, pooling_mode=self.pooling) if self.txt_cond else AttentionRNN(embed_size=256, gru_hidden=128, bidirectional=True, pooling_mode=self.pooling)
    self.word2sent_comb = EncCombinerCell(128, 256, type=self.enc_comb)
    self.word2sent_proj = GaussProj(256, inter_channels.sent)
    self.sent2word = AdaptFFT(config, inter_channels.sent, config.decoder_layer.sent2word, config.decoder_hidden.sent2word, config.decoder_head.sent2word, condition_dim=128, fft=True)
    self.sent2word_comb = DecCombinerCell(128, inter_channels.sent, inter_channels.sent, self.dec_comb)
    self.sent2word_proj = GaussProj(config.decoder_hidden.sent2word, inter_channels.word)

    #subword2word
    self.sent_proj = GaussProj(128, inter_channels.word)
    self.mse_loss = nn.MSELoss()
    self.expand = LengthRegulator(inter_channels.phn, config.max_seq_len)
  def skipbymask(self, x, skip_mask):
    #skip mask
    skip_output = list()
    arrange = torch.arange(skip_mask.shape[1]).to(skip_mask.device)
    for i, vec in enumerate(x):
        index = arrange[torch.eq(skip_mask[i],1)]
        temp = torch.index_select(vec,0,index)
        skip_output.append(temp)
    skip_output = pad(skip_output)
    return skip_output
  def length_expand(self,x, ali, max_len=None, condition=None, sub2phn=False):
    output, _, mel_len = self.length_regulator(x.transpose(1,2), ali, max_len, sub2phn=sub2phn)
    if condition is not None:
        output = torch.cat([output, condition.transpose(1,2)], 2)
    return output,mel_len
  def forward(self, phn, phn_lengths, y, y_lengths, txt, txt_lengths, txt2sub, sid=None, phn2frame_ali_m=None, phn2frame_ali=None, subword2phn_ali_m=None, subword2phn_ali=None, word2subword_ali_m=None, word2subword_ali=None,  sent2word_ali_m=None, sent2word_ali=None, word_lengths=None, sub2sub=None, subword_lengths=None, log_D=None,step=None):
    frame_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
    phn_mask = torch.unsqueeze(commons.sequence_mask(phn_lengths, phn.size(1)), 1).to(y.dtype)
    subword_mask = torch.unsqueeze(commons.sequence_mask(subword_lengths, torch.max(subword_lengths)), 1).to(y.dtype)
    word_mask = torch.unsqueeze(commons.sequence_mask(word_lengths, torch.max(word_lengths)), 1).to(y.dtype)

    if self.n_speakers > 1:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    phn_embeded = self.phn_emb(phn, phn_lengths)
    #text enc
    txt_embeded = self.txt_emb(txt, torch.ne(txt, 0))[0].detach() if self.bert_detach else self.txt_emb(txt, torch.ne(txt, 0))[0]
    if self.punc_context:
        txt_embeded = self.txt_context(txt_embeded.transpose(1,2)).transpose(1,2)
    txt_embeded = self.skipbymask(txt_embeded, txt2sub)
    txt_embeded, _, _ = self.length_regulator(txt_embeded, sub2sub, sub2phn=True)
    txt_embeded = torch.cat([txt_embeded,self.phn_enc(phn_embeded, subword2phn_ali_m).transpose(1,2)],-1)
    subword_embeded = self.subword_enc(txt_embeded.transpose(1,2))
    word_embeded = self.word_enc(subword_embeded, word2subword_ali_m)
    sent_embedded = self.sent_enc(word_embeded, sent2word_ali_m)

    #audio enc
    frame_reps = self.wav2frame(y, frame_mask, g=g)
    phn_reps = self.frame2phn_1(self.frame2phn_0(frame_reps), phn2frame_ali_m)
    subword_reps = self.phn2subword(torch.cat([phn_reps, phn_embeded, torch.sum((~phn2frame_ali_m).float(),2).unsqueeze(1)],1), subword2phn_ali_m) if self.txt_cond else self.phn2subword(phn_reps, subword2phn_ali_m) #subword_ali unnique
    word_reps = self.subword2word(torch.cat([subword_reps, subword_embeded],1), word2subword_ali_m)
    sent_reps = self.word2sent(torch.cat([word_reps,word_embeded],1), sent2word_ali_m)

    #sent
    s = sent_embedded
    m_sent_p, std_sent_p  = self.sent_proj(s)
    prior_sent = D.Normal(m_sent_p, std_sent_p)# if self.zero_txt else D.Normal(m_sent_p, std_sent_p)

    #sent2word
    ftr = self.word2sent_comb(sent_reps, s)
    m_sent_res, std_sent_res = self.word2sent_proj(ftr)
    posterior_sent = D.Normal(m_sent_p+m_sent_res, std_sent_p*std_sent_res) if False else D.Normal(m_sent_res, std_sent_res)
    z_sent = posterior_sent.rsample()
    kl_sent = D.kl.kl_divergence(posterior_sent, prior_sent)
    #kl_sent = torch.sum(kl_sent*x_mask)/torch.sum(x_mask)
    kl_sent = torch.sum(kl_sent, dim=[1,2])
    s = self.sent2word_comb(s, z_sent)
    s = self.sent2word(s, sent2word_ali, condition=word_embeded)

    m_word_p, std_word_p = self.sent2word_proj(s)
    prior_word = D.Normal(m_word_p, std_word_p)

    #word2subword
    ftr = self.subword2word_comb(word_reps, s)
    m_word_res, std_word_res = self.subword2word_proj(ftr)
    posterior_word = D.Normal(m_word_p+m_word_res,std_word_p*std_word_res) if self.residual else D.Normal(m_word_res, std_word_res)
    z_word = posterior_word.rsample()
    kl_word = D.kl.kl_divergence(posterior_word, prior_word)
    kl_word = torch.sum(kl_word*word_mask, dim=[1,2])/torch.sum(word_mask, dim=[1,2])
    s = self.word2subword_comb(s, z_word)
    s = self.word2subword(s, word2subword_ali, condition=subword_embeded)

    m_subword_p, std_subword_p = self.word2subword_proj(s)
    prior_subword = D.Normal(m_subword_p, std_subword_p)

    #subword2phn
    ftr = self.phn2subword_comb(subword_reps, s) 
    m_subword_res, std_subword_res = self.phn2subword_proj(ftr)
    posterior_subword = D.Normal(m_subword_p+m_subword_res,std_subword_p*std_subword_res) if self.residual else D.Normal(m_subword_res, std_subword_res)
    z_subword = posterior_subword.rsample()
    #z_subword = posterior_subword.rsample() if step is None or  step<80000 else prior_subword.rsample()
    kl_subword = D.kl.kl_divergence(posterior_subword, prior_subword)
    kl_subword = torch.sum(kl_subword*subword_mask, dim=[1,2])/torch.sum(subword_mask, dim=[1,2])
    s = self.subword2phn_comb(s, z_subword) 
    s = self.subword2phn(s, subword2phn_ali, condition=phn_embeded) #subword_ali unnique
    
    log_d_predicted = self.dur_linear(s.transpose(1,2)).squeeze(2) #subword_ali unnique

    d_loss = 0
    #r1 = 1.05
    for b, src_l in enumerate(phn_lengths):
        #d_loss += self.mse_loss(log_d_predicted[b, :src_l], (log_D[b, :src_l]+torch.log(torch.ones_like(log_D[b, :src_l])*r1)).detach())
        d_loss += self.mse_loss(log_d_predicted[b, :src_l], log_D[b, :src_l].detach())
    d_loss/=phn_lengths.size(0)

    m_phn_p, std_phn_p = self.subword2phn_proj(s)
    prior_phn = D.Normal(m_phn_p, std_phn_p)

    #phn2frame
    ftr = self.frame2phn_comb(phn_reps, s)
    m_phn_res, std_phn_res = self.frame2phn_proj(ftr)
    posterior_phn = D.Normal(m_phn_p+m_phn_res, std_phn_p*std_phn_res) if self.residual else D.Normal(m_phn_res, std_phn_res)
    z_phn = posterior_phn.rsample()
    #z_phn = posterior_phn.rsample()  if step is None or  step<77000 else prior_phn.rsample()
    kl_phn = D.kl.kl_divergence(posterior_phn, prior_phn)
    kl_phn = torch.sum(kl_phn*phn_mask, dim=[1,2])/torch.sum(phn_mask,dim=[1,2])
    s = self.phn2frame_comb(phn_embeded, z_phn) if self.straight_phn else self.phn2frame_comb(s, z_phn)
    s = self.phn2frame(s, phn2frame_ali)
    # s = s.transpose(1,2)
    # o = self.out(s).transpose(1,2)
    m_frame_p, std_frame_p = self.phn2frame_proj(s)
     
    prior_frame = D.Normal(m_frame_p, std_frame_p)

    #frame2wav
    ftr = self.wav2frame_comb(frame_reps, s)
    m_frame_res, std_frame_res = self.wav2frame_proj(ftr)
    posterior_frame = D.Normal(m_frame_p+m_frame_res, std_frame_p*std_frame_res) if self.residual else D.Normal(m_frame_res, std_frame_res)
    z_frame = posterior_frame.rsample()
    #z_frame = posterior_frame.rsample() if step is None or  step<70000 else prior_frame.rsample()
    kl_frame = D.kl.kl_divergence(posterior_frame, prior_frame)
    kl_frame = torch.sum(kl_frame*frame_mask, dim=[1,2])/torch.sum(frame_mask, dim=[1,2])
    s = self.frame2wav_comb(s, z_frame)
    

    s_slice, ids_slice = commons.rand_slice_segments(s, y_lengths, self.segment_size)

    o = self.frame2wav(s_slice )

    return o, ids_slice, torch.mean(kl_sent), torch.mean(kl_word), torch.mean(kl_subword),torch.mean(kl_phn),torch.mean(kl_frame), d_loss

    
  def emb_infer(self, phn, phn_lengths, y, y_lengths, txt, txt_lengths, txt2sub, sid=None, phn2frame_ali=None, subword2phn_ali=None, word2subword_ali=None, word_lengths=None, sub2sub=None, subword_lengths=None, log_D=None):
    #frame_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
    phn_mask = torch.unsqueeze(commons.sequence_mask(phn_lengths, phn.size(1)), 1).to(y.dtype)
    subword_mask = torch.unsqueeze(commons.sequence_mask(subword_lengths, torch.max(subword_lengths)), 1).to(y.dtype)
    word_mask = torch.unsqueeze(commons.sequence_mask(word_lengths, torch.max(word_lengths)), 1).to(y.dtype)
    sent2word_ali=word_lengths.unsqueeze(1) #改

    if self.n_speakers > 1:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    phn_embeded = self.phn_emb(phn, phn_lengths)
    #text enc
    txt_embeded = self.txt_emb(txt, torch.ne(txt, 0))[0].detach() if self.bert_detach else self.txt_emb(txt, torch.ne(txt, 0))[0]
    txt_embeded = self.skipbymask(txt_embeded, txt2sub)
    txt_embeded, _, _ = self.length_regulator(txt_embeded, sub2sub, sub2phn=True)
    txt_embeded = torch.cat([txt_embeded,self.phn_enc(phn_embeded, subword2phn_ali).transpose(1,2)],-1)
    subword_embeded = self.subword_enc(txt_embeded.transpose(1,2))
    word_embeded = self.word_enc(subword_embeded, word2subword_ali)
    sent_embedded = self.sent_enc(word_embeded, sent2word_ali)

    if self.zero_txt:
        subword_embeded = torch.zeros_like(subword_embeded)
        word_embeded = torch.zeros_like(word_embeded)
        sent_embedded = torch.zeros_like(sent_embedded)

    #audio enc
    #frame_reps = self.wav2frame(y, frame_mask, g=g)
    phn_reps = self.frame2phn_1(self.frame2phn_0(y), phn2frame_ali)
    subword_reps = self.phn2subword(torch.cat([phn_reps, phn_embeded, phn2frame_ali.unsqueeze(1)],1), subword2phn_ali) if self.txt_cond else self.phn2subword(phn_reps, subword2phn_ali) #subword_ali unnique
    word_reps = self.subword2word(torch.cat([subword_reps, subword_embeded],1), word2subword_ali)
    sent_reps = self.word2sent(torch.cat([word_reps,word_embeded],1), sent2word_ali)

    #sent
    s = sent_embedded
    m_sent_p, std_sent_p  = self.sent_proj(s)
    prior_sent = D.Normal(m_sent_p, std_sent_p)# if self.zero_txt else D.Normal(m_sent_p, std_sent_p)
   
    #sent2word
    ftr = self.word2sent_comb(sent_reps, s)
    m_sent_res, std_sent_res = self.word2sent_proj(ftr)
    posterior_sent = D.Normal(m_sent_p+m_sent_res, std_sent_p*std_sent_res) if False else D.Normal(m_sent_res, std_sent_res)
    z_sent = posterior_sent.rsample()
    kl_sent = D.kl.kl_divergence(posterior_sent, prior_sent)
    #kl_sent = torch.sum(kl_sent*x_mask)/torch.sum(x_mask)
    kl_sent = torch.sum(kl_sent, dim=[1,2])
    s = self.sent2word_comb(s, z_sent) 
    s = self.sent2word(s, sent2word_ali, condition=word_embeded)

    m_word_p, std_word_p = self.sent2word_proj(s)
    prior_word = D.Normal(m_word_p, std_word_p)

    #word2subword
    ftr = self.subword2word_comb(word_reps, s)
    m_word_res, std_word_res = self.subword2word_proj(ftr)
    posterior_word = D.Normal(m_word_p+m_word_res,std_word_p*std_word_res) if self.residual else D.Normal(m_word_res, std_word_res)
    z_word = posterior_word.rsample()
    kl_word = D.kl.kl_divergence(posterior_word, prior_word)
    kl_word = torch.sum(kl_word*word_mask, dim=[1,2])/torch.sum(word_mask, dim=[1,2])
    s = self.word2subword_comb(s, z_word) 
    s = self.word2subword(s, word2subword_ali, condition=subword_embeded)

    m_subword_p, std_subword_p = self.word2subword_proj(s)
    prior_subword = D.Normal(m_subword_p, std_subword_p)

    #subword2phn
    ftr = self.phn2subword_comb(subword_reps, s) 
    m_subword_res, std_subword_res = self.phn2subword_proj(ftr)
    posterior_subword = D.Normal(m_subword_p+m_subword_res,std_subword_p*std_subword_res) if self.residual else D.Normal(m_subword_res, std_subword_res)
    z_subword = posterior_subword.rsample()
    kl_subword = D.kl.kl_divergence(posterior_subword, prior_subword)
    kl_subword = torch.sum(kl_subword*subword_mask, dim=[1,2])/torch.sum(subword_mask, dim=[1,2])
    s = self.subword2phn_comb(s, z_subword) 
    s = self.subword2phn(s, subword2phn_ali, condition=phn_embeded) #subword_ali unnique
    
    log_d_predicted = self.dur_linear(s.transpose(1,2)).squeeze(2) #subword_ali unnique

    d_loss = 0
    for b, src_l in enumerate(phn_lengths):
        d_loss += self.mse_loss(log_d_predicted[b, :src_l], log_D[b, :src_l].detach())
    d_loss/=phn_lengths.size(0)

    m_phn_p, std_phn_p = self.subword2phn_proj(s)
    prior_phn = D.Normal(m_phn_p, std_phn_p)

    #phn2frame
    ftr = self.frame2phn_comb(phn_reps, s)
    m_phn_res, std_phn_res = self.frame2phn_proj(ftr)
    posterior_phn = D.Normal(m_phn_p+m_phn_res, std_phn_p*std_phn_res) if self.residual else D.Normal(m_phn_res, std_phn_res)
    z_phn = posterior_phn.rsample()
    kl_phn = D.kl.kl_divergence(posterior_phn, prior_phn)
    kl_phn = torch.sum(kl_phn*phn_mask, dim=[1,2])/torch.sum(phn_mask,dim=[1,2])
    s = self.phn2frame_comb(phn_embeded, z_phn) if self.straight_phn else self.phn2frame_comb(s, z_phn)
    s = self.phn2frame(s, phn2frame_ali)
    m_frame_p, std_frame_p = self.phn2frame_proj(s)
    prior_frame = D.Normal(m_frame_p, std_frame_p)

    #frame2wav
    #ftr = self.wav2frame_comb(frame_reps, s)
    #m_frame_res, std_frame_res = self.wav2frame_proj(ftr)
    #posterior_frame = D.Normal(m_frame_p+m_frame_res, std_frame_p*std_frame_res) if self.residual else D.Normal(m_frame_res, std_frame_res)
    #z_frame = posterior_frame.rsample()
    #kl_frame = D.kl.kl_divergence(posterior_frame, prior_frame)
    #kl_frame = torch.sum(kl_frame*frame_mask, dim=[1,2])/torch.sum(frame_mask, dim=[1,2])
    #s = self.frame2wav_comb(s, z_frame)
    

    return frame_reps, phn_reps, subword_reps, word_reps, sent_reps
    #return m_frame_res, m_phn_res, m_subword_res, m_word_res, m_sent_res
  def text_encoder(self,phn,phn_lengths,txt,txt2sub):
    phn_embeded = self.phn_emb(phn, phn_lengths)
    #text enc
    txt_embeded = self.txt_emb(txt, torch.ne(txt, 0))[0].detach() if self.bert_detach else self.txt_emb(txt, torch.ne(txt, 0))[0]
    if self.punc_context:
        txt_embeded = self.txt_context(txt_embeded.transpose(1,2)).transpose(1,2)
    txt_embeded = self.skipbymask(txt_embeded, txt2sub)
    return phn_embeded,txt_embeded
  def encoder(self,phn_embeded,txt_embeded,subword2phn_ali_m,word2subword_ali_m,sent2word_ali_m):
    txt_embeded = torch.cat([txt_embeded,self.phn_enc(phn_embeded, subword2phn_ali_m).transpose(1,2)],-1)
    subword_embeded = self.subword_enc(txt_embeded.transpose(1,2))
    word_embeded = self.word_enc(subword_embeded, word2subword_ali_m)
    sent_embedded = self.sent_enc(word_embeded, sent2word_ali_m)
    return subword_embeded,word_embeded,sent_embedded
  def sent_decode(self,sent_embedded):
    s = sent_embedded
    m_sent_p, std_sent_p  = self.sent_proj.infer(s)
    return s,m_sent_p, std_sent_p
    
  def word_decode(self,s,z_sent,ali,mel_len, word_embeded):
    s = self.sent2word_comb(s, z_sent)
    s = self.sent2word.infer(s, ali,mel_len, condition=word_embeded)
    m_word_p, std_word_p = self.sent2word_proj.infer(s)
    return s, m_word_p, std_word_p

  def subword_decode(self,s, z_word, ali,mel_len,subword_embeded):
    s = self.word2subword_comb(s, z_word)
    s = self.word2subword.infer(s, ali,mel_len, condition=subword_embeded)
    m_subword_p, std_subword_p = self.word2subword_proj.infer(s)

    return s,m_subword_p, std_subword_p

  def phone_decode(self,s, z_subword,ali,mel_len, phn_embeded):
    s = self.subword2phn_comb(s, z_subword)
    s = self.subword2phn.infer(s, ali,mel_len, condition=phn_embeded)
    m_phn_p, std_phn_p = self.subword2phn_proj.infer(s)
    log_d_predicted = self.dur_linear(s.transpose(1,2)).squeeze(2)

    return s, m_phn_p, std_phn_p,log_d_predicted

  def farm_de(self,s, z_phn,ali,mel_len,):
    s = self.phn2frame_comb(s, z_phn)
    s = self.phn2frame.infer(s, ali,mel_len)
    m_frame_p, std_frame_p = self.phn2frame_proj.infer(s)

    return s,m_frame_p,std_frame_p
  
  def farm_depre(self,s,z_phn,ali):
    s = self.phn2frame_comb(s, z_phn)
    s = output = torch.matmul(s,ali).transpose(1,2)
    return s
  def farm_decode(self,output,mel_len,mem):
    s,mems = self.phn2frame.deconline_infer(output,mel_len,mem=mem)
    m_frame_p, std_frame_p = self.phn2frame_proj.infer(s.transpose(1,2))
    return s.transpose(1,2),m_frame_p,std_frame_p,mems

  def tts_vocoder(self,s,z_frame):
    s = self.frame2wav_comb(s, z_frame)
    o = self.frame2wav.inference(s)
    return o

  def infer(self, phn, phn_lengths, txt, txt_lengths, txt2sub, sid=None, phn2frame_ali_m=None, phn2frame_ali=None, subword2phn_ali_m=None, subword2phn_ali=None, word2subword_ali_m=None, word2subword_ali=None, sent2word_ali_m=None, sent2word_ali=None, word_lengths=None, sub2sub=None, subword_lengths=None):
    
    phn_mask = torch.unsqueeze(commons.sequence_mask(phn_lengths, torch.max(phn_lengths)), 1).to(phn.dtype)
    subword_mask = torch.unsqueeze(commons.sequence_mask(subword_lengths, torch.max(subword_lengths)), 1).to(phn.dtype)

    temperature = {'sent':0.5,
                  'word':0.8,
                  'subword':1,
                  'phn':1,
                  'frame':1}

    if self.n_speakers > 1:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    phn_embeded = self.phn_emb(phn, phn_lengths)
    #text enc
    txt_embeded = self.txt_emb(txt, torch.ne(txt, 0))[0].detach() if self.bert_detach else self.txt_emb(txt, torch.ne(txt, 0))[0]
    if self.punc_context:
        txt_embeded = self.txt_context(txt_embeded.transpose(1,2)).transpose(1,2)
    txt_embeded = self.skipbymask(txt_embeded, txt2sub)
    txt_embeded, _, _ = self.length_regulator(txt_embeded, sub2sub, sub2phn=True)
    txt_embeded = torch.cat([txt_embeded,self.phn_enc(phn_embeded, subword2phn_ali_m).transpose(1,2)],-1)
    subword_embeded = self.subword_enc(txt_embeded.transpose(1,2))
    word_embeded = self.word_enc(subword_embeded, word2subword_ali_m)
    sent_embedded = self.sent_enc(word_embeded, sent2word_ali_m)

    if self.zero_txt:
        subword_embeded = torch.zeros_like(subword_embeded)
    
    #sent
    s = sent_embedded
    m_sent_p, std_sent_p  = self.sent_proj(s)
    if temperature['sent']!=0:
        prior_sent = D.Normal(m_sent_p, std_sent_p*temperature['sent'])# if self.zero_txt else D.Normal(m_sent_p, std_sent_p)
        z_sent = prior_sent.rsample()
    else:
        z_sent = m_sent_p

    #sent2word
    s = self.sent2word_comb(s, z_sent)

    s = self.sent2word(s, sent2word_ali, condition=word_embeded)
    m_word_p, std_word_p = self.sent2word_proj(s)
    if temperature['word']!=0:
        prior_word = D.Normal(m_word_p, std_word_p*temperature['word'])
        z_word = prior_word.rsample()
    else:
        z_word = m_word_p

    #word2subword
    s = self.word2subword_comb(s, z_word)
    s = self.word2subword(s, word2subword_ali, condition=subword_embeded)

    m_subword_p, std_subword_p = self.word2subword_proj(s)
    if temperature['subword']!=0:
        prior_subword = D.Normal(m_subword_p, std_subword_p*temperature['subword'])
        z_subword = prior_subword.rsample()
    else:
        z_subword = m_subword_p

    #subword2phn
    s = self.subword2phn_comb(s, z_subword)
    s = self.subword2phn(s, subword2phn_ali, condition=phn_embeded)
    log_d_predicted = self.dur_linear(s.transpose(1,2)).squeeze(2) #subword_ali unnique

    if phn2frame_ali==None:
        from utils import ali_mask
        d_target = torch.clamp(torch.round(torch.exp(log_d_predicted)*1.0-1.0), min=0)
        phn2frame_ali_m , phn2frame_ali = ali_mask([[int(x.item()) for x in d_target[0]]])
        phn2frame_ali_m = torch.BoolTensor(phn2frame_ali_m)
        phn2frame_ali = torch.LongTensor(phn2frame_ali)


    m_phn_p, std_phn_p = self.subword2phn_proj(s)
    if temperature['phn']!=0:
        prior_phn = D.Normal(m_phn_p, std_phn_p*temperature['phn'])
        z_phn = prior_phn.rsample()
    else:
        z_phn = m_phn_p

    #phn2frame
    s = self.phn2frame_comb(phn_embeded, z_phn) if self.straight_phn else self.phn2frame_comb(s, z_phn)
    
    s = self.phn2frame(s, phn2frame_ali)
    m_frame_p, std_frame_p = self.phn2frame_proj(s)
    if temperature['frame']!=0:
        prior_frame = D.Normal(m_frame_p, std_frame_p*temperature['frame'])
        z_frame = prior_frame.rsample()
    else:
        z_frame = m_frame_p
    #s = s.transpose(1,2)
    #o = self.out(s)
    #frame2wav
    s = self.frame2wav_comb(s, z_frame)
    o = self.frame2wav.inference(s)

    return o

  def analysis(self, phn, phn_lengths, txt, txt_lengths, txt2sub, sid=None, phn2frame_ali=None, subword2phn_ali=None, word2subword_ali=None, word_lengths=None, sub2sub=None, subword_lengths=None):
    phn_mask = torch.unsqueeze(commons.sequence_mask(phn_lengths, torch.max(phn_lengths)), 1).to(phn.dtype)
    subword_mask = torch.unsqueeze(commons.sequence_mask(subword_lengths, torch.max(subword_lengths)), 1).to(phn.dtype) 
    sent2word_ali=word_lengths.unsqueeze(1) #改

    temperature = {'sent':1,
                  'word':1,
                  'subword':1,
                  'phn':1,
                  'frame':1}

    if self.n_speakers > 1:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    phn_embeded = self.phn_emb(phn, phn_lengths)
    #text enc
    txt_embeded = self.txt_emb(txt, torch.ne(txt, 0))[0].detach() if self.bert_detach else self.txt_emb(txt, torch.ne(txt, 0))[0]
    txt_embeded = self.skipbymask(txt_embeded, txt2sub)
    txt_embeded, _, _ = self.length_regulator(txt_embeded, sub2sub, sub2phn=True)
    txt_embeded = torch.cat([txt_embeded,self.phn_enc(phn_embeded, subword2phn_ali).transpose(1,2)],-1)
    subword_embeded = self.subword_enc(txt_embeded.transpose(1,2))
    word_embeded = self.word_enc(subword_embeded, word2subword_ali)
    sent_embedded = self.sent_enc(word_embeded, sent2word_ali)

    if self.zero_txt:
        subword_embeded = torch.zeros_like(subword_embeded)
        word_embeded = torch.zeros_like(word_embeded)
        sent_embedded = torch.zeros_like(sent_embedded)
    
    #sent
    z_sent_dict = {}
    s_sent_dict = {}
    s = sent_embedded
    m_sent_p, std_sent_p  = self.sent_proj(s)
    prior_sent = D.Normal(m_sent_p, std_sent_p)# if self.zero_txt else D.Normal(m_sent_p, std_sent_p)
    z_sent = prior_sent.rsample()

    z_sent_dict['0']=m_sent_p
    z_sent_dict['1']=z_sent
    s_sent_dict['0']=s
    s_sent_dict['1']=s


    #sent2word
    z_word_dict = {}
    s_word_dict = {}
    for key in z_sent_dict.keys():
        s = self.sent2word_comb(s_sent_dict[key], z_sent_dict[key])
        s = self.sent2word(s, sent2word_ali, condition=word_embeded)

        m_word_p, std_word_p = self.sent2word_proj(s)
        prior_word = D.Normal(m_word_p, std_word_p*temperature['word'])
        z_word = prior_word.rsample()
        z_word_dict[key+'0']=m_word_p
        s_word_dict[key+'0']=s
        if '1' not in key:
            z_word_dict[key+'1']=z_word
            s_word_dict[key+'1']=s

    #word2subword
    z_subword_dict = {}
    s_subword_dict = {}
    for key in z_word_dict.keys():
        s = self.word2subword_comb(s_word_dict[key], z_word_dict[key])
        s = self.word2subword(s, word2subword_ali, condition=subword_embeded)

        m_subword_p, std_subword_p = self.word2subword_proj(s)
        prior_subword = D.Normal(m_subword_p, std_subword_p*temperature['subword'])
        z_subword = prior_subword.rsample()
        z_subword_dict[key+'0']=m_subword_p
        s_subword_dict[key+'0']=s
        if '1' not in key:
            z_subword_dict[key+'1']=z_subword
            s_subword_dict[key+'1']=s

    #subword2phn
    z_phn_dict = {}
    s_phn_dict = {}
    phn2frame_ali = {}
    for key in z_subword_dict.keys():
        s = self.subword2phn_comb(s_subword_dict[key], z_subword_dict[key])
        s = self.subword2phn(s, subword2phn_ali, condition=phn_embeded)

        log_d_predicted = self.dur_linear(s.transpose(1,2)).squeeze(2) #subword_ali unnique
        d_target = torch.clamp(torch.round(torch.exp(log_d_predicted)-1.0), min=0)


        m_phn_p, std_phn_p = self.subword2phn_proj(s)
        prior_phn = D.Normal(m_phn_p, std_phn_p*temperature['phn'])
        z_phn = prior_phn.rsample()
        z_phn_dict[key+'0']=m_phn_p
        s_phn_dict[key+'0']=s
        phn2frame_ali[key+'0']=d_target
        if '1' not in key:
            z_phn_dict[key+'1']=z_phn
            s_phn_dict[key+'1']=s
            phn2frame_ali[key+'1']=d_target




    #phn2frame
    z_frame_dict = {}
    s_frame_dict = {}
    for key in z_phn_dict.keys():
        s = self.phn2frame_comb(phn_embeded, z_phn_dict[key]) if self.straight_phn else self.phn2frame_comb(s_phn_dict[key], z_phn_dict[key])
        s = self.phn2frame(s, phn2frame_ali[key])
        m_frame_p, std_frame_p = self.phn2frame_proj(s)
        prior_frame = D.Normal(m_frame_p, std_frame_p*temperature['frame'])
        z_frame = prior_frame.rsample()
        z_frame_dict[key+'0']=m_frame_p
        s_frame_dict[key+'0']=s
        if '1' not in key:
            z_frame_dict[key+'1']=z_frame
            s_frame_dict[key+'1']=s

    #frame2wav
    wav = {}
    for key in z_frame_dict.keys():
        s = self.frame2wav_comb(s_frame_dict[key], z_frame_dict[key])
        o = self.frame2wav.inference(s, g=g)
        wav[key] = o
    return wav
