import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)

class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = in_channel

        self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)

        self.style = AffineLinear(style_dim, in_channel * 2)
        self.style.affine.bias.data[:in_channel] = 1
        self.style.affine.bias.data[in_channel:] = 0

    def forward(self, input, style_code):
        # style
        style = self.style(style_code).unsqueeze(1)
        gamma, beta = style.chunk(2, dim=-1)
        

        out = self.norm(input)
        out = gamma * out + beta
        return out

class LinearNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True, 
                 spectral_norm=False,
                 ):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias)
        
        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        out = self.fc(input)
        return out

        
class ConvNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True, 
                 spectral_norm=False,
                 ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)
        
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out

class MultiHeadCrossAttention(nn.Module):
    ''' Multi-Head Cross Attention module '''
    def __init__(self, n_head, d_model, d_key, d_k, d_v, dropout=0., spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_key, n_head * d_k)
        self.w_vs = nn.Linear(d_key, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_model, 0.5), dropout=dropout)

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = nn.utils.spectral_norm(self.w_qs)
            self.w_ks = nn.utils.spectral_norm(self.w_ks)
            self.w_vs = nn.utils.spectral_norm(self.w_vs)
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, query, key, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = query.size()
        sz_b, len_k, _ = key.size()

        residual = query

        q = self.w_qs(query).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(key).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(key).view(sz_b, len_k, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
                        sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, angle=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5),dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        self.angle = angle
        if self.angle:
            self.angle_layer = Angle(d_model)

    def forward(self, q, k, v, mask=None, bias0=0, bias1=0):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        if self.angle:
            q = self.angle_layer(q, bias=bias0)
            k = self.angle_layer(k, bias=bias1)

        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
  
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.fc(output)
        output = self.dropout(output)


        output = self.layer_norm(output + residual)
        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn1 = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn1)
        attn = torch.where(torch.isnan(attn), torch.full_like(attn, 1)*1e-5, attn)
        attn = attn.masked_fill(mask, 0.0)
        if torch.any(torch.isnan(attn)):
            np.save('tmp.npy', attn1.detach().cpu().numpy(), allow_pickle=False)
            raise RuntimeError('Nan softmax after')
        p_attn = self.dropout(attn)
        
        output = torch.bmm(p_attn, v)
        if torch.any(torch.isnan(output)):
            raise RuntimeError('mask nan error')

        return output, attn


class Conv1dGLU(nn.Module):
    '''
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2*out_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x


class Angle(nn.Module):
    def __init__(self, d_word_vec):
        super().__init__()
        self.d_word_vec = d_word_vec
        #self.angle_embedding = nn.Embedding(d_word_vec//2,256)
        #self.angle_linear = nn.Linear(256,1)
        def cal_angle(hid_idx, d_word_vec):
            return np.power(10000, -2 * hid_idx / d_word_vec)
        self.angle = torch.Tensor([cal_angle(hid_j,self.d_word_vec) for hid_j in range(self.d_word_vec//2)]).unsqueeze(1).unsqueeze(0)
        #print(self.angle.shape)


    def forward(self, dec_output, bias=0):
        device = dec_output.device
        B,L,D = dec_output.shape
        dec1 = dec_output.view(B,L,D//2,2)
        dec2 = dec_output.view(B,L,D//2,2)
        dec2 = dec2.permute(3,0,1,2)[[[1,0]]].permute(1,2,3,0)

        dec2 = dec2.mul(torch.tensor([-1,1]).to(device)).float()
        #angle = torch.arange(D//2).float().to(device)
        #angle = self.angle_embedding(torch.arange(D//2).to(device))
        #angle = torch.tanh(self.angle_linear(angle))
        #print(angle)
        #angle = angle.unsqueeze(1)
        #128,1
        angle = self.angle.repeat(L,1,2).to(device)
        #print(angle)
        #print(angle.shape)
        pos = torch.arange(1,L+1).to(device)
        pos += bias
        pos = pos.view(L,1,1).repeat(1,D//2,2)
        angle = angle.mul(pos.float())
        dec1 = dec1.mul(torch.cos(angle))
        dec2 = dec2.mul(torch.sin(angle))
        dec_output = dec1+dec2
        dec_output = dec_output.view(B,L,D)
        return dec_output
