import torch 
from torch.nn import functional as F

import commons

def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses

def gauss_kl_loss(z, m_q, logs_q, z_mask):
  """
  z, m_q, logs_q: [b, h, t_t]
  """
  m_q = m_q.float()
  logs_q = logs_q.float()
  z_mask = z_mask.float()#.repeat(1,m_q.size(1),1)
  kl = -0.5*(1+ logs_q - m_q.pow(2) - logs_q.exp())
  kl = torch.sum(kl*z_mask)
  l = kl/torch.sum(z_mask)  
  return l

def match_kl_loss(z_p, m_q, logs_q, m_p, logs_p, z_mask, non_neg=True):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()#.repeat(1,m_q.size(1),1)

  if non_neg:
    p_distribution = torch.distributions.Normal(m_p, logs_p)
    q_distribution = torch.distributions.Normal(m_q, logs_q)
    kl = torch.distributions.kl_divergence(p_distribution, q_distribution)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
  else:
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
  return l

'''
def kl_loss_1(z_q, m_q, logs_q, m_p, logs_p, src_len, D):
  B = z_q.shape[0]
  kl_loss = 0.0
  for b, src_l in enumerate(src_len):
    log_q_conv = D.log_density(z_q[b,:src_l], params=torch.cat([m_q[b,:src_l].unsqueeze(-1), logs_q[b,:src_l].unsqueeze(-1)],-1))
    log_p_conv = D.log_density(z_q[b,:src_l], params=torch.cat([m_p[b,:src_l].unsqueeze(-1), logs_p[b,:src_l].unsqueeze(-1)],-1))
    kl_loss += (log_q_conv - log_p_conv).mean()
  kl_loss/=B
  return kl_loss
'''
