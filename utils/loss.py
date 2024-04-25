
import torch
import torch.nn.functional as F

def BPR_LOSS(u_embs, pos_embs, neg_embs):
    pos_scores = torch.sum(u_embs * pos_embs, 1)
    neg_scores = torch.sum(u_embs * neg_embs, 1)
    
    return torch.mean(F.softplus(neg_scores - pos_scores))

def REG_LOSS_POW(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2).pow(2)
    return emb_loss * reg

def UNIFORM_LOSS(x, t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

