import torch
import torch.nn.functional as F
from torch.autograd import grad, Variable
from tools.adv_tools.linf_sgd import Linf_SGD

#LINF_SGD_LR = 0.01 #0.0078 # 0.04

# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def attack_Linf_PGD_labeled(netD, input_v, y_b,y_c, metric, steps, epsilon,lr,rs,b_metric,c_metric, c_alph,n_gpu):
    if steps<=0:
        return input_v
    netD.eval()
    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    if rs:
        rs_pts = torch.FloatTensor(adverse_v.size()).uniform_(-epsilon,epsilon)
        if n_gpu>0:
            rs_pts = rs_pts.cuda()
        adverse_v.data.copy_((rs_pts.data + input_v.data).clamp_(-1, 1))
    optimizer = Linf_SGD([adverse_v], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        netD.zero_grad()
        o_b, o_c = netD(adverse_v,y_c)
        loss = - metric(o_b, o_c, y_b,y_c,b_metric,c_metric,c_alph)
        loss.backward()
        optimizer.step()
        diff = adverse_v.data - input_v.data
        diff.clamp_(-epsilon, epsilon)
        adverse_v.data.copy_((diff + input_v.data).clamp_(-1, 1))
        #print('Linf-l:',round(loss.item(),5),round(diff.abs().mean().item(),5),round(adverse_v.abs().mean().item(),5))
    netD.train()
    netD.zero_grad()
    return adverse_v



def attack_L2_PGD_labeled(netD, input_v, y_b,y_c, metric, steps, epsilon,lr,rs,b_metric,c_metric,c_alph,n_gpu):
    if steps<=0:
        return input_v
    if epsilon == 0:
        return input_v
    netD.eval()
    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    eps = torch.tensor(epsilon).view(1,1,1,1)
    if n_gpu>0:
       eps = eps.cuda()
    if rs:
        rs_pts = torch.FloatTensor(adverse_v.size()).normal_()
        if n_gpu>0:
            rs_pts = rs_pts.cuda()
        rs_norm = rs_pts.view(rs_pts.size(0),-1).norm(dim=1).add(1e-8)
        rs_norm = rs_norm.view(rs_norm.size(0), 1, 1, 1)
        rs_norm_out = torch.min(rs_norm, eps)
        rs_pts = rs_pts / rs_norm * rs_norm_out
        adverse_v.data.copy_((rs_pts + input_v.data).clamp_(-1, 1))
    for _ in range(steps):
        netD.zero_grad()
        o_b, o_c = netD(adverse_v,y_c)
        loss = - metric(o_b, o_c, y_b,y_c,b_metric,c_metric,c_alph)
        loss.backward()
        gradients = adverse_v.grad * lr/ adverse_v.grad.data.view(adverse_v.size(0), -1).norm(dim=1).add(1e-8).view(-1, 1, 1, 1)
        adverse_v.data.copy_(gradients.data + adverse_v.data)
        diff = adverse_v.data - input_v.data
        norm = diff.view(diff.size(0),-1).norm(dim=1).add(1e-8)  #norm = torch.sqrt(torch.sum(diff * diff, (1, 2, 3)))
        norm = norm.view(norm.size(0), 1, 1, 1)
        norm_out = torch.min(norm, eps)
        diff = diff / norm * norm_out
        adverse_v.data.copy_((diff + input_v.data).clamp_(-1, 1))
        #print('L2-l:',round(loss.item(),5),round(norm.mean().item(),5),round(diff.abs().mean().item(),5),round(adverse_v.abs().mean().item(),5))
    netD.train()
    netD.zero_grad()
    return adverse_v
