import torch
import torch.nn.functional as F
from torch.autograd import Variable


#LINF_SGD_LR = 0.01 #0.0078 # 0.04

# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def attack_Linf_PGD_labeled_free(netD, input_v, delta,y_b,y_c, metric, epsilon,lr,b_metric,c_metric,c_alph,n_gpu,num_bp):
    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    adverse_v.data.copy_((delta.data + adverse_v.data).clamp_(-1, 1))
    o_b, o_c = netD(adverse_v,y_c)
    loss = metric(o_b, o_c, y_b,y_c,b_metric,c_metric,c_alph)
    loss = (loss/num_bp) #*0.5
    loss.backward()
    gradients = adverse_v.grad.data.sign() * lr
    new_delta = delta.data + gradients.data
    new_delta.clamp_(-epsilon, epsilon)

    return new_delta,loss


def attack_L2_PGD_labeled_free(netD, input_v, delta, y_b,y_c, metric, epsilon,lr,b_metric,c_metric,c_alph,n_gpu,num_bp):

    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    adverse_v.data.copy_((delta.data + adverse_v.data).clamp_(-1, 1))
    o_b, o_c = netD(adverse_v,y_c)
    loss = metric(o_b, o_c, y_b,y_c,b_metric,c_metric,c_alph)
    loss = (loss/num_bp) #*0.5
    loss.backward()
    gradients = adverse_v.grad * lr/ adverse_v.grad.data.view(adverse_v.size(0), -1).norm(dim=1).add(1e-8).view(-1, 1, 1, 1)
    new_delta = delta.data + gradients.data
    new_delta = l2_cliper(new_delta,epsilon,n_gpu)

    return new_delta,loss


def l2_cliper(x,eps,n_gpu):
    epsilon = torch.tensor(eps).view(1,1,1,1)
    if n_gpu>0:
       epsilon = epsilon.cuda()
    norm = x.view(x.size(0),-1).norm(dim=1).add(1e-8)
    norm = norm.view(norm.size(0), 1, 1, 1)
    norm_out = torch.min(norm, epsilon)
    new_x = x / norm * norm_out
    return new_x




