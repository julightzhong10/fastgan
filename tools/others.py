import torch
import torch.nn.functional as F
from tools.adv_tools.pgd import attack_Linf_PGD_labeled, attack_L2_PGD_labeled
from tools.adv_tools.pgd_free import attack_Linf_PGD_labeled_free, attack_L2_PGD_labeled_free
from tools.metric import Metric_Types

PGD_TYPE = ('Linf','L2')

PGD_FREE_TYPE = ('Linf','L2')

def x_R_proccess(netD,x_R,y_R_b,y_R_c, metric,pgd_type, pgd_steps, pgd_eps, pgd_tau, pgd_rs, b_metric,c_metric,c_alph,n_gpu):

    # labeled data
    if x_R.size(0)>0:
        # find adv example for labeled data
        if pgd_type==PGD_TYPE[0]:
            x_R_adv = attack_Linf_PGD_labeled(netD=netD,input_v=x_R,y_b=y_R_b,y_c=y_R_c,\
                            metric=metric,steps=pgd_steps,epsilon=pgd_eps,lr=pgd_tau,rs=pgd_rs,b_metric=b_metric,c_metric=c_metric,\
                            c_alph=c_alph,n_gpu=n_gpu)
        elif pgd_type==PGD_TYPE[1]:
            x_R_adv = attack_L2_PGD_labeled(netD=netD,input_v=x_R,y_b=y_R_b,y_c=y_R_c,metric=metric,\
                            steps=pgd_steps,epsilon=pgd_eps,lr=pgd_tau,rs=pgd_rs,b_metric=b_metric,c_metric=c_metric,\
                            c_alph=c_alph,n_gpu=n_gpu)
        else:
            raise ValueError(f"no pgd attack name: {pgd_type} for labeled data, only use {PGD_TYPE}")      
    else:
        x_R_adv = x_R
    return x_R_adv




def x_R_proccess_free(netD,x_R,y_R_b,y_R_c,metric, delta, pgd_type, pgd_eps, pgd_tau,b_metric,c_metric, c_alph,n_gpu,num_bp):

    # labeled data
    if x_R.size(0)>0:
        # find adv example for labeled data
        if pgd_type==PGD_FREE_TYPE[0]:
            new_delta,loss = attack_Linf_PGD_labeled_free(netD=netD,input_v=x_R,delta=delta,y_b=y_R_b,y_c=y_R_c,\
                            metric=metric,epsilon=pgd_eps,lr=pgd_tau,b_metric=b_metric,c_metric=c_metric,\
                            c_alph=c_alph,n_gpu=n_gpu,num_bp=num_bp)
        elif pgd_type==PGD_FREE_TYPE[1]:
            new_delta,loss = attack_L2_PGD_labeled_free(netD=netD,input_v=x_R,delta=delta,y_b=y_R_b,y_c=y_R_c,\
                            metric=metric,epsilon=pgd_eps,lr=pgd_tau,b_metric=b_metric,c_metric=c_metric,\
                            c_alph=c_alph,n_gpu=n_gpu,num_bp=num_bp)
        else:
            raise ValueError(f"no pgd attack name: {pgd_type} for labeled data, only use {PGD_FREE_TYPE}")      
    else:
        new_delta, loss = delta, 0
    return new_delta, loss



def make_bin_label(r_BS,g_BS,n_gpu):
    y_R_b = torch.full((r_BS,), 1)
    y_G_b = torch.full((g_BS,), 1)
    y_F_b = torch.full((g_BS,), 0)
    if n_gpu>0:
        y_R_b, y_G_b, y_F_b = y_R_b.cuda(), y_G_b.cuda(),y_F_b.cuda()
    return y_R_b, y_G_b,y_F_b

def init_list(n_bp):
    return [None for i in range(n_bp)]

def sample_ZCsList(total_data,dim_z,num_c,Z_dist,Z_params):
    Z_list,C_int_list,C_vec_list=[],[],[]
    for i in range(total_data):    
        Z,C_int,C_vec=sample_ZCs(1,dim_z,num_c,Z_dist,Z_params)
        Z_list.append(Z)
        C_int_list.append(C_int)
        C_vec_list.append(C_vec)
    return Z_list,C_int_list,C_vec_list



def sample_ZCs(batch_size,z_dim,c_dim,noise_type='uniform',params=[-1.0,1.0],n_gpu=0):
    #noise = torch.randn(batch_size, z_dim)
    c_dim = c_dim if c_dim>0 else 1
    Z = generate_noise(noise_type,params,torch.Size([batch_size,z_dim]),n_gpu)
    C_int=torch.randint(low=0,high=c_dim,size=(batch_size,1))
    C_vec=torch.zeros([batch_size, c_dim], dtype=torch.float32)
    C_vec.scatter_(1,C_int,1)
    if n_gpu>0:
        Z,C_int,C_vec =Z.cuda(),C_int.cuda(),C_vec.cuda()
    return Z,C_int,C_vec


def generate_noise(noise_type, params, size,n_gpu=0):
    noise = torch.FloatTensor(size)
    if noise_type=='normal':    # normal distribution 
        noise.normal_(mean=params[0],std=params[1])
    elif noise_type=='uniform':    # continuous uniform distribution
        noise.uniform_(params[0],params[1])
    elif noise_type=='uniform_disc':  # discrete uniform distribution  
        noise.random_(params[0],params[1])
    elif noise_type=='log_normal':  #log-normal distribution  
        noise.log_normal_(mean=params[0],std=params[1])
    elif noise_type=='geometric':  #geometric distribution  
        noise.geometric_(p=params[0])
    elif noise_type=='exponential':  #exponential distribution
        noise.exponential_(lambd=params[0])
    elif noise_type=='cauchy':  #cauchy distribution
        noise.cauchy_(median=params[0], sigma=params[1])
    else:
        noise.zero_()
    if n_gpu>0:
        noise = noise.cuda()
    return noise
