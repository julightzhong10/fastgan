from tools.loading import load_models,load_datasets,load_optimizer, ANNEAL_TYPE
from tools.others import sample_ZCs,make_bin_label,x_R_proccess, init_list
from tools.metric import get_metrics, set_hinge_bd
import torch.optim as optim
from torchvision.utils import save_image
import argparse
import torch
import time
from tools.metric import Metric_Types
import math
import numpy as np
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=10, help = 'random seed')
parser.add_argument('--net', type=str, required=True, help='GAN structure: sngan_cifar, sngan_catdog64, sngan_catdog128, sngan_imgnet128, biggan')
parser.add_argument('--dim_z', type=int, required=True, help = 'dimension of nosie Z')
parser.add_argument('--img_c', type=int, required=True, help = 'image color channels')
parser.add_argument('--fm_base_g', type=int, required=True, help = 'conv layer channels multiplier for g')
parser.add_argument('--fm_base_d', type=int, required=True, help = 'conv layer channels multiplier for d')
parser.add_argument('--n_classes', type=int, required=True, help = 'number of classes')
parser.add_argument('--r_batch_size', type=int, required=True, help = 'real batch size')
parser.add_argument('--g_batch_size', type=int, required=True, help = 'fake batch size')
parser.add_argument('--num_bp', type=int, required=True, help = 'num of backprop to accumulate')
parser.add_argument('--data_path', type=str, required=True, help = 'dataset path')
parser.add_argument('--out_path', type=str, required=True, help = 'path of result to save, gen_*.pth and dis_*.pth')
parser.add_argument('--bottom_width', type=int, required=True, help = 'the width of feature maps at the first layer of G')
parser.add_argument('--img_width', type=int, required=True, help = 'the width of imgs')
parser.add_argument('--itr_d', type=int, required=True, help = 'MAX_D_STEP')
parser.add_argument('--lr_d', type=float, required=True, help = 'learning rate for D')
parser.add_argument('--beta1_d', type=float, required=True, help = 'adam b1 for D')
parser.add_argument('--beta2_d', type=float, required=True, help = 'adam b2 for D') 
parser.add_argument('--itr_g', type=int, default=1, help = 'default to 1')
parser.add_argument('--lr_g', type=float, required=True, help = 'learning rate for G')
parser.add_argument('--beta1_g', type=float, required=True, help = 'adam b1 for G')
parser.add_argument('--beta2_g', type=float, required=True, help = 'adam b2 for G')
parser.add_argument('--anneal_lr', type=str, default='nll', help = 'lr anneall type: exp, linear')
parser.add_argument('--anneal_lr_p1', type=float, default=1, help = 'lr anneal p1 respect to total_itrs')
parser.add_argument('--anneal_lr_p2', type=int, default=1, help = 'lr anneal p2 iters respect to total_itrs')
parser.add_argument('--n_gpu', type=int, required=True, help = 'num of gpus')
parser.add_argument('--dali', action='store_true', default=False, help = 'if use nvidia dali to load image')
parser.add_argument('--da', action='store_true', default=False, help = 'data augmentation?')
parser.add_argument('--n_workers', type=int, required=True, help = 'num of dataloader works')
parser.add_argument('--G_total_itrs', type=int, required=True, help = 'total updates of gen(total itrs)')
parser.add_argument('--wide', action='store_true', default=False, help = 'if use wide D')
parser.add_argument('--bn_d', action='store_true', default=False, help = 'if use batch_norm in D')
parser.add_argument('--bn_g', action='store_true', default=False, help = 'if use batch_norm in G')
parser.add_argument('--sn_d', action='store_true', default=False, help = 'if use sepcture_norm in D')
parser.add_argument('--sn_g', action='store_true', default=False, help = 'if use sepcture_norm in G')
parser.add_argument('--sa_d', action='store_true', default=False, help = 'if use self-attention in D')
parser.add_argument('--sa_g', action='store_true', default=False, help = 'if use self-attention in G')
parser.add_argument('--Z_dist', type=str, default='normal', help = 'Z noise distribution type: normal')
parser.add_argument('--Z_p1', type=float, default=0., help = 'Z noise distribution parameter 1')
parser.add_argument('--Z_p2', type=float, default=1., help = 'Z noise distribution parameter 2')
parser.add_argument('--save_bias', type=int, required=True, help = 'save bias respect to iterations')
parser.add_argument('--start_itr', type=int, default=0, help = 'start itrs, 0 means new training')
parser.add_argument('--shuffle', action='store_true', default=False, help = 'if shuffle data')
parser.add_argument('--b_metric', type=str, default='ce',help='bin loss for training, ce for cross_entropy, or hinge')
parser.add_argument('--c_metric', type=str, default='ce',help='class loss for training, ce for cross_entropy, or hinge projection, ce_kl for ce with KL in L^f_D')
parser.add_argument('--b_alph_f', type=float, default=1.0,help = 'binary loss coefficient of fake data in D')
parser.add_argument('--c_alph_f', type=float, default=1.0,help = 'class loss coefficient of fake data for D')
parser.add_argument('--c_alph_r', type=float, default=1.0,help = 'class loss coefficient of real data for D')
parser.add_argument('--c_alph_g', type=float, default=1.0,help = 'class loss coefficient of generated data for G')
parser.add_argument('--hinge_bd', type=float, default=1.0,help = 'margin boudary of hinge loss')
parser.add_argument('--pgd_type', type=str, required=True, help = 'Linf or L2')
parser.add_argument('--pgd_steps', type=int, required=True, help = 'number of steps for pgd attack')
parser.add_argument('--pgd_eps', type=float, required=True, help = 'radius of Linf-norm ball for pgd attack')
parser.add_argument('--pgd_tau', type=float, required=True, help = 'lr of pgd step')
parser.add_argument('--pgd_rs', action='store_true', default=True, help = 'flag to random start pgd attack')
opt = parser.parse_args()

print(opt)
torch.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)
set_hinge_bd(opt.hinge_bd)
def train():
    lr_str = 'lr'
    Z_dist = opt.Z_dist
    Z_params = [opt.Z_p1,opt.Z_p2]
    sigma = 1e-8

    # models 
    netG,netD = load_models(net=opt.net,n_gpu=opt.n_gpu,dim_z=opt.dim_z,img_c=opt.img_c,img_width=opt.img_width,\
                fm_base_d=opt.fm_base_d,fm_base_g=opt.fm_base_g,bn_d=opt.bn_d,bn_g=opt.bn_g,sn_d=opt.sn_d,sn_g=opt.sn_g,\
                sa_d=opt.sa_d,sa_g=opt.sa_g,bottom_width=opt.bottom_width,n_classes=opt.n_classes,c_metric=opt.c_metric,wide=opt.wide)

    if opt.start_itr>0:
        netG.load_state_dict(torch.load(f'{opt.out_path}check_pts/gen_{opt.start_itr}.pth'))
        netD.load_state_dict(torch.load(f'{opt.out_path}check_pts/dis_{opt.start_itr}.pth'))
    G_itrs = int(opt.start_itr*opt.save_bias-1)
    D_itrs = int(G_itrs*(opt.itr_d/opt.itr_g))

    # optimizers
    optimizerG = load_optimizer(model=netG,lr=opt.lr_g,betas=(opt.beta1_g,opt.beta2_g))
    optimizerD = load_optimizer(model=netD,lr=opt.lr_d,betas=(opt.beta1_d,opt.beta2_d))
    if opt.anneal_lr in ANNEAL_TYPE:
        if opt.anneal_lr==ANNEAL_TYPE[0]:
            lr_lambda_D = lambda epoch: max(int(G_itrs<=opt.anneal_lr_p1),int(G_itrs>opt.anneal_lr_p1)*(opt.G_total_itrs-G_itrs)/(opt.G_total_itrs-opt.anneal_lr_p1))
            lr_lambda_G = lambda epoch: max(int(G_itrs<=opt.anneal_lr_p1),int(G_itrs>opt.anneal_lr_p1)*(opt.G_total_itrs-G_itrs)/(opt.G_total_itrs-opt.anneal_lr_p1))
        elif opt.anneal_lr==ANNEAL_TYPE[1]:
            lr_lambda_D = lambda epoch: opt.anneal_lr_p1 ** (G_itrs/opt.anneal_lr_p2)
            lr_lambda_G = lambda epoch: opt.anneal_lr_p1 ** (G_itrs/opt.anneal_lr_p2)
        lr_adjuster_D = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda_D, last_epoch=-1)
        lr_adjuster_G = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda_G, last_epoch=-1)

    if opt.r_batch_size<=0:
        raise ValueError(f"batch size must >0")

    loader,class2idx = load_datasets(data_path=opt.data_path,img_width=opt.img_width, \
                        batch_size=opt.r_batch_size,n_workers=opt.n_workers,n_gpu=opt.n_gpu,shuffle=opt.shuffle,dali=opt.dali,seed=opt.random_seed,da=opt.da)
    loader_iter = iter(loader)
    # fixed inputs for visialization
    fixed_Z,fixed_C_int,fixed_C_vec = sample_ZCs(64,opt.dim_z,opt.n_classes,Z_dist,Z_params,opt.n_gpu)
    
    print("Starting GAN Training Loop...")
    Metric_G, Metric_R, Metric_F = get_metrics()
    D_err_R,D_err_F,G_err,D_cnt,G_cnt=0,0,0,0,0
    tpts = time.time()
    itrs = 0
    while G_itrs<opt.G_total_itrs:
        # load data
        if opt.r_batch_size>0:
            try:
                data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                data = next(loader_iter)
            x_R = data[0]["data"] if opt.dali else data[0]
            y_R_c = data[0]["label"].view(-1).long() if opt.dali else data[1]
        else:
            x_R = torch.zeros(0, opt.img_c, opt.img_width,opt.img_width)
            y_R_c = torch.full((0,), 0).long()
        # to cuda ?
        if opt.n_gpu>0:
            x_R, y_R_c = x_R.cuda(), y_R_c.cuda()
        x_R, y_R_c = list(torch.chunk(x_R,opt.num_bp)),list(torch.chunk(y_R_c,opt.num_bp))
        r_BS, g_BS = init_list(opt.num_bp),init_list(opt.num_bp)

        for bp_i in range(opt.num_bp):
            r_BS[bp_i] = x_R[bp_i].size(0)
            g_BS[bp_i] = int(opt.g_batch_size/opt.num_bp)
        if opt.g_batch_size%opt.num_bp !=0:
            g_BS[-1] = opt.g_batch_size%opt.num_bps


        y_R_b = init_list(opt.num_bp)
        y_G_b,y_F_b = init_list(opt.num_bp),init_list(opt.num_bp)
        for bp_i in range(opt.num_bp):
            y_R_b[bp_i],y_G_b[bp_i],y_F_b[bp_i] = make_bin_label(r_BS[bp_i],g_BS[bp_i],opt.n_gpu) # prepare inputs and label

        ############################
        # (1) Update D network:
        ###########################
        if itrs%opt.itr_g==0:
            D_cnt += 1
            D_itrs += 1
            netD.zero_grad()
            for bp_i in range(opt.num_bp):
                # produce perturpation
                x_R[bp_i] = x_R_proccess(netD=netD,x_R=x_R[bp_i],y_R_b=y_R_b[bp_i], y_R_c=y_R_c[bp_i],metric = Metric_R,\
                                        pgd_type=opt.pgd_type,pgd_steps=opt.pgd_steps, pgd_eps=opt.pgd_eps, pgd_tau=opt.pgd_tau,pgd_rs=opt.pgd_rs, \
                                        b_metric=opt.b_metric,c_metric=opt.c_metric, c_alph=opt.c_alph_r,n_gpu=opt.n_gpu)
            for bp_i in range(opt.num_bp):
                loss_dis_R = 0 
                o_R_b,o_R_c = netD(x_R[bp_i],y_R_c[bp_i])
                loss_dis_R = Metric_R(o_R_b,o_R_c,y_R_b[bp_i],y_R_c[bp_i],opt.b_metric,opt.c_metric,opt.c_alph_r)
                loss_dis_R = loss_dis_R/opt.num_bp
                loss_dis_R.backward()
                D_err_R += loss_dis_R.item()
                Z, C_int, C_vec = sample_ZCs(g_BS[bp_i],opt.dim_z,opt.n_classes,Z_dist,Z_params,opt.n_gpu)
                x_F = netG(Z,C_vec).detach()
                y_F_c = C_int.view(-1)
                o_F_b,o_F_c = netD(x_F,y_F_c)
                loss_dis_F = Metric_F(o_F_b,o_F_c,y_F_b[bp_i],y_F_c,opt.b_metric,opt.c_metric,opt.b_alph_f,opt.c_alph_f)
                loss_dis_F = loss_dis_F/opt.num_bp
                loss_dis_F.backward()
                D_err_F += loss_dis_F.item()
            optimizerD.step()

        if itrs % opt.itr_d == 0:
            G_cnt += 1
            G_itrs += 1
            ############################
            # (2) Update G network:
            ###########################
            netD.zero_grad()
            netG.zero_grad()
            for bp_i in range(opt.num_bp):
                Z, C_int, C_vec = sample_ZCs(g_BS[bp_i],opt.dim_z,opt.n_classes,Z_dist,Z_params,opt.n_gpu)
                y_G_c = C_int.view(-1)
                x_G = netG(Z,C_vec)
                o_G_b,o_G_c = netD(x_G,y_G_c)
                loss_gen = Metric_G(o_G_b,o_G_c,y_G_b[bp_i],y_G_c,opt.b_metric,opt.c_metric,opt.c_alph_g)
                loss_gen = (loss_gen/opt.num_bp)
                G_err += loss_gen.item()
                loss_gen.backward()
            optimizerG.step()
        if G_itrs%opt.save_bias == 0 and (itrs % opt.itr_d==0):
            print(f'Training[{G_itrs}/{opt.G_total_itrs}] G_loss:{round(G_err/(G_cnt+sigma),3)},D_real:{round(D_err_R/(D_cnt+sigma),3)},D_fake:{round(D_err_F/(D_cnt+sigma),3)},secs:{round(time.time()-tpts,1)},lrD:{round(optimizerD.param_groups[0][lr_str],8)},lrG:{round(optimizerG.param_groups[0][lr_str],8)},r:{opt.r_batch_size},g:{opt.g_batch_size}')
            D_err_R,D_err_F,G_err,D_cnt,G_cnt=0,0,0,0,0
            tpts = time.time()
            curr_pts = int(G_itrs/opt.save_bias)
            with torch.no_grad():
                fixed_x_fake = netG(fixed_Z,fixed_C_vec)
                fixed_x_fake.data.mul_(0.5).add_(0.5)
                fixed_x_fake = fixed_x_fake.view(64,opt.img_c,opt.img_width,opt.img_width)
                save_image(fixed_x_fake.data, f'{opt.out_path}imgs/{curr_pts}.png', nrow=8)    
            torch.save(netG.state_dict(), f'{opt.out_path}check_pts/gen_{curr_pts}.pth')
            torch.save(netD.state_dict(), f'{opt.out_path}check_pts/dis_{curr_pts}.pth')
        
        # anneal part
        if opt.anneal_lr in ANNEAL_TYPE:
            lr_adjuster_G.step()
            lr_adjuster_D.step()

        # finally add itrs for keep count continuing
        itrs += 1 



if __name__ == "__main__":
    if not os.path.exists(opt.out_path):
        raise ValueError(f"No output path {opt.out_path}")
    if not os.path.exists(opt.out_path+'/check_pts'):
        os.makedirs(opt.out_path+'/check_pts')
    if not os.path.exists(opt.out_path+'/imgs'):
        os.makedirs(opt.out_path+'/imgs')
    if not os.path.exists(opt.out_path+'/log'):
        os.makedirs(opt.out_path+'/log')
    train()
