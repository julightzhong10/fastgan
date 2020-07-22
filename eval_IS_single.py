import torch
from tools.evalution import inception_score,inception_score_test
from tools.loading import load_models_gen,load_inception_IS,load_datasets
import numpy as np
import argparse
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, required=True)
parser.add_argument('--dim_z', type=int, required=True)
parser.add_argument('--img_c', type=int, required=True)
parser.add_argument('--fm_base', type=int, required=True)
parser.add_argument('--n_classes', type=int, required=True)
parser.add_argument('--bottom_width', type=int, required=True)
parser.add_argument('--img_width', type=int, required=True)
parser.add_argument('--n_gpu', type=int, required=True)
parser.add_argument('--dali', action='store_true', default=False)
parser.add_argument('--da', action='store_true', default=False, help = 'data augmentation?')
parser.add_argument('--tf', action='store_true', default=False)
parser.add_argument('--bn', action='store_true', default=False)
parser.add_argument('--sn', action='store_true', default=False)
parser.add_argument('--sa', action='store_true', default=False)
parser.add_argument('--Z_dist', type=str, default='normal')
parser.add_argument('--Z_p1', type=float, default=0.)
parser.add_argument('--Z_p2', type=float, default=1.)
parser.add_argument('--load_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_splits', type=int, default=10)
parser.add_argument('--n_samples', type=int, required=50000)
parser.add_argument('--real_data_IS', action='store_true', default=False)
opt = parser.parse_args()

def eval():
    Z_dist=opt.Z_dist
    Z_params=[opt.Z_p1,opt.Z_p2]

    if opt.real_data_IS:
        train_loader,class_to_idx = load_datasets(data_path=opt.load_path,img_width=opt.img_width,\
                        batch_size=opt.batch_size,n_workers=1,n_gpu=opt.n_gpu,shuffle=True,dali=opt.dali,seed=random.randint(0,9999),da=opt.da)
        total_iters = int(opt.n_samples/opt.batch_size)
        if opt.tf:
                import tools.networks.inception_eval.inception_tf as inception_tf
                from tools.evalution import sample_imgs
                img_list = sample_imgs(train_loader=train_loader,total=opt.n_samples,total_iters=total_iters,img_width=opt.img_width,dali=opt.dali) 
                train_loader = None
                IS_mean, IS_std = inception_tf.get_inception_score(img_list,opt.n_splits)
        else:
                inception_model = load_inception_IS(opt.n_gpu)
                with torch.no_grad():
                        IS_mean, IS_std=inception_score_test(ipt_net=inception_model,train_loader=train_loader,img_width=opt.img_width,
                                                        n_splits=opt.n_splits,total_iters=total_iters,n_gpu=opt.n_gpu,dali=opt.dali)
        print(f"IS_Mean: {IS_mean}, IS_Std: {IS_std}")
    else:
        if opt.tf:
                import tools.networks.inception_eval.inception_tf as inception_tf
                from tools.evalution import generate_imgs
        else:
                inception_model = load_inception_IS(opt.n_gpu)        
        curr_path = opt.load_path
        netG = load_models_gen(net=opt.net,n_gpu=opt.n_gpu,dim_z=opt.dim_z,img_c=opt.img_c,img_width=opt.img_width,fm_base=opt.fm_base,
                bn=opt.bn,sn=opt.sn,sa=opt.sa,bottom_width=opt.bottom_width,n_classes=opt.n_classes,load_path=curr_path)
        netG.eval()
        if opt.tf:
                img_list = generate_imgs(netG,opt.n_samples,opt.batch_size,opt.img_width, opt.dim_z,opt.n_classes,Z_dist,Z_params,opt.n_gpu)
                netG = None
                IS_mean, IS_std = inception_tf.get_inception_score(img_list,opt.n_splits)
        else:
                with torch.no_grad():
                        IS_mean, IS_std=inception_score(netG, inception_model, opt.n_samples, opt.batch_size, opt.dim_z,opt.n_classes,
                                        opt.img_width,opt.n_splits,Z_dist,Z_params,opt.n_gpu)
        print(f"IS_mean:{round(IS_mean.item(),4)},IS_std:{round(IS_std.item(),4)},{curr_path}")
if __name__ == "__main__":
    eval()
