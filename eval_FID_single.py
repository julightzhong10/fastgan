import torch
from tools.loading import load_models_gen,load_datasets, load_inception_FID
import argparse
import random
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser()
parser.add_argument('--stat_path', type=str, required=True)
parser.add_argument('--load_path', type=str, required=True)
parser.add_argument('--net', type=str, required=True)
parser.add_argument('--dim_z', type=int, required=True)
parser.add_argument('--img_c', type=int, required=True)
parser.add_argument('--fm_base', type=int, required=True)
parser.add_argument('--n_classes', type=int, required=True)
parser.add_argument('--bottom_width', type=int, required=True)
parser.add_argument('--img_width', type=int, required=True)
parser.add_argument('--ipt_dims', type=int, required=True)
parser.add_argument('--n_gpu', type=int, required=True)
parser.add_argument('--dali', action='store_true', default=False)
parser.add_argument('--da', action='store_true', default=False, help = 'data augmentation?')
parser.add_argument('--tf', action='store_true', default=False)
parser.add_argument('--n_workers', type=int, required=True)
parser.add_argument('--bn', action='store_true', default=False)
parser.add_argument('--sn', action='store_true', default=False)
parser.add_argument('--sa', action='store_true', default=False)
parser.add_argument('--Z_dist', type=str, default='normal')
parser.add_argument('--Z_p1', type=float, default=0.)
parser.add_argument('--Z_p2', type=float, default=1.)
parser.add_argument('--batch_size', type=int, required=100)
parser.add_argument('--n_samples', type=int, required=50000)
parser.add_argument('--real_data_FID', action='store_true', default=False)
opt = parser.parse_args()

def eval():
    Z_dist=opt.Z_dist
    Z_params=[opt.Z_p1,opt.Z_p2]
    # models
    m2 = np.load(opt.stat_path)['mu']
    s2 = np.load(opt.stat_path)['sig']
    if opt.tf:
        import tools.networks.inception_eval.inception_tf as inception_tf
        from tools.evalution import sample_imgs, generate_imgs
    else:
        from tools.evalution import frechet_inception_distance,get_activations_stat_orig, calculate_frechet_distance, get_activations_stat_gen
        inception_model = load_inception_FID(opt.n_gpu,opt.ipt_dims)

    if opt.real_data_FID:
        datasets, class_to_idx = load_datasets(data_path=opt.load_path,img_width=opt.img_width,\
                    batch_size=opt.batch_size,n_workers=opt.n_workers,n_gpu=opt.n_gpu,shuffle=True,dali=opt.dali,seed=random.randint(0,9999),da=opt.da)
        if opt.tf:
            total_iters = int(opt.n_samples/opt.batch_size)
            img_list = sample_imgs(train_loader=datasets,total=opt.n_samples,total_iters=total_iters,img_width=opt.img_width,dali=opt.dali)
            m1,s1 = inception_tf.get_mean_and_cov(img_list)
            FID_score = inception_tf.get_fid(m1,s1,m2,s2)
        else:
            m1,s1 = get_activations_stat_orig(datasets,opt.n_samples,inception_model,opt.ipt_dims,opt.n_gpu,opt.dali)
            FID_score = calculate_frechet_distance(m1,s1,m2,s2)
        print(f"FID:{round(FID_score,4)}")
    else:
        datasets = None
        curr_path = opt.load_path
        netG = load_models_gen(net=opt.net,n_gpu=opt.n_gpu,dim_z=opt.dim_z,img_c=opt.img_c,img_width=opt.img_width,fm_base=opt.fm_base,
                bn=opt.bn,sn=opt.sn,sa=opt.sa,bottom_width=opt.bottom_width,n_classes=opt.n_classes,load_path=curr_path)
        netG.eval()
        if opt.tf:
                img_list = generate_imgs(netG,opt.n_samples,opt.batch_size,opt.img_width, opt.dim_z,opt.n_classes,Z_dist,Z_params,opt.n_gpu)
                netG = None
                m1,s1 = inception_tf.get_mean_and_cov(img_list)
                FID_score = inception_tf.get_fid(m1,s1,m2,s2)              
        else:
            with torch.no_grad():
                total_itrs = int(opt.n_samples/opt.batch_size)
                m1, s1 = get_activations_stat_gen(netG=netG,z_dim=opt.dim_z,n_classes=opt.n_classes,Z_dist=Z_dist,Z_params=Z_params,\
                                                ipt_net=inception_model,total_itrs=total_itrs,batch_size=opt.batch_size,ipt_dims=opt.ipt_dims,n_gpu=opt.n_gpu)
                FID_score = calculate_frechet_distance(m1,s1,m2,s2)
        print(f"FID:{round(FID_score,4)},{curr_path}")
if __name__ == "__main__":
    eval()
