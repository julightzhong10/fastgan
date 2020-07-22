import torch
import torch.nn.functional as F
import math

Metric_Types = ['ce','hinge','ce_kl']

hinge_bd = 1.0

def set_hinge_bd(bd):
    global hinge_bd
    hinge_bd =bd

def get_metrics():
    return gen_metric,dis_R_metric, dis_F_metric


def gen_metric(o_b,o_c,y_b,y_c,b_metric,c_metric, c_alph):
    metric = 0
    project_value = 0

    if b_metric == Metric_Types[0]:
        metric = metric + F.binary_cross_entropy_with_logits(o_b.view(-1), y_b)
    elif b_metric ==Metric_Types[1]:
        project_value = project_value + o_b
    else:
        raise ValueError(f"only ce or hinge metric in adv binary metric: {b_metric}")

    if c_metric == Metric_Types[0] or c_metric == Metric_Types[2]: 
        metric = metric + c_alph*F.cross_entropy(o_c, y_c)
    elif c_metric == Metric_Types[1]: 
        project_value = project_value + c_alph*o_c
    else:
       raise ValueError(f"only ce, hinge or ce_kl metric in c_metric: {c_metric}")

    if b_metric==Metric_Types[1] or c_metric==Metric_Types[1]:
        metric = metric - torch.mean(project_value)
    return metric


def dis_F_metric(o_b,o_c,y_b,y_c,b_metric,c_metric,b_alph,c_alph):
    # metric_b: 0-CE, 1-hinge
    # metric_c: 0-None, 1-CE, 2-hinge
    metric = 0
    project_value = 0
    if b_metric == Metric_Types[0]:
        metric = metric + b_alph*F.binary_cross_entropy_with_logits(o_b.view(-1), y_b)
    elif b_metric == Metric_Types[1]:
        project_value = project_value + b_alph*o_b
    else:
        raise ValueError(f"only ce or hinge metric in bin metric: {b_metric}")
    
    if c_metric == Metric_Types[0]:
        metric = metric + c_alph*F.cross_entropy(o_c, y_c)
    elif c_metric == Metric_Types[1]: 
        project_value = project_value + c_alph*o_c
    elif c_metric == Metric_Types[2]:
        uniform_label = torch.ones(o_c.size())
        if y_c.is_cuda:
           uniform_label = uniform_label.cuda()
        uniform_label = F.softmax(uniform_label,1)
        metric = metric + c_alph*F.kl_div(F.log_softmax(o_c, dim=1),uniform_label, reduction='batchmean')
    else:
       raise ValueError(f"only ce, hinge or ce_kl metric in c_metric: {c_metric}")

    if b_metric==Metric_Types[1] or c_metric==Metric_Types[1]:
        metric = metric + torch.mean(F.relu(hinge_bd + project_value))
    return metric


def dis_R_metric(o_b,o_c,y_b,y_c,b_metric,c_metric,c_alph):
    # metric_b: 0-CE, 1-hinge
    # metric_c: 0-None, 1-CE, 2-hinge
    metric = 0
    project_value = 0

    if b_metric == Metric_Types[0]:
        metric = metric + F.binary_cross_entropy_with_logits(o_b.view(-1), y_b)
    elif b_metric ==Metric_Types[1]:
        project_value = project_value + o_b
    else:
       raise ValueError(f"only ce or hinge metric in bin metric: {b_metric}")
    
    
    if c_metric == Metric_Types[0] or c_metric == Metric_Types[2]:
        metric = metric + c_alph*F.cross_entropy(o_c, y_c)
    elif c_metric == Metric_Types[1]:
        project_value = project_value + c_alph*o_c
    else:
       raise ValueError(f"only ce, hinge or ce_kl metric in c_metric: {c_metric}")
    
    if b_metric==Metric_Types[1] or c_metric==Metric_Types[1]:
        metric = metric + torch.mean(F.relu(hinge_bd - project_value))
    return metric
