import numpy as np
import torch

def p_probs(net, device, data, model_type="mlp", val=True):
    # Adapted from https://github.com/acmi-lab/PU_learning/blob/main/estimator.py
    net.eval()
    pp_probs = None
    with torch.no_grad():
        data = data.to(device)
        if model_type == "mlp":
            outputs = net(data.x)
        else:
            outputs = net(data.x, data.edge_index)

        if val:
            mask = torch.logical_and(data.src_mask, data.val_mask)
        else:
            mask = data.src_mask
        outputs = outputs[mask]
        probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 0]
        if pp_probs is None:
            pp_probs = probs.detach().cpu().numpy().squeeze()
        else:
            pp_probs = np.concatenate((pp_probs, \
                                       probs.detach().cpu().numpy().squeeze()), axis=0)

    return pp_probs


def u_probs(net, device, data, model_type="mlp", val=True, novel_cls=0):
    # Adapted from https://github.com/acmi-lab/PU_learning/blob/main/estimator.py
    net.eval()
    pu_probs = None
    pu_targets = None
    with torch.no_grad():
        data = data.to(device)
        targets = torch.zeros_like(data.y, dtype=torch.int64)
        targets[data.y == novel_cls] = 1 # oracle novel label
        if model_type == "mlp":
            outputs = net(data.x)
        else:
            outputs = net(data.x, data.edge_index)

        if val:
            mask = torch.logical_and(data.src_mask, data.val_mask)
        else:
            mask = data.src_mask
        outputs = outputs[mask]
        targets = targets[mask]
        probs = torch.nn.functional.softmax(outputs, dim=-1)

        if pu_probs is None:
            pu_probs = probs.detach().cpu().numpy().squeeze()
            pu_targets = targets.cpu().numpy().squeeze()

        else:
            pu_probs = np.concatenate((pu_probs, \
                                       probs.detach().cpu().numpy().squeeze()))
            pu_targets = np.concatenate((pu_targets, \
                                         targets.cpu().numpy().squeeze()))

    return pu_probs, pu_targets

def DKW_bound(x, y, t, m, n, delta=0.1, gamma=0.01):
    # Copied from https://github.com/acmi-lab/PU_learning/blob/main/estimator.py

    temp = np.sqrt(np.log(1/delta)/2/n) + np.sqrt(np.log(1/delta)/2/m)
    bound = temp*(1+gamma)/(y/n)

    estimate = t

    return estimate, t - bound, t + bound

def BBE_estimator(pdata_probs, udata_probs, udata_targets):
    # Copied from https://github.com/acmi-lab/PU_learning/blob/main/estimator.py

    p_indices = np.argsort(pdata_probs)
    sorted_p_probs = pdata_probs[p_indices]
    u_indices = np.argsort(udata_probs[:,0])
    sorted_u_probs = udata_probs[:,0][u_indices]
    sorted_u_targets = udata_targets[u_indices]

    sorted_u_probs = sorted_u_probs[::-1]
    sorted_p_probs = sorted_p_probs[::-1]
    sorted_u_targets = sorted_u_targets[::-1]
    num = len(sorted_u_probs)
    estimate_arr = []

    upper_cfb = []
    lower_cfb = []

    i = 0
    j = 0
    num_u_samples = 0

    while (i < num):
        start_interval =  sorted_u_probs[i]
        k = i
        if (i<num-1 and start_interval> sorted_u_probs[i+1]):
            pass
        else:
            i += 1
            continue
        if (sorted_u_targets[i]==1):
            num_u_samples += 1

        while ( j<len(sorted_p_probs) and sorted_p_probs[j] >= start_interval):
            j+= 1

        if j>1 and i > 1:
            t = (i)*1.0*len(sorted_p_probs)/j/len(sorted_u_probs)
            estimate, lower , upper = DKW_bound(i, j, t, len(sorted_u_probs), len(sorted_p_probs))
            estimate_arr.append(estimate)
            upper_cfb.append( upper)
            lower_cfb.append( lower)
        i+=1

    if (len(upper_cfb) != 0):
        idx = np.argmin(upper_cfb)
        mpe_estimate = estimate_arr[idx]

        return mpe_estimate, lower_cfb, upper_cfb
    else:
        return 0.0, 0.0, 0.0