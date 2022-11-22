import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from metric import compute_ACC_mAP

def mean_exclude_negative(metric, axis=-1, de_diag=False):
    # metric: num_type x num_probe_view x num_gallery_view
    if de_diag:
        assert(metric.shape[1] == metric.shape[2])
        for i in range(metric.shape[0]):
            metric_i = metric[i, :, :]
            metric[i, :, :] = metric_i - np.diag(np.diag(metric_i))
    num_pos = np.sum(metric >= 0, axis=axis)
    metric[metric < 0] = 0
    sum_pos = np.sum(metric, axis=axis)
    if de_diag:
        mean = sum_pos / (num_pos - 1)
    else:
        mean = sum_pos / num_pos
    return mean

def cuda_euc_dist(x, y, part_average, chunk=5000):
    # x/y: num_probe/num_gallery * num_parts * part_dim
    num_probe, num_parts = x.size(0), x.size(1)
    num_gallery = y.size(0)
    assert(num_parts == y.size(1))
    dist = torch.zeros(num_probe, num_gallery)
    for p_start in range(0, num_probe, chunk):
        for g_start in range(0, num_gallery, chunk):
            p_end = p_start+chunk if p_start+chunk < num_probe else num_probe
            g_end = g_start+chunk if g_start+chunk < num_gallery else num_gallery
            chunk_x = x[p_start:p_end, :, :] # chunk * num_parts * part_dim
            chunk_y = y[g_start:g_end, :, :] # chunk * num_parts * part_dim
            if part_average:
                chunk_x = chunk_x.permute(1, 0, 2).contiguous() # num_parts * chunk * part_dim
                chunk_y = chunk_y.permute(1, 0, 2).contiguous() # num_parts * chunk * part_dim
                chunk_dist = torch.sum(chunk_x ** 2, 2).unsqueeze(2) + torch.sum(chunk_y ** 2, 2).unsqueeze(
                    2).transpose(1, 2) - 2 * torch.matmul(chunk_x, chunk_y.transpose(1, 2)) # num_parts * chunk * chunk
                chunk_dist = torch.sqrt(F.relu(chunk_dist)) # num_parts * chunk * chunk
                chunk_dist = torch.mean(chunk_dist, 0) # chunk * chunk
            else:
                chunk_x = chunk_x.view(chunk_x.size(0), -1) # chunk * (num_parts * part_dim)
                chunk_y = chunk_y.view(chunk_y.size(0), -1) # chunk * (num_parts * part_dim)
                chunk_dist = torch.sum(chunk_x ** 2, 1).unsqueeze(1) + torch.sum(chunk_y ** 2, 1).unsqueeze(
                    1).transpose(0, 1) - 2 * torch.matmul(chunk_x, chunk_y.transpose(0, 1)) # chunk * chunk
                chunk_dist = torch.sqrt(F.relu(chunk_dist)) # chunk * chunk
            dist[p_start:p_end, g_start:g_end] = chunk_dist    
            del chunk_x, chunk_y, chunk_dist
    return dist

def cuda_cos_dist(x, y, chunk=5000):
    # x/y: num_probe/num_gallery * num_parts * part_dim
    num_probe, num_parts = x.size(0), x.size(1)
    num_gallery = y.size(0)
    assert(num_parts == y.size(1))
    dist = torch.zeros(num_probe, num_gallery)
    for p_start in range(0, num_probe, chunk):
        for g_start in range(0, num_gallery, chunk):
            p_end = p_start+chunk if p_start+chunk < num_probe else num_probe
            g_end = g_start+chunk if g_start+chunk < num_gallery else num_gallery
            chunk_x = x[p_start:p_end, :, :] # chunk * num_parts * part_dim
            chunk_y = y[g_start:g_end, :, :] # chunk * num_parts * part_dim
            chunk_x = F.normalize(chunk_x, p=2, dim=2).permute(1, 0, 2) # num_parts * chunk * part_dim
            chunk_y = F.normalize(chunk_y, p=2, dim=2).permute(1, 0, 2) # num_parts * chunk * part_dim
            chunk_dist = 1 - torch.mean(torch.matmul(chunk_x, chunk_y.transpose(1, 2)), 0) # chunk * chunk
            dist[p_start:p_end, g_start:g_end] = chunk_dist
            del chunk_x, chunk_y, chunk_dist
    return dist

def evaluation(dataset, eval_data, probe_type_dict, gallery_type_dict, euc_or_cos_dist, \
                    part_dim, part_average, exclude_idt_view, remove_no_gallery, cross_view_gallery):
    print('#######################################')
    if euc_or_cos_dist == 'euc':
        print("Compute Euclidean Distance")
    elif euc_or_cos_dist == 'cos':
        print("Compute Cosine Distance")
    else:
        print('Illegal Distance Type')
        os._exit(0)
    print('#######################################')

    if cross_view_gallery:
        ACC, mAP = evaluation_cross_view_gallery(dataset, eval_data, probe_type_dict, gallery_type_dict, euc_or_cos_dist, \
                                                    part_dim, part_average, exclude_idt_view, remove_no_gallery)
    else:
        ACC, mAP = evaluation_single_view_gallery(dataset, eval_data, probe_type_dict, gallery_type_dict, euc_or_cos_dist, \
                                                    part_dim, part_average, exclude_idt_view, remove_no_gallery)
    
    return ACC, mAP

def evaluation_single_view_gallery(dataset, eval_data, probe_type_dict, gallery_type_dict, euc_or_cos_dist, \
                                    part_dim, part_average, exclude_idt_view, remove_no_gallery, print_info=False):
    feature, view, seq_type, label = eval_data
    label = np.asarray(label)
    view_list = sorted(list(set(view)))
    view_num = len(view_list)
    print('#######################################')
    feature = torch.from_numpy(feature).cuda()
    if part_dim > 0:
        feature = feature.view(feature.size(0), -1, part_dim).contiguous() # num_seqs * num_parts * part_dim
    else:
        feature = feature.unsqueeze(1).contiguous() # num_seqs * 1 * part_dim
    print("Feature Shape: ", feature.shape)
    print('#######################################')

    all_ACC = np.zeros([len(probe_type_dict[dataset]), view_num, view_num])
    all_mAP = np.zeros([len(probe_type_dict[dataset]), view_num, view_num])
    all_P_thres = np.zeros([len(probe_type_dict[dataset]), view_num, view_num])
    all_R_thres = np.zeros([len(probe_type_dict[dataset]), view_num, view_num])
    for (p, probe_type) in enumerate(probe_type_dict[dataset]):
        for gallery_type in gallery_type_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_type) & np.isin(view, [gallery_view])
                    gallery_y = label[gseq_mask]
                    gseq_mask = torch.from_numpy(np.asarray(gseq_mask, dtype=np.uint8))
                    gallery_x = feature[gseq_mask, :, :]

                    if remove_no_gallery:
                        pseq_mask = np.isin(seq_type, probe_type) & np.isin(view, [probe_view]) & np.isin(label, gallery_y)
                    else:
                        pseq_mask = np.isin(seq_type, probe_type) & np.isin(view, [probe_view])
                    probe_y = label[pseq_mask]
                    pseq_mask = torch.from_numpy(np.asarray(pseq_mask, dtype=np.uint8))
                    probe_x = feature[pseq_mask, :, :]

                    if print_info:
                        print('probe_type={}, gallery_type={}, probe_view={}, gallery_view={}, num_probe={}, num_gallery={}'.format( \
                                probe_type, gallery_type, probe_view, gallery_view, pseq_mask.sum(), gseq_mask.sum()))

                    if euc_or_cos_dist == 'euc':
                        dist = cuda_euc_dist(probe_x, gallery_x, part_average)
                    elif euc_or_cos_dist == 'cos':
                        dist = cuda_cos_dist(probe_x, gallery_x)
                    dist = dist.cpu().numpy()
                    eval_results = compute_ACC_mAP(dist, probe_y, gallery_y, 1)
                    del dist
                    all_ACC[p, v1, v2] = np.round(eval_results[0] * 100, 2)
                    all_mAP[p, v1, v2] = np.round(eval_results[1] * 100, 2)
    
    ACC = mean_exclude_negative(all_ACC, axis=-1, de_diag=exclude_idt_view)
    mAP = mean_exclude_negative(all_mAP, axis=-1, de_diag=exclude_idt_view)

    return ACC, mAP

def evaluation_cross_view_gallery(dataset, eval_data, probe_type_dict, gallery_type_dict, euc_or_cos_dist, \
                                    part_dim, part_average, exclude_idt_view, remove_no_gallery, print_info=False):
    feature, view, seq_type, label = eval_data
    label, view = np.asarray(label), np.asarray(view)
    view_list = sorted(list(set(view)))
    view_num = len(view_list)
    print('#######################################')
    feature = torch.from_numpy(feature).cuda()
    if part_dim > 0:
        feature = feature.view(feature.size(0), -1, part_dim).contiguous() # num_seqs * num_parts * part_dim
    else:
        feature = feature.unsqueeze(1).contiguous() # num_seqs * 1 * part_dim
    print("Feature Shape: ", feature.shape)
    print('#######################################')

    all_ACC = np.zeros([len(probe_type_dict[dataset]), view_num, 1])
    all_mAP = np.zeros([len(probe_type_dict[dataset]), view_num, 1])
    all_P_thres = np.zeros([len(probe_type_dict[dataset]), view_num, 1])
    all_R_thres = np.zeros([len(probe_type_dict[dataset]), view_num, 1])
    for (p, probe_type) in enumerate(probe_type_dict[dataset]):
        for gallery_type in gallery_type_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_type)
                gallery_y = label[gseq_mask]
                gallery_v = view[gseq_mask]
                gseq_mask = torch.from_numpy(np.asarray(gseq_mask, dtype=np.uint8))
                gallery_x = feature[gseq_mask, :, :]

                if remove_no_gallery:
                    pseq_mask = np.isin(seq_type, probe_type) & np.isin(view, [probe_view]) & np.isin(label, gallery_y)
                else:
                    pseq_mask = np.isin(seq_type, probe_type) & np.isin(view, [probe_view])
                probe_y = label[pseq_mask]
                probe_v = view[pseq_mask]
                pseq_mask = torch.from_numpy(np.asarray(pseq_mask, dtype=np.uint8))
                probe_x = feature[pseq_mask, :, :]

                if print_info:
                    print('probe_type={}, gallery_type={}, probe_view={}, num_probe={}, num_gallery={}'.format( \
                            probe_type, gallery_type, probe_view, pseq_mask.sum(), gseq_mask.sum()))

                if euc_or_cos_dist == 'euc':
                    dist = cuda_euc_dist(probe_x, gallery_x, part_average)
                elif euc_or_cos_dist == 'cos':
                    dist = cuda_cos_dist(probe_x, gallery_x)
                dist = dist.cpu().numpy()
                eval_results = compute_ACC_mAP(dist, probe_y, gallery_y, 1, probe_v, gallery_v, exclude_idt_view)
                all_ACC[p, v1, 0] = np.round(eval_results[0] * 100, 2)
                all_mAP[p, v1, 0] = np.round(eval_results[1] * 100, 2)
    
    ACC = mean_exclude_negative(all_ACC, axis=-1, de_diag=False)
    mAP = mean_exclude_negative(all_mAP, axis=-1, de_diag=False)

    return ACC, mAP