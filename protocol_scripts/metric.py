import numpy as np

def compute_ACC_mAP(distmat, q_pids, g_pids, rank, q_views=None, g_views=None, exclude_idt_view=False, print_info=False):
    num_q, num_g = distmat.shape
    # indices = np.argsort(distmat, axis=1)
    # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_ACC = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        q_idx_dist = distmat[q_idx]
        q_idx_glabels = g_pids
        if print_info:
            print('number of gallery before identical-view cases: {}'.format(len(q_idx_glabels)))
        if q_views is not None and g_views is not None and exclude_idt_view:
            q_idx_mask = np.isin(g_views, [q_views[q_idx]], invert=True) | np.isin(g_pids, [q_pids[q_idx]], invert=True)
            q_idx_dist = q_idx_dist[q_idx_mask]
            q_idx_glabels = q_idx_glabels[q_idx_mask]
            if print_info:
                print('number of gallery after identical-view cases: {}'.format(len(q_idx_glabels)))
        assert(len(q_idx_glabels) > 0), "no gallery after excluding identical-view cases"
        q_idx_indices = np.argsort(q_idx_dist)
        q_idx_matches = (q_idx_glabels[q_idx_indices] == q_pids[q_idx]).astype(np.int32)

        # binary vector, positions with value 1 are correct matches
        # orig_cmc = matches[q_idx]
        orig_cmc = q_idx_matches
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_ACC.append(cmc[rank-1])
        
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        if print_info:
            print('number of ground truth: {}'.format(num_rel))
        if num_rel > 0:
            num_valid_q += 1.
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    all_ACC = np.asarray(all_ACC).astype(np.float32)
    ACC = np.mean(all_ACC)
    mAP = np.mean(all_AP)

    return ACC, mAP