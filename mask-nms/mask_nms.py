import numba
from numba import njit
from numba.typed import List as NList
from numba.types import int64 as nb_int64

import numpy as np

@njit
def mask_overlap(mask1, mask2):
    _union = np.count_nonzero(np.bitwise_or(mask1, mask2))
    if _union == 0:
        return 0
    _inter = np.count_nonzero(np.bitwise_and(mask1, mask2))
    return _inter / _union


@njit
def mask_nms_cpu(masks, scores, score_thr = 0.5, nms_thr = 0.5):

    raw_indices = np.arange(0, scores.shape[0])
    score_thr_mask = (scores >= score_thr)
    masks = masks[score_thr_mask]
    scores = scores[score_thr_mask]
    raw_indices = raw_indices[score_thr_mask]

    order = scores.argsort()[::-1]
    keep = NList.empty_list(nb_int64)
    while order.size > 0:
        i = order[0]
        keep.append(raw_indices[i])

        ovr = np.asarray([mask_overlap(masks[i], masks[_order]) for _order in order[1:]])

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep

def multiclass_mask_nms_class_aware_cpu(masks, scores, score_thr, nms_thr):
    """
    Mutli class mask NMS (class-aware)

    Class-unaware: a proposal can belong to mutiple single class
    
    inputs:
        masks: NDArray (num_masks, W, H) (type: Boolean)
        scores: NDArray (num_masks, num_classes) in [0, 1] 
        
    output:
        [NDArray of indices to keep, NDArray of class id]
    """

    valid_idx = []
    valid_idx_class_id = []

    if np.bool_ != masks.dtype:
        raise Exception("Masks must be boolean type")
        
    num_classes = scores.shape[-1]
    for cls_id in range(num_classes):
        class_valid_idx = mask_nms_cpu(masks, scores[:, cls_id], score_thr=score_thr, nms_thr=nms_thr)
        valid_idx.extend(class_valid_idx)
        valid_idx_class_id.extend([ cls_id for _ in range(len(class_valid_idx))])
    
    return np.array(valid_idx), np.array(valid_idx_class_id)
