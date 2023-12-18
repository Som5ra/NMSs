import numpy as np

def nms_cpu(boxes, scores, score_thr, nms_thr):
    """
    Single class NMS
    
    inputs:
        boxes: NDArray (num_boxes, 4) in xyxy
        scores: NDArray (num_boxes, 1) in [0,1]
    
    output:
        NDArray of indices to keep
    """

    raw_indices = np.arange(0, scores.shape[0])
    score_thr_mask = scores >= score_thr
    boxes = boxes[score_thr_mask]
    scores = scores[score_thr_mask]
    raw_indices = raw_indices[score_thr_mask]

    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(raw_indices[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return np.array(keep)


def multiclass_nms_class_unaware_cpu(boxes, scores, score_thr, nms_thr):
    """
    Mutli class NMS (class-unaware)

    Class-unaware: a proposal can only belong to a single class
    
    inputs:
        boxes: NDArray (num_boxes, 4) in xyxy
        scores: NDArray (num_boxes, num_classes) in [0, 1]
    
    output:
        [NDArray of indices to keep, NDArray of class id]
    """
    
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange( scores.shape[0]), cls_inds]

    valid_idx = nms_cpu(boxes=boxes, scores=cls_scores, score_thr=score_thr, nms_thr=nms_thr)
    valid_idx_class_id = np.take(cls_inds, valid_idx)

    return valid_idx, valid_idx_class_id


def multiclass_nms_class_aware_cpu(boxes, scores, score_thr, nms_thr):
    """
    Mutli class NMS (class-aware)

    Class-unaware: a proposal can belong to mutiple single class
    
    inputs:
        boxes: NDArray (num_boxes, 4) in xyxy
        scores: NDArray (num_boxes, num_classes) in [0, 1]
    
    output:
        [NDArray of indices to keep, NDArray of class id]
    """

    valid_idx = []
    valid_idx_class_id = []

    num_classes = scores.shape[-1]
    for cls_id in range(num_classes):
        class_valid_idx = nms_cpu(boxes, scores[:, cls_id], score_thr=score_thr, nms_thr=nms_thr)
        valid_idx.extend(class_valid_idx)
        valid_idx_class_id.extend([ cls_id for _ in range(len(class_valid_idx))])
    
    
    return np.array(valid_idx), np.array(valid_idx_class_id)
