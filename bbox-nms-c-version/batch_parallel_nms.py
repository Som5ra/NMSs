import os
import numpy as np

from ctypes import *
from numpy .ctypeslib import ndpointer

import time



TIME1 = 0
TIME2 = 0

class Batch_Parallel_Nms:
    def __init__(self, dll:str = None) -> None:
        if dll is None:
            dll = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'compiled/batch_parallel_nms.so')
        self.dll = CDLL(dll)


        self.dll.batch_parallel_nms.argtypes = [
            ndpointer(c_uint64, flags="C_CONTIGUOUS"), # bboxes
            ndpointer(c_uint64, flags="C_CONTIGUOUS"),
            ndpointer(c_double, flags="C_CONTIGUOUS"), # scores
            ndpointer(c_uint64, flags="C_CONTIGUOUS"),

            ndpointer(c_uint64, flags="C_CONTIGUOUS"), 
            c_uint64,

            c_double, # score_thr
            c_double, # nms_thr

            ndpointer(c_uint64, flags="C_CONTIGUOUS"), # batch_valid_indices
            ndpointer(c_uint64, flags="C_CONTIGUOUS"), # batch_valid_indices_cls_id
            ndpointer(c_uint64, flags="C_CONTIGUOUS"),
        ]

        self.dll.multiclass_nms_class_aware_cpu.argtypes = [
            ndpointer(c_uint64, flags="C_CONTIGUOUS"), # bboxes
            ndpointer(c_uint64, flags="C_CONTIGUOUS"),
            ndpointer(c_double, flags="C_CONTIGUOUS"), # scores
            ndpointer(c_uint64, flags="C_CONTIGUOUS"),
            c_double, # score_thr
            c_double, # nms_thr
            ndpointer(c_uint64, flags="C_CONTIGUOUS"), # valid_indices
            ndpointer(c_uint64, flags="C_CONTIGUOUS"), # valid_indices_cls_id
            ndpointer(c_uint64, flags="C_CONTIGUOUS"),
        ]

    def batch_parallel_nms(self, bboxes, scores, score_thr, nms_thr):
        global TIME2
        batch_size = len(bboxes)
        batch_num_recorder = np.zeros(batch_size, dtype=np.uint64, order='C')
        for i in range(batch_size):
            batch_num_recorder[i] = len(bboxes[i])
        # bboxes = np.ascontiguousarray(np.vstack(bboxes, dtype=np.uint64))
        # scores = np.ascontiguousarray(np.vstack(scores, dtype=np.float64))
        bboxes = np.ascontiguousarray(np.vstack(bboxes), dtype=np.uint64)
        scores = np.ascontiguousarray(np.vstack(scores), dtype=np.float64)
        batch_valid_indices = np.full(scores.shape[0] * scores.shape[1], fill_value=0, dtype=np.uint64, order='C')
        batch_valid_indices_cls_id = np.zeros(scores.shape[0] * scores.shape[1], dtype=np.uint64, order='C')
        res_length = np.array([0] * batch_size, dtype=np.uint64, order='C')

        time1 = time.time()
        ret = self.dll.batch_parallel_nms(bboxes,
                            np.array(bboxes.shape, dtype=np.uint64),
                            scores,
                            np.array(scores.shape, dtype=np.uint64),

                            batch_num_recorder,
                            batch_size,
                            
                            score_thr,
                            nms_thr,

                            batch_valid_indices,
                            batch_valid_indices_cls_id,
                            res_length)
        TIME2 += time.time() - time1
        cur = 0
        indices_to_keeps = []
        nms_out_clss = []
        for i in range(batch_size):
            _length = int(batch_num_recorder[i] * scores.shape[1])
            indices_to_keep = batch_valid_indices[cur: cur + _length][: res_length[i]]
            indices_to_keeps.append(indices_to_keep)
            nms_out_cls = batch_valid_indices_cls_id[cur: cur + _length][: res_length[i]]
            nms_out_clss.append(nms_out_cls)
            cur += _length

        return indices_to_keeps, nms_out_clss

    def nms(self, bboxes, scores, score_thr, nms_thr):
        global TIME1
        valid_indices = np.full(scores.shape[0] * scores.shape[1], fill_value=0, dtype=np.uint64, order='C')
        valid_indices_cls_id = np.zeros(scores.shape[0] * scores.shape[1], dtype=np.uint64, order='C')
        res_length = np.array([0], dtype=np.uint64,)
        time1 = time.time()
        ret = self.dll.multiclass_nms_class_aware_cpu(bboxes,
                            np.array(bboxes.shape, dtype=np.uint64),
                            scores,
                            np.array(scores.shape, dtype=np.uint64),
                            score_thr,
                            nms_thr,
                            valid_indices,
                            valid_indices_cls_id,
                            res_length)
        TIME1 += time.time() - time1
        return valid_indices[: res_length[0]], valid_indices_cls_id[: res_length[0]]
    


def batch_parallel_nms_example():
    '''
    Input: 
        bboxes: [(num_boxes1, 4), (num_boxes2, 4), (num_boxes3, 4), (num_boxes4, 4)...]
        scores: [(num_boxes1, num_classes), (num_boxes2, num_classes), (num_boxes3, num_classes), (num_boxes4, num_classes)...]
    Output:
        Incices_to_keep: [
            [bbox_idx1, bbox_idx2, bbox_idx3, bbox_idx4...], 
            [bbox_idx1, bbox_idx2, ...],
            [bbox_idx1, bbox_idx2, bbox_idx3, bbox_idx4...],
            [bbox_idx1, bbox_idx2, bbox_idx3...],
        ]
        nms_out_cls: [
            [0, 0, 0, 4...], 
            [0, 1, ...],
            [0, 0, 5, 6...],
            [0, 1, 3...],
        ]
    '''

    import json

    test_bboxes_json_file = '/media/risksis/HDD_1/railway_safety_2023_movement/test_bboxes.json'

    with open(test_bboxes_json_file, 'r') as fp:
        data = json.load(fp)



    bboxes = [data["bounding boxes"], data["bounding boxes"][: 10], data["bounding boxes"][: 5]]
    scores = [data["scores"], data["scores"][: 10], data["scores"][: 5]]
    
    score_thr = 0.5
    nms_thr = 0.5

    nms_c = Batch_Parallel_Nms()
    indices_to_keep, nms_out_cls = nms_c.batch_parallel_nms(bboxes, scores, score_thr, nms_thr)
    print("bounding boxes indices: ",indices_to_keep)
    print("bounding boxes cls: ", nms_out_cls)

def nms_example():
    '''
    Input: 
        bboxes: (num_boxes, 4)
        scores: (num_boxes, num_classes)
    Output:
        Incices_to_keep: [bbox_idx1, bbox_idx2, bbox_idx3, bbox_idx4,...]

        nms_out_cls: [0, 0, 0, 4,...]
    '''

    import json

    test_bboxes_json_file = '/media/risksis/HDD_1/railway_safety_2023_movement/test_bboxes.json'

    with open(test_bboxes_json_file, 'r') as fp:
        data = json.load(fp)



    bboxes = np.asarray(data["bounding boxes"], order='C', dtype=np.uint64)
    scores = np.asarray(data["scores"], order='C', dtype=np.float64)
    
    score_thr = 0.5
    nms_thr = 0.5

    nms_c = Batch_Parallel_Nms()
    indices_to_keep, nms_out_cls = nms_c.nms(bboxes, scores, score_thr, nms_thr)
    print("bounding boxes indices: ", indices_to_keep)
    print("bounding boxes cls: ", nms_out_cls)



def nms_performance_compare(batch_num = 100, run_times = 1000):
    score_thr = 0.5
    nms_thr = 0.5

    import json
    from tqdm import trange
    from models.boundingbox_detector.nms import multiclass_nms_class_aware_cpu

    test_bboxes_json_file = '/media/risksis/HDD_1/railway_safety_2023_movement/test_bboxes.json'

    with open(test_bboxes_json_file, 'r') as fp:
        data = json.load(fp)



    data_bboxes = np.asarray(data["bounding boxes"], order='C', dtype=np.uint64)
    data_scores = np.asarray(data["scores"], order='C', dtype=np.float64)


    batched_bboxes = [data_bboxes] * batch_num
    batched_scores = [data_scores] * batch_num
    
    nms_c = Batch_Parallel_Nms()

    time1 = time.time()
    for i in trange(run_times):
        for boxes, scores in zip(batched_bboxes, batched_scores):
            indices_to_keep, nms_out_cls = multiclass_nms_class_aware_cpu(boxes, scores, score_thr, nms_thr)
    time2 = time.time() 
    print("nms python consumption: ",(time2 - time1) / batch_num / run_times * 1000)

    for i in trange(run_times):
        indices_to_keep, nms_out_cls = nms_c.batch_parallel_nms(batched_bboxes, batched_scores, score_thr, nms_thr)
    time3 = time.time() 
    print("batched parallel consumption: ",(time3 - time2)  / batch_num / run_times * 1000)

    for i in trange(run_times):
        for boxes, scores in zip(batched_bboxes, batched_scores):
            indices_to_keep, nms_out_cls = nms_c.nms(boxes, scores, score_thr, nms_thr)
    time4 = time.time() 
    print("nms c version consumption: ", (time4 - time3) / batch_num / run_times * 1000)


    print("pure c nms: ", TIME1 / batch_num / run_times * 1000)
    print("pure c batched nms: ", TIME2 / batch_num / run_times * 1000)

if __name__ == '__main__':
    # batch_parallel_nms_example()
    # nms_example()
    nms_performance_compare()



