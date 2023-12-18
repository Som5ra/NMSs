#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

typedef unsigned long int ulong;
typedef unsigned int uint;

void batch_parallel_nms(
    ulong* batch_bboxes, // boxes: NDArray (num_boxes, 4) in xyxy 
    ulong* batch_bboxes_shape, // (num_boxes, 4)
    double* batch_scores, // scores: NDArray (num_boxes, num_classes) in [0, 1]
    ulong* batch_scores_shape, // (num_boxes, num_classes)
    ulong* batch_num_recorder,
    ulong batch_size,
    double score_thr,
    double nms_thr,
    ulong* batch_valid_idx, 
    ulong* batch_valid_idx_class_id,
    ulong* batch_result_length
);


bool* nms_cpu(
    ulong* bboxes, // boxes: NDArray (num_boxes, 4) in xyxy 
    ulong* bboxes_shape, // (num_boxes, 4)
    double* scores, // scores: NDArray (num_boxes, 1) in [0, 1]
    ulong* scores_shape, // (num_boxes, 1)
    double score_thr,
    double nms_thr
);


void multiclass_nms_class_aware_cpu(
    ulong* bboxes, // boxes: NDArray (num_boxes, 4) in xyxy 
    ulong* bboxes_shape, // (num_boxes, 4)
    double* scores, // scores: NDArray (num_boxes, num_classes) in [0, 1]
    ulong* scores_shape, // (num_boxes, num_classes)
    double score_thr,
    double nms_thr,

    ulong* valid_idx,
    ulong* valid_idx_class_id,
    ulong* result_length
);

void batch_parallel_nms(
    ulong* batch_bboxes, // boxes: NDArray (num_boxes, 4) in xyxy 
    ulong* batch_bboxes_shape, // (num_boxes, 4)
    double* batch_scores, // scores: NDArray ( num_boxes, num_classes) in [0, 1]
    ulong* batch_scores_shape, // (num_boxes, num_classes)

    ulong* batch_num_recorder,
    ulong batch_size,

    double score_thr,
    double nms_thr,

    ulong* batch_valid_idx,
    ulong* batch_valid_idx_class_id,
    ulong* batch_result_length
){

    ulong* batch_end_pos = malloc(sizeof(ulong) * batch_size);
    ulong tmp_recorder = 0;
    for(size_t i = 0; i < batch_size; i++){
        tmp_recorder += batch_num_recorder[i];
        batch_end_pos[i] = tmp_recorder;
    }

    #pragma omp parallel for
    for(size_t batch_idx = 0; batch_idx < batch_size; batch_idx++){

        ulong this_batch_size = batch_num_recorder[batch_idx];
        ulong this_batch_start_pos = 0;
        if (batch_idx != 0){
            this_batch_start_pos = batch_end_pos[batch_idx - 1];
        }
        
        ulong this_batch_end_pos = batch_end_pos[batch_idx];

        ulong bboxes_shape[] = {this_batch_size, 4};
        ulong scores_shape[] = {this_batch_size, batch_scores_shape[1]};
        
        ulong* bboxes = malloc(sizeof(ulong) * this_batch_size * 4);
        memcpy(bboxes, batch_bboxes + this_batch_start_pos * 4, sizeof(ulong) * (this_batch_end_pos - this_batch_start_pos) * 4);


        double* scores = malloc(sizeof(double) * this_batch_size * batch_scores_shape[1]);
        memcpy(scores, batch_scores + this_batch_start_pos * batch_scores_shape[1],  sizeof(double) * (this_batch_end_pos - this_batch_start_pos) * batch_scores_shape[1]);

        ulong* valid_idx = batch_valid_idx + this_batch_start_pos * batch_scores_shape[1];
        ulong* valid_idx_class_id = batch_valid_idx_class_id + this_batch_start_pos * batch_scores_shape[1];
        ulong* result_length = batch_result_length + batch_idx;
        multiclass_nms_class_aware_cpu(
            bboxes, 
            bboxes_shape, 
            scores, 
            scores_shape,
            score_thr,
            nms_thr,

            valid_idx,
            valid_idx_class_id,
            result_length
        );


        free(bboxes);
        free(scores);
    }
    free(batch_end_pos);
}




void multiclass_nms_class_aware_cpu(
    ulong* bboxes, // boxes: NDArray (num_boxes, 4) in xyxy 
    ulong* bboxes_shape, // (num_boxes, 4)
    double* scores, // scores: NDArray (num_boxes, num_classes) in [0, 1]
    ulong* scores_shape, // (num_boxes, num_classes)
    double score_thr,
    double nms_thr,

    ulong* valid_idx,
    ulong* valid_idx_class_id,
    ulong* result_length
){
    int bboxes_length = bboxes_shape[0];
    int scores_length = scores_shape[0];

    ulong num_classes = scores_shape[1];
    for(size_t cls_id = 0; cls_id < num_classes; cls_id++){
        double* scores_cls = malloc(sizeof(double) * scores_length);
        int scores_cls_pointer = 0;
        
        for(size_t scores_cls_offset = cls_id; scores_cls_pointer < scores_length; scores_cls_offset += num_classes){
            scores_cls[scores_cls_pointer++] = scores[scores_cls_offset];
            
        }
        ulong scores_cls_shape[] = {scores_shape[0], 1}; 
        bool* valid_idx_mask = nms_cpu(bboxes, bboxes_shape, scores_cls, scores_cls_shape, score_thr, nms_thr);
        for(size_t keep_idx = 0; keep_idx < bboxes_shape[0]; keep_idx++){
            if (valid_idx_mask[keep_idx] == true){
                valid_idx[result_length[0]] = keep_idx;
                valid_idx_class_id[result_length[0]++] = cls_id;
            }
        }
        free(scores_cls);
        free(valid_idx_mask);
    }
}



// NMS Implementation of C.
bool* nms_cpu(
    ulong* bboxes, // boxes: NDArray (num_boxes, 4) in xyxy 
    ulong* bboxes_shape, // (num_boxes, 4)
    double* scores, // scores: NDArray (num_boxes, 1) in [0, 1]
    ulong* scores_shape, // (num_boxes, 1)
    double score_thr,
    double nms_thr
){



    bool* score_thr_mask = malloc(sizeof(bool) * scores_shape[0]);
    int valid_bboxes_count = 0;
    for(size_t score_thr_mask_id = 0; score_thr_mask_id < scores_shape[0]; score_thr_mask_id++){
        if(scores[score_thr_mask_id] > score_thr)
            valid_bboxes_count += 1;
    }

    ulong* valid_boxes = malloc(sizeof(ulong) * valid_bboxes_count * bboxes_shape[1]);
    ulong* x1 = malloc(sizeof(ulong) * valid_bboxes_count);
    ulong* y1 = malloc(sizeof(ulong) * valid_bboxes_count);
    ulong* x2 = malloc(sizeof(ulong) * valid_bboxes_count);
    ulong* y2 = malloc(sizeof(ulong) * valid_bboxes_count);

    double* valid_scores = malloc(sizeof(double) * valid_bboxes_count);
    ulong* valid_raw_indices = malloc(sizeof(ulong) * valid_bboxes_count);

    int valid_box_idx = 0;
    // #pragma omp parallel for
    for(size_t score_thr_mask_id = 0; score_thr_mask_id < scores_shape[0]; score_thr_mask_id++){
        if(scores[score_thr_mask_id] > score_thr){

            for(size_t box_offset = 0; box_offset < bboxes_shape[1]; box_offset++){ 
                valid_boxes[valid_box_idx * bboxes_shape[1] + box_offset] = bboxes[score_thr_mask_id * bboxes_shape[1] + box_offset];

                if(box_offset == 0) x1[valid_box_idx] = bboxes[score_thr_mask_id * bboxes_shape[1] + box_offset];
                else if(box_offset == 1) y1[valid_box_idx] = bboxes[score_thr_mask_id * bboxes_shape[1] + box_offset];
                else if(box_offset == 2) x2[valid_box_idx] = bboxes[score_thr_mask_id * bboxes_shape[1] + box_offset];
                else y2[valid_box_idx] = bboxes[score_thr_mask_id * bboxes_shape[1] + box_offset];
            }
            valid_scores[valid_box_idx] = scores[score_thr_mask_id];
            valid_raw_indices[valid_box_idx] = score_thr_mask_id;
            valid_box_idx += 1;
        }
    }
    

    bool* keep = malloc(sizeof(bool) * valid_bboxes_count);
    // #pragma omp parallel for
    for(size_t keep_id = 0; keep_id < valid_bboxes_count; keep_id++){
        keep[keep_id] = false;
    }

    double* areas = malloc(sizeof(double) * valid_bboxes_count);
    // #pragma omp parallel for
    for(size_t areas_id = 0; areas_id < valid_bboxes_count; areas_id++){
        areas[areas_id] = (x2[areas_id] - x1[areas_id] + 1) * (y2[areas_id] - y1[areas_id] + 1);
    }


    int num_left_bboxes = valid_bboxes_count;
    bool* left_bboxes = malloc(sizeof(bool) * num_left_bboxes);
    // #pragma omp parallel for
    for(size_t left_bboxes_id = 0; left_bboxes_id < valid_bboxes_count; left_bboxes_id++){
        left_bboxes[left_bboxes_id] = true;
    }


    while(num_left_bboxes > 0){
        int best_index = -1;
        double highest_score = -1;
        for(size_t i = 0; i < valid_bboxes_count; i++){
            if (keep[i] == true || left_bboxes[i] == false) continue; 
            if (valid_scores[i] > highest_score){
                best_index = i;
                highest_score = valid_scores[i];
            }
        }
        
        keep[best_index] = true;
        left_bboxes[best_index] = false;
        num_left_bboxes -= 1;
        for(size_t i = 0; i < valid_bboxes_count; i++){
            if(keep[i] == true || left_bboxes[i] == false) 
                continue;
            ulong xx1 = MAX(x1[best_index], x1[i]);
            ulong yy1 = MAX(y1[best_index], y1[i]);
            ulong xx2 = MIN(x2[best_index], x2[i]);
            ulong yy2 = MIN(y2[best_index], y2[i]);

            double w = MAX(0.0, xx2 - xx1 + 1);
            double h = MAX(0.0, yy2 - yy1 + 1);
            double inter = w * h;
            double ovr = inter / (areas[best_index] + areas[i] - inter);
            // printf("%ld %ld overlapping: %lf\n", valid_raw_indices[best_index], valid_raw_indices[i], ovr);
            if (ovr > nms_thr){
                left_bboxes[i] = false;
                num_left_bboxes -= 1;
            }
        } 
    }

    bool* result = malloc(sizeof(bool) * scores_shape[0]);
    // #pragma omp parallel for
    for(size_t result_idx = 0; result_idx < scores_shape[0]; result_idx++){ 
        result[result_idx] = false;
    }
    for(size_t keep_idx = 0; keep_idx < valid_bboxes_count; keep_idx++){
        if (keep[keep_idx] == true){
            result[valid_raw_indices[keep_idx]] = true;
        }
    }

    free(score_thr_mask);
    free(valid_boxes);
    free(x1);
    free(y1);
    free(x2);
    free(y2);
    free(valid_scores);
    free(valid_raw_indices);
    free(areas);
    free(left_bboxes);
    free(keep);

    return result;
}
