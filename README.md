# NMS(Non-maximum Suppression) tools


<details open>
  <summary>Bounding Box NMS</summary>
    - bbox-nms
</details>

<details open>
  <summary>Bounding Box NMS - C language version</summary>

## Bounding Box NMS - C language version
### Benchmark (Single Batch / s)

- Each Single-Batch-Data have 2000 bounding boxes
- Each test run 1000 times to obtain results

- Speed(ms) : including: preprocessing, nms
- W/O processing (ms): only including: nms

| Algo / Paramters | Python    | C             | C                   | Batch Pallel C | Batch Pallel C      |
|------------------|-----------|---------------|---------------------|----------------|---------------------|
|  Batch Num    |  Speed(ms)|  Speed(ms)    |  W/O processing (ms)|  Speed(ms)     |  W/O processing (ms)|
|  1               | 0.611     |   **0.258**   |  0.211              |  0.834         |  0.735              |
|  10              | 0.610     |   **0.256**   |  0.211              |  0.343         |  0.175              |
|  100             | 0.603     |   **0.260**   |  0.214              |  0.354         |  0.094              |

### Usage: Refer to batch_parallel_nms.py

```Python 
num_classes = 80
score_thr = 0.5
nms_thr = 0.5

batched_bboxes = [np.ones((2000, 4)), np.ones((123, 4)), np.ones((321, 4)), ...]
batched_scores = [np.ones((2000, num_classes)), np.ones((123, num_classes)), np.ones((321, num_classes)), ...]

nms_c = Batch_Parallel_Nms()

# NMS
for boxes, scores in zip(batched_bboxes, batched_scores):
            indices_to_keep, nms_out_cls = nms_c.nms(boxes, scores, score_thr, nms_thr)

# BATCH PARALLEL
indices_to_keep, nms_out_cls = nms_c.batch_parallel_nms(batched_bboxes, batched_scores, score_thr, nms_thr)
```

### If there is any modified
```bash
gcc -O3 -msse2 -mfpmath=sse -ftree-vectorizer-verbose=5 -fopenmp -fPIC -shared -o c/compiled/batch_parallel_nms.so c/batch_parallel_nms.c 
```
</details>