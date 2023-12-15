


### Mutli class mask NMS (class-aware)

- Class-unaware: a proposal can belong to mutiple single class

- inputs:
    - masks: NDArray (num_masks, W, H) (type: Boolean)
    - scores: NDArray (num_masks, num_classes) in [0, 1] 
    - score_thr: float (score threshold of bounding box)
    - nms_thr: float (intersection threshold of mask)
- output:
    - [NDArray of indices to keep, NDArray of class id]

```
pip install numba
pip install numpy
```