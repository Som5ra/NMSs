```bash
gcc -O3 -msse2 -mfpmath=sse -ftree-vectorizer-verbose=5 -fopenmp -fPIC -shared -o c/compiled/batch_parallel_nms.so c/batch_parallel_nms.c 
```


