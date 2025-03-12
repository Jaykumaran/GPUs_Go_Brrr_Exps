Batch size: 128

- Normal Training: 19secs
- MultiProcessing with num_process = 6 --> shared_memory with num_workers = 0  =hh==> 1min 10secs
- With num_process = 1; with num_workers = 0 ==> 49secs
- With num_process = 1; num_workers = 8 --> Bad approach slower than all approach
