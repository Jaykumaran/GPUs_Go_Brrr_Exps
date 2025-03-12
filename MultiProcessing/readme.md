### How ***not*** to use Multiprocessing on Single GPU Guide

Batch size: 128; num_workers in DataLoader (CPU specific)


Spawn Method:
- Normal Training: 19secs
- MultiProcessing with num_process = 6 --> shared_memory with num_workers = 0  ===> 1min 10secs
- With num_process = 1; with num_workers = 0 ==> 49secs
- With num_process = 1; num_workers = 8 --> Bad approach slower than all approach  ===> 3 min 29 secs

ForkServer Methods:
- MultiProcessing with num_process = 6 --> shared_memory with num_workers = 0  ===> 1min 19secs

Fork Method:
- MultiProcessing with num_process = 6 --> shared_memory with num_workers = 0  ===> 1min 10secs



References:
1. https://medium.com/%40heyamit10/how-to-use-pytorch-multiprocessing-0ddd2014f4fd


From article:

Multiprocessing is perfect for tasks that can run independently, like parallel data loading, batch processing, or augmentations that don’t require synchronous communication between processes.

DDP, on the other hand, shines in distributed model training. If you need to synchronize weights and gradients across GPUs, DDP is your go-to.
