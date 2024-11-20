"""
------------------------------------------------------------
Code Source:
Author: WegraLee
Repository: deep-learning-from-sratch-2
URL: https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/common/np.py
License: MIT License
Accessed: 2024-11-20

Modified to work on jupyter notebook
------------------------------------------------------------
"""

# coding: utf-8
from config import GPU


if GPU:
    import cupy as np
    import cupyx
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    # np.add.at = cupyx.scatter_add

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')
else:
    import numpy as np