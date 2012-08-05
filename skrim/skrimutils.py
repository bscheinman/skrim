import numpy as np

pad_ones = lambda x: np.append(np.ones([x.shape[0], 1]), x, 1)
