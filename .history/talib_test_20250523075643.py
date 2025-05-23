import talib
import numpy as np
a = np.arange(10, dtype='float64')
print("Input array:", a)
print("SMA(1):", talib.SMA(a, timeperiod=1))
