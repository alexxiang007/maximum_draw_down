import numpy as np
import pandas as pd
import math
from numpy.lib.stride_tricks import as_strided

class Maximum_draw_down_calculator():
    def __init__(self):
        pass

    def two_dimensional_view(self, data_array, window_size):
        """
        To create a memory efficient 2d view of 1d numpy array.

        Example:
        input:
            1d numpy array [1,2,3,4,5,6,7,8,9,10]
            window_size 4
        output:
            2d memory view of the 1d array, shape = (7, 4)
            [
             [1,2,3,4],
             [2,3,4,5],
             [3,4,5,6],
             [4,5,6,7],
             [5,6,7,8],
             [6,7,8,9],
             [7,8,9,10]
            ]

        note:
            data is not copied
        """
        return  as_strided(data_array,
                            shape=(data_array.size - window_size + 1, window_size), 
                            strides=(data_array.strides[0], data_array.strides[0]))


    def rolling_max_draw_down(self, prices, window_size, log_return = False):
        """
        To compute the rolling maximum draw down price a price array.

        input:
            prices: 1d numpy array of prices

        output:
            1d numpy array with length with first window_size-1 elements to be N.A
        """

        if log_return:
            log_prices = np.array([math.log(price) for price in prices])

        prices_2d_view = self.two_dimensional_view(prices, window_size) if not log_return else self.two_dimensional_view(log_prices, window_size)
        rolling_max = np.maximum.accumulate(prices_2d_view, axis=1)
        
        if log_return:
            draw_down = prices_2d_view - rolling_max
        else:
            draw_down = prices_2d_view / rolling_max  - 1

        empty_head = np.empty(window_size-1, dtype=draw_down.dtype)
        empty_head.fill(np.nan)
        mdd_tail = draw_down.min(axis=1)

        return np.concatenate([empty_head, mdd_tail])

if __name__ == "__main__":
    prices =  np.random.uniform(90,110,11)
    mdd_calculator = Maximum_draw_down_calculator()
    mdd1 = mdd_calculator.rolling_max_draw_down(prices, 10, log_return = False)
    mdd2 = mdd_calculator.rolling_max_draw_down(prices, 10, log_return = True)
    print(prices, "\n", mdd1, "\n", mdd2)
    # print(prices)
    # print(mdd)