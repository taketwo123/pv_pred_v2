import numpy as np 

def timefeature(array):
# month

    array[::,0] = array[::,0] / 12 - 0.5
# day
    array[::,1] = (array[::,1] - 1 ) / 30 - 0.5
# hour
    array[::,2] = array[::,2] / 23 - 0.5
# minute
    array[::,3] = array[::,3] / 59 - 0.5

    return array
