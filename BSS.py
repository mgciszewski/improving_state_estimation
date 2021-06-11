from postprocessingbyprojection import *
from misc_func import *
import numpy as np
import pandas as pd

class BinaryStateSequence:
    """
    A class used to represent binary state sequences

    Attributes
    ----------
    first_value : double
        the first state of the state sequence
    jumps : array of doubles
        an array of jumps/state transitions in the state sequence
    time_horizon: double
        the time horizon of the state sequence
    series: array of ints
        the state sequence in the form of arrays of 0-1s
    time: array of doubles
        an array of timepoints corresponding to specific states defined in series
    extended: boolean
        if False, the function was not extended for GTS or LTS measure, if True it was

    Methods
    -------
    find_approx(gamma)
        returns the projection of the BinaryStateSequences using DAG functionality
    """
    
    fs = 0.002 # sampling frequency
    
    def __init__(self, series = None, first_value = None, jumps = None, time_horizon = None):
        """
        Parameters
        ----------
        series: array of ints
            the state sequence in the form of arrays of 0-1s
        first_value : double
            the first state of the state sequence
        jumps : array of doubles
            an array of jumps/state transitions in the state sequence
        time_horizon: double
            the time horizon of the state sequence
        """
        # a following assumption on the input; either series is given or first_value, jumps and time_horizon are given
        if series is not None:
            self.first_value = series[0]
            self.series = pd.Series(series)
            # we can derive the time_horizon given sampling frequency
            self.time_horizon = len(self.series) * self.fs
            # time can be easily recreated using indices of the series and sampling frequency
            self.time = pd.Series(self.series.index * self.fs)
            # pandas diff() finds differences between subsequent values in pandas Series; taking absolute value of it returns a series of 0's and 1's, where 1 is synonymous with change of value in the series
            self.jumps = self.time[abs(self.series.diff()) > 0].tolist()
        else:
            self.first_value = first_value
            self.jumps = jumps
            self.time_horizon = time_horizon
            # different strategy for finding time is needed here, we calculate last index using casting to int the value of time_horizon divided by sampling frequency
            self.time = pd.Series([i * self.fs for i in range(round(time_horizon / self.fs))])
            # we initialize series to an array of first_value of the same length as time
            self.series = pd.Series([first_value for i in range(round(time_horizon / self.fs))])
            curr_time = 0
            curr_value = first_value
            for jump in jumps:
                if curr_time == 0:
                    curr_time = jump
                else:
                    curr_value = 1 - curr_value
                    # we change the value of the series in the interval with bounds being the previous jump and the current one
                    self.series[(self.time >= curr_time) & (self.time < jump)] = curr_value
                    curr_time = jump
            if jumps != []:
                self.series[(self.time >= curr_time) & (self.time <= time_horizon)] = 1 - curr_value
    
    def find_approx(self, gamma):
        """
        Returns the projection of the BinaryStateSequences using DAG functionality
        
        Parameters
        ----------
        gamma: double
            Lower bound on the lengths of the intervals (also double the weight of the jump)
            
        Returns
        -------
        BinaryStateSequence
            The projection of the BinaryStateSequence onto G_\gamma
        """
        if self.jumps != []:
            # first we divide the jumps of the BSS into subarrays in each of which the subsequent jumps are closer to each other than \gamma
            jump_seq_list = identify_jump_subseq(self.jumps, gamma)
            # jumps will store all jumps of the projection
            jumps = []
            # iteration over each separate subsequence
            for jump_seq in jump_seq_list:
                # if an array consists of only one jump there is nothing to project
                if len(jump_seq) > 1:
                    # we initialize the DAG structure and find the shortest path in the graph which is equivalent to projecting in the given interval
                    graph = DAG(jump_seq, gamma)
                    shortestPathTuple = graph.shortestPath()
                    # we skip first and last element of the second entry of the shortestPathTuple since those are only auxiliary values specific to the graphs (-np.inf and np.inf); second entry of the shortestPathTuple is the set of jumps of the projection in the given interval
                    jumps.append(shortestPathTuple[1][1:-1])
                else:
                    jumps.append(jump_seq)
        else:
            return BinaryStateSequence(first_value = self.first_value, jumps = [], time_horizon = self.time_horizon)

        return BinaryStateSequence(first_value = self.first_value, jumps = flatten_list(jumps), 
                                                   time_horizon = self.time_horizon)