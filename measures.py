from BSS import *
import numpy as np
from misc_func import integral, mirrored, identify_jump_subseq

class Params:
    """
    A class used to represent parameters of the GTS/LTS measure

    Attributes
    ----------
    w : double
        the weight of the timing uncertainty errors
    s : double
        timing uncertainty neighbourhood radius
    lam: double
        the weight of the total variation penalty
    gamma: double
        the lower bound on the duration of the states

    Methods
    -------
    __str__()
        prints `nice' version of the parameters
    change_params(series)
        changes parameters (any or all of them)
    """
    
    def __init__(self, w = None, s = None, lam = None, gamma = None):
        """
        Parameters
        ----------
        w : double, default is 0.2
            the weight of the timing uncertainty errors
        s : double, default is 0.5
            timing uncertainty neighbourhood radius
        lam: double, default is 0.05
            the weight of the total variation penalty
        gamma: double, default is 0.5
            the lower bound on the duration of the states
        """
        self.w = w
        self.s = s
        self.lam = lam
        self.gamma = gamma
    
    def __str__(self): 
        """
        Prints `nice' version of the parameters
        
        Returns
        -------
        A string of all parameters listed, separated by a comma
        """
        return f"w: {self.w}, s: {self.s}, lam: {self.lam}, gamma: {self.gamma}"
    
    def change_params(self, w = None, s = None, lam = None, gamma = None):
        """
        Changes parameters (any or all of them)
        
        Parameters
        ----------
        w : double
            the weight of the timing uncertainty errors
        s : double
            timing uncertainty neighbourhood radius
        lam: double
            the weight of the total variation penalty
        gamma: double
            the lower bound on the duration of the states
        """
        if w is not None:
            self.w = w
        if s is not None:
            self.s = s
        if lam is not None:
            self.lam = lam
        if gamma is not None:
            self.gamma = gamma

class GTS_measure:
    """
    A class used to represent GTS measure, a performance measure for binary classification of time series

    Attributes
    ----------
    params: Params
        parameters of the GTS measure

    Methods
    -------
    change_params(w, s, lam, gamma)
        changes parameters of the GTS measure
    performance(ground_truth, estimated_labels)
        calculates GTS performance measure of the estimated_labels given ground_truth labels
    error_characterization(ground_truth, estimated_labels)
        returns dictionary of errors as defined in GTS performance measure
    """
    def __init__(self, w = 0.6, s = 0.5, lam = 0.01, gamma = 0.5):
        self.params = Params(w = w, s = s, lam = lam, gamma = gamma)
        
    def change_params(self, w = None, s = None, lam = None, gamma = None):
        """
        Changes parameters of the GTS measure
        
        Parameters
        ----------
        w : double
            the weight of the timing uncertainty errors
        s : double
            timing uncertainty neighbourhood radius
        lam: double
            the weight of the total variation penalty
        gamma: double
            the lower bound on the duration of the states
        """
        params.change_params(w, s, lam, gamma)
        
    
    def performance(self, ground_truth, estimated_labels):
        """
        Calculates GTS performance measure of the estimated_labels given ground_truth labels
        
        Parameters
        ----------
        ground_truth : BinaryStateSequence
            the true labels represented using BinaryStateSequence class
        estimated_labels : BinaryStateSequence
            the estimated labels represented using BinaryStateSequence class
            
        Returns
        -------
        The GTS measure between ground_truth and estimated_labels
        """
        # a copy is needed so that the jumps array in BSS instances is not altered
        jumps_gt = ground_truth.jumps.copy()
        jumps_est = estimated_labels.jumps.copy()
        
        # calculation of the duration penalty term; the length of the arrays in identify_jump_subseq refers to the number of jumps that are closer than gamma to each other, so that number minus one gives a number of subsequent states that are too short
        penalty = 0
        for array in identify_jump_subseq(jumps_est, self.params.gamma):
            penalty += self.params.lam * (len(array) - 1)
        # it can occur that there are no jumps in this case there will be just one empty array and penalty would be negative then, this prevents it from happening
        if penalty < 0:
            penalty = 0
        
        # GTS extension of the BSS instances (this one is assymetric)
        if ground_truth.first_value != estimated_labels.first_value:
            jumps_est.insert(0, 0)
        if len(jumps_gt) % 2 != len(jumps_est) % 2:
            jumps_est = jumps_est + [estimated_labels.time_horizon]

        # finding the global time shift
        epsilons = mirrored(self.params.s, 0.001)
        shifted_l1 = [integral(jumps_gt, [jump - epsilon for jump in jumps_est]) + self.params.w * abs(epsilon) 
                   for epsilon in epsilons]

        return np.exp(-(min(shifted_l1) / estimated_labels.time_horizon + penalty))
        
    
    def error_characterization(self, ground_truth, estimated_labels):
        """
        Returns dictionary of errors as defined in GTS performance measure
        
        Parameters
        ----------
        ground_truth : BinaryStateSequence
            the true labels represented using BinaryStateSequence class
        estimated_labels : BinaryStateSequence
            the estimated labels represented using BinaryStateSequence class
            
        Returns
        -------
        A dictionary of 3 items: the error in GTS measure pertaining to L_1 penalty, to timing uncertainty and to duration penalty term
        """
        # a copy is needed so that the jumps array in BSS instances is not altered
        jumps_gt = ground_truth.jumps.copy()
        jumps_est = estimated_labels.jumps.copy()
        
        # calculation of the duration penalty term; the length of the arrays in identify_jump_subseq refers to the number of jumps that are closer than gamma to each other, so that number minus one gives a number of subsequent states that are too short
        penalty = 0
        for array in identify_jump_subseq(jumps_est, self.params.gamma):
            penalty += self.params.lam * (len(array) - 1)
        # it can occur that there are no jumps in this case there will be just one empty array and penalty would be negative then, this prevents it from happening
        if penalty < 0:
            penalty = 0
        
        # GTS extension of the BSS instances (this one is assymetric)
        if ground_truth.first_value != estimated_labels.first_value:
            jumps_est.insert(0, 0)
        if len(jumps_gt) % 2 != len(jumps_est) % 2:
            jumps_est = jumps_est + [estimated_labels.time_horizon]

        # finding the global time shift
        epsilons = mirrored(self.params.s, 0.001)
        shifted_l1 = [integral(jumps_gt, [jump - epsilon for jump in jumps_est]) + self.params.w * abs(epsilon) 
                   for epsilon in epsilons]
        eps = epsilons[np.argmin(shifted_l1)]
        l1_pen = integral(jumps_gt, [jump - eps for jump in jumps_est])

        return {'$L_1$ penalty': l1_pen / estimated_labels.time_horizon,
                'timing_uncertainty': self.params.w * abs(eps) / estimated_labels.time_horizon,
                'duration penalty term': penalty}

class LTS_measure:
    """
    A class used to represent LTS measure, a performance measure for binary classification of time series

    Attributes
    ----------
    params: Params
        parameters of the LTS measure

    Methods
    -------
    change_params(w, s, lam)
        changes parameters of the LTS measure
    performance(ground_truth, estimated_labels)
        calculates LTS performance measure of the estimated_labels given ground_truth labels
    error_characterization(ground_truth, estimated_labels)
        returns dictionary of errors as defined in LTS performance measure
    """
    def __init__(self, w = 0.6, s = 0.35, lam = 0.01, gamma = 0.5):
        self.params = Params(w = w, s = s, lam = lam, gamma = gamma)
        
    def change_params(self, w = None, s = None, lam = None, gamma = None):
        """
        Changes parameters of the LTS measure
        
        Parameters
        ----------
        w : double
            the weight of the timing uncertainty errors
        s : double
            timing uncertainty neighbourhood radius
        lam: double
            the weight of the total variation penalty
        gamma: double
            the lower bound on the duration of the states
        """
        params.change_params(w, s, lam, gamma)
        
    
    def performance(self, ground_truth, estimated_labels):
        """
        Calculates LTS performance measure of the estimated_labels given ground_truth labels
        
        Parameters
        ----------
        ground_truth : BinaryStateSequence
            the true labels represented using BinaryStateSequence class
        estimated_labels : BinaryStateSequence
            the estimated labels represented using BinaryStateSequence class
            
        Returns
        -------
        The LTS measure between ground_truth and estimated_labels
        """
        # a copy is needed so that the jumps array in BSS instances is not altered
        jumps_gt = ground_truth.jumps.copy()
        jumps_est = estimated_labels.jumps.copy()
        
        # calculation of the duration penalty term; the length of the arrays in identify_jump_subseq refers to the number of jumps that are closer than gamma to each other, so that number minus one gives a number of subsequent states that are too short
        penalty = 0
        for array in identify_jump_subseq(jumps_est, self.params.gamma):
            penalty += self.params.lam * (len(array) - 1)
        # it can occur that there are no jumps in this case there will be just one empty array and penalty would be negative then, this prevents it from happening
        if penalty < 0:
            penalty = 0
        
        # LTS extension of the ground truth
        if ground_truth.first_value != 0:
            jumps_gt.insert(0, 0)
        if len(jumps_gt) % 2:
            jumps_gt = jumps_gt + [ground_truth.time_horizon]
        
        # LTS extension of the estimated labels
        if estimated_labels.first_value != 0:
            jumps_est.insert(0, 0)
        if len(jumps_est) % 2:
            jumps_est = jumps_est + [estimated_labels.time_horizon]
        
        # preparation phase; we need a joint jump array and an array of 0's and 1's that connects the joint_jumps array with the jumps_gt and jumps_est
        joint_jumps = np.concatenate([jumps_est,jumps_gt])
        memberships = np.concatenate([np.zeros_like(jumps_est), np.ones_like(jumps_gt)])
        jumps = np.sort(joint_jumps)
        jumps_idx = np.argsort(joint_jumps)
        memberships = memberships[jumps_idx]

        # calculating the local time shifts for each segment
        dist = 0
        i = 0
        while i < len(jumps):
            if (jumps[i + 1] - jumps[i] <= self.params.s) & (memberships[i] != memberships[i + 1]):
                dist += self.params.w * (jumps[i + 1] - jumps[i])
            else:
                dist += jumps[i + 1] - jumps[i]
            i += 2

        return np.exp(-(dist / estimated_labels.time_horizon + penalty))
        
    
    def error_characterization(self, ground_truth, estimated_labels):
        """
        Returns dictionary of errors as defined in LTS performance measure
        
        Parameters
        ----------
        ground_truth : BinaryStateSequence
            the true labels represented using BinaryStateSequence class
        estimated_labels : BinaryStateSequence
            the estimated labels represented using BinaryStateSequence class
            
        Returns
        -------
        A dictionary of 3 items: the error in GTS measure pertaining to L_1 penalty, to timing uncertainty and to duration penalty term
        """
        # a copy is needed so that the jumps array in BSS instances is not altered
        jumps_gt = ground_truth.jumps.copy()
        jumps_est = estimated_labels.jumps.copy()
        
        # calculation of the duration penalty term; the length of the arrays in identify_jump_subseq refers to the number of jumps that are closer than gamma to each other, so that number minus one gives a number of subsequent states that are too short
        penalty = 0
        for array in identify_jump_subseq(jumps_est, self.params.gamma):
            penalty += self.params.lam * (len(array) - 1)
        # it can occur that there are no jumps in this case there will be just one empty array and penalty would be negative then, this prevents it from happening
        if penalty < 0:
            penalty = 0
        
        # LTS extension of the ground truth
        if ground_truth.first_value != 0:
            jumps_gt.insert(0, 0)
        if len(jumps_gt) % 2:
            jumps_gt = jumps_gt + [ground_truth.time_horizon]
        
        # LTS extension of the estimated labels
        if estimated_labels.first_value != 0:
            jumps_est.insert(0, 0)
        if len(jumps_est) % 2:
            jumps_est = jumps_est + [estimated_labels.time_horizon]
        
        # preparation phase; we need a joint jump array and an array of 0's and 1's that connects the joint_jumps array with the jumps_gt and jumps_est
        joint_jumps = np.concatenate([jumps_est, jumps_gt])
        memberships = np.concatenate([np.zeros_like(jumps_est), np.ones_like(jumps_gt)])
        jumps = np.sort(joint_jumps)
        jumps_idx = np.argsort(joint_jumps)
        memberships = memberships[jumps_idx]

        l1_pen = 0
        timing_uncertainty = 0
        
        # calculating the local time shifts for each segment
        i = 0
        while i < len(jumps):
            if (jumps[i + 1] - jumps[i] <= self.params.s) & (memberships[i] != memberships[i + 1]):
                timing_uncertainty += self.params.w * (jumps[i + 1] - jumps[i])
            else:
                l1_pen += jumps[i + 1] - jumps[i]
            i += 2

        return {'$L_1$ penalty': l1_pen / estimated_labels.time_horizon,
                'timing_uncertainty': timing_uncertainty / estimated_labels.time_horizon,
                'duration penalty term': penalty}