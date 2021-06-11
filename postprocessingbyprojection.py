import numpy as np
from itertools import compress

class DAG(object):
    """
    A class used to represent a directed acyclic graph for post-processing by projection

    Attributes
    ----------
    gamma : float
        lower bound on the lengths of the intervals (also double the weight of the jump)
    intervals : array of doubles
        an array of distances between subsequent jumps
    vertices : array of ints and infinities
        an array of all vertices/jumps including the two fictitious ones: -infinity and infinity
    number_of_vertices : int
        length of vertices array
    usable_vertices : array of ints
        all vertices excluding 2 and the one before last actual jump (they cannot be used in projection)
    weightMatrix : numpy 2-dimensional array of doubles
        a matrix of weights of arcs between all connected vertices

    Methods
    -------
    change_weights()
        learns the weights of the graph given the jumps given
    changeBestDist(dist, prec, nodeStart, nodeEnd)
        changes the current best distance from -np.inf to nodeEnd by moving the predecessor of nodeEnd to nodeStart
    shortestPath()
        finds the shortest path from -np.inf to np.inf in the graph
    """
    def __init__(self, jumps, gamma):
        """
        Parameters
        ----------
        jumps : array of doubles
            An array of jumps of the projected function
        gamma : str
            The lower bound on the lengths of the intervals (also double the weight of the jump)
        """
        # initializing gamma
        self.gamma = gamma
        
        # calculation of the distance between subsequent jumps
        self.intervals = [j-i for i, j in zip(jumps[:-1], jumps[1:])]
        
        # initializing the vertex set of the graph; the added points in infinity make the first and last state infinitely long
        self.vertices = [-np.inf] + jumps + [np.inf]
        
        # this value is going to be frequently used
        self.number_of_vertices = len(self.vertices)
        
        # second jump as well second to last jump are never present in the projection
        self.usable_vertices = [i for i in range(len(self.vertices)) if (i != 2) & (i != self.number_of_vertices - 3)]
        
        # weight matrix for initialized to have inifinite distance between all vertices; the change_weights function is used in order to find correct weights
        self.weightMatrix = np.full((len(self.vertices), len(self.vertices)), np.inf)
        self.change_weights()
    
    def change_weights(self):
        """
        Learns the weights of the arcs of the graph
        """
        # we start at the source (the vertex j_o: -np.inf) and find the weights of the arcs coming out from it; first jump can only occur at an odd-numbered jump
        self.weightMatrix[0][1::2] = [self.gamma / 2 * int(self.vertices[k] < np.inf) + sum(self.intervals[0:max(k-1, 0):2])
                                      if k != self.number_of_vertices - 3 else np.inf
                                      for k in range(len(self.vertices))[1::2]]
        
        # for each vertex of the graph we find the new weights
        for k in self.usable_vertices[1:-2]:
            # self.intervals have different indices than the weight matrix; to find the interval in which value changes if jump at k+3 follows the one at k we take the interval following the (k-1)th interval (this is the interval whose start is kth jump), which means (k-1)+1=kth interval
            self.weightMatrix[k][k+3::2] = np.cumsum([self.gamma / 2 * int(self.vertices[k+3] < np.inf) + self.intervals[k]] +
                                                     [self.intervals[l - 3] - self.gamma / 2 * int(self.vertices[l] == np.inf)
                                                      for l in range(len(self.vertices))[k + 5::2]])
            
            # there can never be a jump at second to last jump
            self.weightMatrix[k][self.number_of_vertices - 3] = np.inf
            l = k+3
            
            # we ensure that the distance between kth and lth vertex is smaller than gamma (only than can the weight be finite)
            # if l >= self.number_of_vertices-1 then the condition in the while will not be satisfied so that's the natural end of the loop
            while (self.vertices[min(l, self.number_of_vertices - 1)] - self.vertices[k] < self.gamma):
                self.weightMatrix[k][l] = np.inf
                l += 2
        
        if self.number_of_vertices > 4:
            self.weightMatrix[self.number_of_vertices - 2, self.number_of_vertices - 1] = 0
            
    def changeBestDist(self, dist, prec, nodeStart, nodeEnd):
        """
        Changes the current best distance from -np.inf to nodeEnd by moving the predecessor of nodeEnd to nodeStart
        
        Parameters
        ----------
        dist : array of doubles
            array of current best distances from -np.inf to all vertices
        prec : array of doubles
            array of current best predecessors of all vertices on the path from -np.inf
        nodeStart : double
            new best predecessor of nodeEnd on the path from -np.inf
        nodeEnd : double
            vertex whose path is changed
        """
        # the best distance to nodeEnd is updated by taking the best distance to nodeStart and then adding the weight of the arc from nodeStart to nodeEnd
        dist[nodeEnd] = dist[nodeStart] + self.weightMatrix[nodeStart, nodeEnd]
        prec[nodeEnd] = nodeStart
    
    def shortestPath(self):
        """
        Finds the shortest path from -np.inf to np.inf in the graph

        Returns
        -------
        a tuple: first position is the distance between projected function and the projection; second position are the jumps of the projection.
        """
        # dist is the array of current best distances from -np.inf to all other vertices
        dist = dict.fromkeys(self.usable_vertices, np.inf)
        
        # prec is the array of current predecessors of all vertices on the best path from -np.inf
        prec = dict.fromkeys(self.usable_vertices)
        dist[0] = 0
    
        # updating of dist and prec using changeBestDist function (filter allows to update only when the update is for the better)
        list(map(lambda nodeStart: list(map(lambda nodeEnd: self.changeBestDist(dist, prec, nodeStart, nodeEnd),
                                            filter(lambda nodeEnd: dist[nodeEnd] > dist[nodeStart] + self.weightMatrix[nodeStart][nodeEnd], 
                                                   list(compress(range(len(self.vertices)), self.weightMatrix[nodeStart] != np.inf))))), self.usable_vertices))
        
        # best_path contains all vertices on the best path from -np.inf to np.inf; we start from the end
        best_path = [np.inf]
        
        # the index of the last jump in self.vertices is self.number_of_vertices - 1
        node_idx = self.number_of_vertices - 1
        
        # while its possible add all vertices on the best path based on the prec array going from the end to the source.
        while node_idx > 0:
            best_path.append(self.vertices[prec[node_idx]])
            node_idx = prec[node_idx]
        
        return (dist[self.number_of_vertices - 1], best_path[::-1])