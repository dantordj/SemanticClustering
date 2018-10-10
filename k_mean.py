import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sem_distance import sem_distance
from scipy.spatial import distance as dist_pack


def k_mean(data, num_clusters, texts=None, centroids=None):
    """creates the points for the k_mean algorithms and run it"""

    num_points = len(data)

    dimensions = len(list(data))
    print("num_points", num_points)
    print("dimension", dimensions)
    values = data.values.tolist()
    ids = data.index.values.tolist()

    points = [Point(values[i], ids[i]) for i in range(num_points)]

    # When do we say the optimization has 'converged' and stop updating clusters
    cutoff = 0.0000001

    # Clustering the points
    clusters = kmeans_algorithm(points, num_clusters, cutoff, centroids)
    centroids = [cluster.centroid for cluster in clusters]
    cost = cost_function(clusters)
    print("Cost", cost)
    belongs_clusers = {}
    index_cluster = 0
    for cluster in clusters:
        for p in cluster.points:
            belongs_clusers[p.id] = index_cluster
        index_cluster += 1
    return (belongs_clusers, cost, centroids)
  


class Point(object):
    """
    A point in n dimensional space
    """
    def __init__(self, coords, id=0, text=""):
        """
        coords: a list of values, one per dimension
        """
        self.id = id
        self.coords = coords
        self.n = len(coords)
        self.text = text

    def __repr__(self):
        return str(self.coords)



class Cluster(object):
    """
    A set of points and their centroid
    """

    def __init__(self, points):
        """
        points - A list of point objects
        """

        if len(points) == 0:
            raise Exception("ERROR: empty cluster")

        # Points that belong to this cluster
        self.points = points

        # Dimensionality of the points 
        self.n = points[0].n

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculate_centroid()

    def __repr__(self):
        """
        String representation of this object
        """
        return str(self.points)

    def update(self, points):
        """ update the centroids"""
        old_centroid = self.centroid
        self.points = points
        if (len(self.points) == 0):
            return 0
        self.centroid = self.calculate_centroid()

        shift = get_distance(old_centroid, self.centroid)
        return shift

    def calculate_centroid(self):
        """
        Finds the centroid a group of n-dimensional points
        """
        num_points = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/num_points for dList in unzipped]

        # Get closest sentence

        centroid = Point(centroid_coords)

        # Uncomment when weighted distance used 
        """
        if not self.points:
            return 
        idx_point = np.argmin([get_cosine_distance(centroid, point) for point in self.points])


        centroid.text = self.points[idx_point].text
        """

        return centroid

def kmeans_algorithm(points, k, cutoff, centroids=None):
    """k_means algorithm, return a list of clusters"""
    # k random points to use as our initial centroids
    initial = centroids
    if not initial:
        initial = random.sample(points, k)


    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loop_counter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [[] for _ in clusters]
        clusterCount = len(clusters)

        loop_counter += 1

        for p in points:
            # Distance between that point and the centroid of the first
            # cluster.
            smallest_distance = get_distance(p, clusters[0].centroid)

            clusterIndex = 0

            for i in range(clusterCount - 1):
                # distance of that point to each other cluster's centroid
                distance = get_distance(p, clusters[i+1].centroid)
            
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i + 1
            lists[clusterIndex].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # For each cluster ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])

            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print("Converged after %s iterations" % loop_counter)
            break

    return clusters

def cost_function(clusters):
    """Cost function of the k_mean algorithm"""
    cost = 0
    for cluster in clusters:
        points = cluster.points
        c = cluster.centroid

        for p in points:
            cost += get_distance(p,c)**2
    return cost

def get_distance(a, b):
    """To be modified when we will want to custumize the distance function"""
    # Uncomment to use weighted distance between sem matching and cosine distance

    """
    w0 = 0.5
    w1 = 0.5

    d = w0*get_euclidean_distance(a,b) + w1*sem_distance(a.text, b.text)
    print(a.text)
    return w0*get_euclidean_distance(a,b) + w1*sem_distance(a.text, b.text)
    """
    return get_cosine_distance(a,b)

 
def get_cosine_distance(a, b):
    """
    Cosine distance between two n-dimensional points.
    """
    
    distance = dist_pack.cosine(a.coords, b.coords)

    return distance

def get_euclidean_distance(a, b):
    """
    Euclidean distance between two n-dimensional points.
    """
    accumulated_difference = 0.0


    for i in range(a.n):
        square_difference = pow((a.coords[i] - b.coords[i]), 2)
        accumulated_difference += square_difference
    distance = math.sqrt(accumulated_difference)


    return distance


def find_gap_cost_function(data):
    """Plot the cost according to the number of clusters"""
    costs = []
    max_num_clusters = 70
    X = []
    for i in range(1, max_num_clusters, 3):
        cost = k_mean(data, i)[1]
        costs += [cost]
        X += [i]
    plt.plot(X,costs)
    plt.show()


def compute_average_distance(data):
    values = data.values.tolist()
    points = [Point(value) for value in values]
    num_tests = 20000
    dist = 0
    for i in range(num_tests):
        a, b = np.random.randint(len(data), size=2)
        dist += get_distance(points[a], points[b])**2
    return dist / num_tests 
 



