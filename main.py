from process_data import create_df
from k_mean import k_mean, compute_average_distance, find_gap_cost_function
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_distances
from neural_network import predict_neural_network
import json
import numpy as np
from visualisation import visualize
from pylda_visualisation import pylda_visualize

PCA_val = True
MDS_val = False
N = 20
num_first_tests = 10 # Run intitailization num_first_tests times and choose best initialization
number_clusters = 5
percent_min_elements = 0.1

def cluster_articles(supervised=False, target_dimension=10, sentence=False, path_result="result.csv"):
    """ Cluster the articles with a k_mean algorithms, writes the results
    in a csv file and return a list of cluster objects"""

    chemin = "preprocessed_df" + sentence * "_sentence" + ".csv"

    try:
        # Load a preprocessed df
        df = pd.read_csv(chemin, index_col=0)
    
    except IOError:
        # Build the preprecessed df if it is missing
        print("Missing preprocessed file")
        df = create_df("df_brown.csv", sentence=sentence)
        df.to_csv(chemin)
    
    if supervised:
        real_clusters = df["real_cluster"]
        df = df.drop("real_cluster", axis=1)

    
    df = df.drop("text", axis=1)

    if PCA_val:
        # Reduce the dimension with a PCA 
        pca = PCA(n_components=target_dimension, svd_solver='full')

        print("Fit and Transform...")
        pca.fit(df) 
        print("Transform...")
        df_reduced = pd.DataFrame(pca.transform(df))
    

    if MDS_val:
        # Reduce the dimension with a MDS
        mds = manifold.MDS(target_dimension, max_iter=100, n_init=1)
        df_reduced = pd.DataFrame(mds.fit_transform(df))
  
    
    # Del the initial df
    del df
 
    # The score at each iteration of the algorithm
    scores = []
    model = None
    
    centroids = None
    
    for i in range(N):
        if centroids:
            # Centroids already intialized
            clusters, cost, centroids = k_mean(df_reduced, number_clusters, centroids=centroids)
            labels = [value for (key, value) in sorted(clusters.items())]
            score = silhouette_score(df_reduced, labels, metric='cosine')

        if not centroids:
            best_score = 0 
            # First iterations, we run the algorithm mutliple times and look for the best centroids
            for _ in range(num_first_tests):
                clusters, cost, centroids = k_mean(df_reduced, number_clusters)
                labels = [value for (key, value) in sorted(clusters.items())]
                score = silhouette_score(df_reduced, labels, metric='cosine')
                if score > best_score:
                    # Best iteratoin, we save the configuration
                    best_centroids = centroids
                    best_score = score
                    best_labels = labels
                    best_clusters = clusters
            centroids = best_centroids
            score = best_score
            labels = best_labels
            clusters = best_clusters

        scores += [score]

        # Pairwise distances between every points
        distances = pairwise_distances(df_reduced, metric='cosine')
        # Silhouette score for each point
        sil_samples = silhouette_samples(distances, labels , metric='cosine')
        
        # Find the the misclassified elements
        min_elements = choose_min_elements(sil_samples)
    
        # Load the inital dataset and split in training and testing set
        df_brown = pd.DataFrame.from_csv("df_brown.csv")
        df_test_brown = df_brown.ix[min_elements]
        df_train_brown = df_brown.drop(df_brown.index[min_elements])
        
        
        labels_as_dict = {}
        for a in range(len(labels)):
            if a not in min_elements:
                labels_as_dict[a] = labels[a]
        
        train_Y = pd.DataFrame(labels_as_dict.values())
        
        labels = pd.get_dummies(train_Y[0])
        # number of clusters in the df
        s = labels.shape[1]
    
        # Correction when a clusters is deleted by the algorithm
        if s < number_clusters:
            headers = list(labels)
            size_train = df_train_brown.shape[0]
            for i in range(number_clusters):
                if i not in headers:
                    labels[i] = np.array([0]*size_train)
  
        train_Y.to_csv("train_y_nn.csv")
        df_train_brown.to_csv("train_nn.csv")
        df_test_brown.to_csv("test_nn.csv")

        # Neural network returns prediction, new vector and model    
        predictions, new_vectors, model = predict_neural_network(df_train_brown, labels, df_test_brown, target_dimension, number_clusters, model)  
        
        # Update the vectors in the dataframe
        for j in range(len(min_elements)):
            index = min_elements[j]
            vector = new_vectors[j]
            for k in range(len(vector)):
                df_reduced[k][index] = vector[k]

        
    # Write the results in a csv file
    predicted_clusters = [value for (key, value) in sorted(clusters.items())]

    df_brown["pred_cluster"] = predicted_clusters
    df_brown.to_csv(path_result)

    # Uncomment to plot the silhouette according to the iterations
    """
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette")
    plt.title("Evolution of the silhouette after " + str(N) + " iterations")
    plt.plot(range(N), scores)
    plt.show()
    """

    return clusters, df_reduced




def choose_min_elements(scores):
    num_elements = int(len(scores) * percent_min_elements)
    return scores.argsort()[:num_elements]


    
if __name__ == '__main__':
    path_result = "clusters.csv"
    clusters, df = cluster_articles(sentence=False, path_result=path_result)
    predicted_clusters = [value for (key, value) in sorted(clusters.items())]
    #visualize(df, predicted_clusters, number_clusters, N)
    path_visualize = "test_visualize"
    pylda_visualize(path_result, path_visualize, num_topic=3, filter_by_cluster=None)

    






