import numpy as np
import datetime

from sklearn.cluster import KMeans, Birch, MiniBatchKMeans, AffinityPropagation, AgglomerativeClustering, DBSCAN
from sklearn.cluster import MeanShift, OPTICS, kmeans_plusplus

# Use pickle file not df for save the results.

def algorithm_params():
    alg_ref_list = [KMeans, Birch, MiniBatchKMeans, MeanShift]#, OPTICS, ] # Removed DBSCAN
    alg_name_list = ['kmeans', 'birch', 'mini_batch_kmeans', 'mean_shift']#, 'OPTICS', ] # 'DBSCAN'

    kmeans_params = [{'n_clusters': i, 'random_state': 42} for i in range(2, 11)]
    birch_params = [{'n_clusters': i, 'threshold': j} for i in range(2, 11) for j in np.arange(0.1, 1, 0.1)]
    mini_batch_params = [{'n_clusters': i, 'random_state': 42} for i in range(2, 11)]


    mean_shift_params = [{'bin_seeding': bin_seeding, 'cluster_all': cluster_all} 
                         for bin_seeding in [True, False]
                         for cluster_all in [True, False]]
    

    dict_of_list_of_dict_params = {KMeans: kmeans_params, Birch: birch_params, MiniBatchKMeans: mini_batch_params,
                                   MeanShift: mean_shift_params
                                   }
    total_params = len(kmeans_params) * len(birch_params) * len(mini_batch_params)  * len(mean_shift_params)
    return alg_ref_list, alg_name_list, dict_of_list_of_dict_params, total_params


# iterate through splitting_data_params 
def splitting_data_params(list_of_possible_time_steps):
    """Returns 
    {
        50: [{'date': .., 'overlap': ..}, {'date': .., 'overlap': ..}, {'date': .., 'overlap': ..}],
        40: [{'date': .., 'overlap': ..}, {'date': .., 'overlap': ..}, {'date': .., 'overlap': ..}]
    } """
    date_from_list = [np.datetime64(datetime.date(2018, 1, 2))]
    date_to_list = [np.datetime64(datetime.date(2021, 4, 26))]
    res = {}
    total_params = 0
    for time_step in list_of_possible_time_steps:
        sub_params = [{'date_from': d1, 'date_to': d2, 'overlap': overlap} for overlap in range(0, time_step-10, 10) for d1, d2 in zip(date_from_list, date_to_list)] 
        res[time_step] = sub_params
        total_params += len(sub_params)
    return res, total_params


def splitting_data_params_advanced(list_of_possible_time_steps):
    date_from_list = [np.datetime64(datetime.date(2018, 1, 2))]
    date_to_list = [np.datetime64(datetime.date(2021, 4, 26))]
    res = {}
    total_params = 0
    for time_step in list_of_possible_time_steps:
        sub_params = [{'date_from': d1, 'date_to': d2, 'overlap': 0} for d1, d2 in zip(date_from_list, date_to_list)] 
        res[time_step] = sub_params
        total_params += len(sub_params)
    return res, total_params


def override_knn():
    None
    #connect kmeans_plusplus with k means


def more_complex_algorithm_params():
    alg_ref_list = [OPTICS, DBSCAN, AffinityPropagation, AgglomerativeClustering]
    alg_name_list = ['optics', 'dbscan', 'affinity_prop', 'agglomerative_clustering']
    possible_distance_params = {'kd_tree': ['euclidean', 'l2', 'minkowski', 'manhattan',
                                            'chebyshev'], 
                                'ball_tree': np.array(['euclidean', 'l2', 'minkowski', 'manhattan',
                                                       'l1', 'chebyshev',
                                                       'wminkowski', 'canberra',
                                                       'braycurtis',
                                                       ], dtype='<U14'),
                                'brute': np.array(['cityblock', 'euclidean', 'l2', 'manhattan',
                                                'braycurtis', 'canberra', 'chebyshev',
                                                   'correlation', 'cosine',
                                                    'minkowski', 'sqeuclidean', 'wminkowski'], dtype='<U14')
                                }

    optics_params = [{'min_samples': min_samples, 'metric': metrics, 'algorithm': algorithm, 'leaf_size': 30} 
                      for min_samples in [3, 5, 10]
                      for algorithm in possible_distance_params.keys()
                      for metrics in possible_distance_params[algorithm]]
    

    ignore_li = [{'n_clusters': 5, 'random_state': 42, 'eigen_solver': 'arpack', 'affinity': 'rbf', 'assign_labels': 'kmeans'},
              {'n_clusters': 5, 'random_state': 42, 'eigen_solver': 'arpack', 'affinity': 'rbf', 'assign_labels': 'discretize'}]
    
    is_in_ignore_list = lambda i, ei, aff, ass: np.array([True if i == ignore['n_clusters'] and ei == ignore['eigen_solver'] and aff == ignore['affinity'] and ass == ignore['assign_labels'] else False for ignore in ignore_li]).any()

    
    dbscan_params = [{'eps': eps, 'min_samples': min_samples, 'metric': metrics, 'algorithm': algorithm, 'n_jobs': 1} 
                     for eps in [0.3, 0.5, 0.8]
                      for min_samples in [3, 5, 10] 
                      for algorithm in possible_distance_params.keys()
                      for metrics in possible_distance_params[algorithm]]
    
    affinity_prop_params = [{'random_state': 42, 'damping': damp} 
                            for damp in np.arange(0.5, 1, 0.05)]
    
    ignore_linkage_list = {'ward': ['l1', 'l2', 'manhattan', 'cosine']}
    ignore_linkage_fun = lambda lin, aff: np.any([True if aff in ignore_linkage_list[key] else False for key in ignore_linkage_list.keys() if lin == key ])
    agglomerative_clustering_params = [{'n_clusters': i, 'affinity': affinity, 'linkage': linkage}
                                  for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]
                                  for linkage in ['ward', 'average', 'complete', 'single']
                                  for affinity in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'] if not ignore_linkage_fun(linkage, affinity)
                                  ]
    

    dict_of_list_of_dict_params = {OPTICS: optics_params, DBSCAN: dbscan_params, AffinityPropagation: affinity_prop_params, AgglomerativeClustering: agglomerative_clustering_params}
    
    return alg_ref_list, alg_name_list, dict_of_list_of_dict_params, [100]
    
    
    # Affinity Propagation because it's slow
    
    # birch can be provided as an input to another clustering algorithm as AgglomerativeClustering
    
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


def outlier_algorithm_params():
    possible_distance_params = {'kd_tree': ['euclidean', 'l2', 'minkowski', 'manhattan',
                                            'chebyshev'], 
                                'ball_tree': np.array(['euclidean', 'l2', 'minkowski', 'manhattan',
                                                       'l1', 'chebyshev',
                                                       'wminkowski', 'canberra',
                                                       'braycurtis',
                                                       ], dtype='<U14'),
                                'brute': np.array(['cityblock', 'euclidean', 'l2', 'manhattan',
                                                'braycurtis', 'canberra', 'chebyshev',
                                                   'correlation', 'cosine',
                                                    'minkowski', 'sqeuclidean', 'wminkowski'], dtype='<U14')
                                }


    alg_ref_list = [IsolationForest, OneClassSVM, LocalOutlierFactor]
    alg_name_list = ['iforest', 'one_class_svm']
    
    iforest_params = [{'n_estimators': n_est, 'max_features': max_f, 'contamination': contam}
                      for n_est in range(50, 251, 50)
                      for max_f in range(1, 15, 1)
                      for contam in np.arange(0, 0.55, 0.1)]

    one_class_svm_params = [{'kernel': kern, 'gamma': gamm}
                            for kern in ['linear', 'poly', 'rbf', 'sigmoid']
                            for gamm in ['scale', 'auto']
                            for algh in possible_distance_params.keys()
                            for dist in possible_distance_params[algh]
                            for cont in np.arange(0, 0.51, 0.05)]
    
    ignore_linkage_list = {'ball_tree': ['wminkowski', 'haversine', 'pyfunc'], 
                           'brute': ['wminkowski', 'haversine', 'pyfunc']}
    ignore_linkage_fun = lambda lin, aff: np.any([True if aff in ignore_linkage_list[key] else False for key in ignore_linkage_list.keys() if lin == key ])
    
    
    local_outlier_factor_params = [{'n_neighbors': n_neighb, 'algorithm': algh, 'metric': dist, 'contamination': cont, 'leaf_size': 30}
                                   for n_neighb in range(2, 20, 1)
                                   for algh in possible_distance_params.keys()
                                    for dist in possible_distance_params[algh] if not ignore_linkage_fun(algh, dist)
                                    for cont in np.arange(0.001, 0.5, 0.05)]
    
    

    dict_of_list_of_dict_params = {IsolationForest: iforest_params, OneClassSVM: one_class_svm_params, LocalOutlierFactor: local_outlier_factor_params
                                   }
    return alg_ref_list, alg_name_list, dict_of_list_of_dict_params, [100]

    
    
    # EXAMPLE:
    #bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #ms.fit(X)
    #labels = ms.labels_
    #cluster_centers = ms.cluster_centers_








def my_fun(Li, k, ov):
    splitted_li = []
    num_obs = (len(Li) - k) // (k - ov)
    for i in range(num_obs):
        min_ind = i * k  if i == 0 else i * k - ov * i
        max_ind = (i + 1) * k if i == 0 else (i + 1) * k - ov * i
        splitted_li.append(Li[min_ind:max_ind])
    return splitted_li