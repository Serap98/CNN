import numpy as np
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras as keras
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans, AffinityPropagation, AgglomerativeClustering, DBSCAN
from sklearn.cluster import MeanShift, OPTICS, SpectralClustering#, GaussianMixture
import multiprocessing
import params
from wrapt_timeout_decorator import timeout
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition



np.random.seed(42)
"""
import more_itertools
list(more_itertools.windowed([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],n=3, step=3))
"""


def read_data(path_to_file):
    """Returns dic with encoded pension fund names as keys and values as sorted dates"""
    read_time = '%m/%d/%Y'
    time_series_data = pd.read_csv(path_to_file, encoding='cp775')
    time_series_data['Calculation date'] = [datetime.datetime.strptime(date, read_time) for date in time_series_data['Calculation date'].values]
    unique_names = time_series_data['IP name'].value_counts()
    # creating name mappings
    name_mapping_to = {list(unique_names.keys())[i]: i for i in range(len(unique_names.keys()))}
    name_mapping_from = {i: list(unique_names.keys())[i] for i in range(len(unique_names.keys()))}
    time_series_data['name'] = time_series_data['IP name'].map(name_mapping_to)
    return time_series_data, name_mapping_to, name_mapping_from


def pension_funds_between_interval_dates(pension_data, name_mapping_from, min_filter_date=None, max_filter_date=None):
    """Returns keys of pension funds which have started prior min date and continued till max date from name mapping from dict"""

    date_column = 'Calculation date'
    suitable_keys = []
    for key in name_mapping_from.keys():
        subgroup = time_series_data[time_series_data['name'] == key]
        min_date = min(subgroup[date_column].values)
        max_date = max(subgroup[date_column].values)
        started_prior = min_date <= min_filter_date if min_filter_date else True
        continued_after = max_date >= max_filter_date if max_filter_date else True
        if started_prior and continued_after:
            suitable_keys.append(key)
    return suitable_keys


def data_processing(dataset, name_mapping_from, time_steps, date_from=None, date_to=None, overlap=None):
    suitable_keys = pension_funds_between_interval_dates(dataset, name_mapping_from, min_filter_date=date_from, max_filter_date=date_to)
    data_list = []
    info = {}
    date_column = 'Calculation date'
    j = -1
    benchmark_num_obs = 0
    for key in suitable_keys:
        filtered_data = dataset[dataset['name'] == key].sort_values(by=date_column, ascending=True)
        filtered_data = filtered_data[(filtered_data[date_column] >= date_from) & (filtered_data[date_column] <= date_to)]
        num_obs = (len(filtered_data) - time_steps) // (time_steps - overlap) if overlap else len(filtered_data) // time_steps
        if not benchmark_num_obs:
            benchmark_num_obs = num_obs

        sub_list = []
        if benchmark_num_obs != num_obs:
            continue

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.array(filtered_data['NAV value EUR'].values).reshape(-1, 1))
        for i in range(num_obs):
            min_ind = i * time_steps  if i == 0 else i * time_steps - overlap * i
            max_ind = (i + 1) * time_steps if i == 0 else (i + 1) * time_steps - overlap * i
            sub_list.append(scaled_data[min_ind : max_ind])
        j += 1
        max_date = filtered_data[date_column].values[max_ind]
        info[key] = {'index': j, 'num_obs': num_obs, 'max_date': max_date}
        data_list.append(np.array(sub_list))
    return np.array(data_list), info


def load_h5_model(model_name, add_extension=False):
    adj_model_name = f'{model_name}.h5' if add_extension else model_name
    reconstructed_model = tf.keras.models.load_model(adj_model_name,
                                                     custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
    return reconstructed_model


def remove_last_layers(model, remove_n_layers, input_shape=None):
    new_model = tf.keras.models.Sequential()
    if input_shape:
        new_model.add(keras.Input(shape=input_shape, name='input'))
    for layer in model.layers[:-remove_n_layers]:
        new_model.add(layer)
    return new_model


def apply_cnn_to_data(cnn_model, prepared_data, is_flattened=False):
    predictions = []
    for i in range(len(prepared_data)):
        pred = cnn_model.predict(prepared_data[i])
        pred_shape = pred.shape
        if not is_flattened:
            reshaped_pred = pred.reshape(pred_shape[0] * pred_shape[1] * pred_shape[2])
        else:
            reshaped_pred = pred.reshape(pred_shape[0] * pred_shape[1])
        predictions.append(reshaped_pred)
    return np.array(predictions)


def fit_with_time_out(alg_ref, param, extracted_features):
    alg = alg_ref(**param).fit(extracted_features)
    return alg


def evaluate_clustering(timeout_function, extracted_features, clustering_algorithm_ref_list, clustering_algorithm_names, dict_of_list_of_dict_params):
    """
    Example: dict_of_list_of_dict_params - {clustering_alg_ref: [{clusters : 5}, {clusters: 2}]}
    """
    name_list, param_list, sil_score_list, dav_bould_score_list, centroid_list, alg_label_list = [], [], [], [], [], []
    for alg_ref, alg_name in zip(clustering_algorithm_ref_list, clustering_algorithm_names):
        for param in dict_of_list_of_dict_params[alg_ref]:
            try:
                #alg = alg_ref(**param).fit(extracted_features)
                clusters = alg_ref(**param).fit_predict(extracted_features)
                sil_metrics = param['metric'] if 'metric' in param.keys() else param['affinity'] if 'affinity' in param.keys() else 'euclidean'
                try:
                    if len(Counter(clusters)) > 1:
                        sil_score = silhouette_score(extracted_features, clusters, metric=sil_metrics)
                    else:
                        sil_score = 0
                except:
                    print(f'Failed to use distance metric {sil_metrics}')
                    sil_score = silhouette_score(extracted_features, clusters)
                dav_boul_score = davies_bouldin_score(extracted_features, clusters)
                print(f'{alg_name}, sil: {sil_score}, dev boul: {dav_boul_score}')
                try:
                    centroids = alg.cluster_centers_
                except:
                    centroids = None
                name_list.append(alg_name)
                param_list.append(param)
                sil_score_list.append(sil_score)
                dav_bould_score_list.append(dav_boul_score)
                centroid_list.append(centroids)
                alg_label_list.append(clusters)
            except:
                continue
    return name_list, param_list, sil_score_list, dav_bould_score_list, centroid_list, alg_label_list


def preloaded_models(models_data):
    preloaded_feature_extractor_dict = {}
    for model_name in models_data.keys():
        model_time_step = models_data[model_name]['time_steps']
        model = load_h5_model(models_data[model_name]['path'])
        feature_extractor = remove_last_layers(model, models_data[model_name]['remove_layers_flat'])
        temp_dict = {'model_name': model_name, 'feature_extractor': feature_extractor}
        if model_time_step in preloaded_feature_extractor_dict.keys():
            preloaded_feature_extractor_dict[model_time_step].append(temp_dict)
        else:
            preloaded_feature_extractor_dict[model_time_step] = [temp_dict]
    return preloaded_feature_extractor_dict

        
        

def main_function_prototype(timeout_function, preloaded_model_dict, time_series_data, name_mapping_from, time_steps_list, models_data, data_params, clustering_algorithm_ref_list,
                            clustering_algorithm_names, dict_of_list_of_dict_params, save_df_file_path='results.csv', use_pca=False):
    """Runs examples and returns results.

    Args:
        time_series_data (dataframe): Dataset.
        name_mapping_from (dict): Mapping dict.
        models_data (dict): Dict which contains information about models, ex: {'model_name': {'path': .., 'remove_layers': ..}, ..}
        splitting_data_params (list): List of different params, ex: [{'time_steps': 40, 'overlap': 0, 'date_from': .., 'date_to': ..}, {..}]
        clustering_algorithm_ref_list (list): List of algorithm references.
        clustering_algorithm_names (list): List of algorithm names.
        dict_of_list_of_dict_params (dict): Dict of each algorithm parameters. Example: 
            {clustering_alg_ref: [{clusters : 5}, {clusters: 2}]}

    """
    # preload models and remove last layers and put them in dict with keys are time_steps
    cols = ['model_name', 'time_steps', 'overlap', 'date_from', 'date_to', 'alg_name', 'param_list', 'sil_score', 'dav_bould_score', 'centroids', 'labels', 'explained_var']
    # Save somewhere extracted features because they will be required for analysis maybe???????????? and also save data_list and info
    # TODO: open and load pickle file.
    final_result_df = pd.DataFrame(columns=cols)

    for time_step in time_steps_list:
        print(f'time_step {time_step}')
        for data_param_dict in data_params[time_step]:

            overlap = data_param_dict['overlap']
            date_from = data_param_dict['date_from']
            date_to = data_param_dict['date_to']
            data_list, info = data_processing(time_series_data, name_mapping_from, time_step, date_from=date_from,
                                              date_to=date_to, overlap=overlap)
            print(f'Overlap {overlap}')
            used_models = preloaded_model_dict[time_step]
            for model_res in used_models:
                model_name = model_res['model_name']
                feature_extractor = model_res['feature_extractor']
                extracted_features = apply_cnn_to_data(feature_extractor, data_list, is_flattened=True) # adjust to flatted layer
                explained_var = 0
                if use_pca:
                    extracted_features_scaled = StandardScaler().fit_transform(extracted_features)
                    pca = decomposition.PCA(n_components=15)
                    pca.fit(extracted_features_scaled)
                    extracted_features = pca.transform(extracted_features_scaled)
                    explained_var = sum(pca.explained_variance_ratio_ * 100)
                    
                # add timeout inside evaluate_clustering
                name_list, param_list, sil_score_list, dav_bould_score_list, centroid_list, alg_label_list = evaluate_clustering(timeout_function,
                                                                                                                                 extracted_features,
                                                                                                                                 clustering_algorithm_ref_list, clustering_algorithm_names,
                                                                                                                                 dict_of_list_of_dict_params)
                res_len = len(name_list)
                temp_df = pd.DataFrame.from_dict({cols[0]: [model_name] * res_len,
                                              cols[1]: [time_step] * res_len,
                                              cols[2]: [overlap] * res_len,
                                              cols[3]: [date_from] * res_len,
                                              cols[4]: [date_to] * res_len,
                                              cols[5]: name_list,
                                              cols[6]: param_list,
                                              cols[7]: sil_score_list,
                                              cols[8]: dav_bould_score_list,
                                              cols[9]: centroid_list,
                                              cols[10]: alg_label_list,
                                              cols[11]: [explained_var] * res_len
                                             })
                final_result_df = final_result_df.append(temp_df)
        
    final_result_df.to_csv(save_df_file_path)
    return final_result_df

# Iterate through models after splitting data, if models have the same input value
def first_trial():
    timeout_function = 2 # in
    
    #path_to_file = os.path.join('pension_funds', 'manapensija', 'hist_en_latin.csv')
    #data = pd.read_csv(path_to_file, encoding='cp775')
    #models_data = pickle.load(open(os.path.join('my_models', 'models_info_updated.pickle'), 'rb'))
    #time_series_data, name_mapping_to, name_mapping_from = read_data(path_to_file)

    model_num = len(models_data)
    # getting unique time steps
    list_of_possible_time_steps = np.unique([models_data[key]['time_steps'] for key in models_data.keys()])
    processed_data_params, data_len = params.splitting_data_params(list_of_possible_time_steps)
    alg_ref_list, alg_name_list, dict_of_list_of_dict_params, total_params = params.algorithm_params()

    total_runs = model_num * total_params * data_len
    feature_extractor_dict = preloaded_models(models_data)
    final_result_df = main_function_prototype(None, feature_extractor_dict, time_series_data, name_mapping_from, list_of_possible_time_steps, models_data, 
                                              processed_data_params, alg_ref_list, alg_name_list, dict_of_list_of_dict_params, save_df_file_path='basic_pca_results.csv', use_pca=True)


def advanced_trial():
    model_num = len(use_model_names)
    list_of_possible_time_steps = np.unique([models_data[key]['time_steps'] for key in models_data.keys() if key in use_model_names])
    processed_data_params, data_len = params.splitting_data_params_advanced(list_of_possible_time_steps)
    feature_extractor_dict = preloaded_models(models_data)

def outlier_trial():
    model_num = len(use_model_names)
    list_of_possible_time_steps = np.unique([models_data[key]['time_steps'] for key in models_data.keys() if key in use_model_names])
    processed_data_params, data_len = params.splitting_data_params_advanced(list_of_possible_time_steps)
    feature_extractor_dict = preloaded_models(models_data)


if __name__ == '__main__':
    
    
    path_to_file = os.path.join('pension_funds', 'manapensija', 'hist_en_latin.csv')
    data = pd.read_csv(path_to_file, encoding='cp775')
    models_data = pickle.load(open(os.path.join('my_models', 'models_info_updated.pickle'), 'rb'))
    time_series_data, name_mapping_to, name_mapping_from = read_data(path_to_file)
    
    
    use_model_names = ['fifth_model.h5', 'first_model.h5', 'second_model.h5', 'tenth.h5',
                       'tenth_1.h5', 'third_model.h5',
                       'sixth_model.h5', 'seventh_model.h5', 
                       'eleventh.h5', 'fourth_model.h5', 'pred_15_overlap_20_another.h5',
                       'pred_10_overlap_10_m (1).h5']
    
    
    first_trial()
    
    