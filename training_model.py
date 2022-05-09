import os

import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
from numpy.random import seed
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

from IPython.display import SVG
from tensorflow.keras.utils import plot_model
from sklearn import metrics
import pickle
from collections import Counter
import time


seed(42)


def initial_data_preparation(data_df, historical_stocks, y_column, date_min=None, date_max=None, use_sectors=None):
    historical_stocks.dropna(inplace=True)
    #stock_prices.dropna(inplace=True)

    read_time = '%Y-%m-%d'
    keys = historical_stocks[y_column].value_counts().keys()
    values = np.arange(0, len(keys), dtype=int)
    # creating cat mapping
    mapping = {}
    for key, val in zip(keys, values):
        if use_sectors and not key in use_sectors:
            val = -1
        mapping[key] = val
    data_df['date'] = [datetime.datetime.strptime(date, read_time) for date in data_df['date'].values]
    new_y_column_name = f'{y_column}_code'
    historical_stocks[new_y_column_name] = historical_stocks[y_column].map(mapping)
    if use_sectors:
        historical_stocks = historical_stocks.loc[historical_stocks[new_y_column_name] != -1]
    if date_min or date_max:
        if not date_min is None:
            data_df = data_df.loc[(data_df['date'] >= date_min)]
        if not date_max is None:
            data_df = data_df.loc[(data_df['date'] <= date_max)]

    return data_df, historical_stocks, new_y_column_name, mapping


def map_value_to_category(mapping_dict, value):
    final_class_value = None
    for key, class_val in mapping_dict.items():
        if key[0] != 0 and key[1] != 0:
            final_class_value = class_val if value >= key[0] and value <= key[1] else final_class_value
        elif key[0] != 0:
            final_class_value = class_val if value <= key[0] else final_class_value
        else:
            final_class_value = class_val if value >= key[1] else final_class_value
    
    return final_class_value
        

def prepare_data_for_price_classification(data, x_column_names, y_dict, time_steps, predict_future_n_step, overlap_steps, class_mapping=None):
    # class_mapping keys are tuples indicating value ranges which are in precentages.
    X_list = []
    y_list = []
    valid_tickers = {}
    grouped_data = data.groupby(by='ticker')
    for y_value in y_dict.keys():
        valid_tickers[y_value] = {'n_obs': [], 'tickers': []}
        tickers = y_dict[y_value]
        for ticker in tickers:
            try:
                tick_data = grouped_data.get_group(ticker)
                tick_data.sort_values('date', ascending=True)
                num_observations = (len(tick_data) - time_steps - predict_future_n_step) // (time_steps - overlap_steps)
                #num_observations = (len(tick_data)- time_steps - predict_future_n_step) // overlap_steps - 2
                #num_observations = num_observations if num_observations > 0 else 0
                valid_tickers[y_value]['tickers'].append(ticker)
                valid_tickers[y_value]['n_obs'].append(num_observations)
            except:
                continue 
            for i in range(num_observations):
                min_ind = i * time_steps  if i == 0 else i * time_steps - overlap * i
                max_ind = (i + 1) * time_steps if i == 0 else (i + 1) * time_steps - overlap * i
                X_val = tick_data.iloc[min_ind: max_ind][x_column_names].values
                try:
                    x_comparable_before = X_val[-1] # we take last value and compare it to max_ind + predict_future_n_step and if it is above >0 then class 1 if <= 0 then class is 0
                    x_comparable_after = tick_data.iloc[max_ind + predict_future_n_step][x_column_names].values
                    if class_mapping is None:
                        y_val = 1 if x_comparable_after > x_comparable_before else 0
                    else:
                        comparison_val = x_comparable_after / x_comparable_before * 100 - 100
                        y_val = map_value_to_category(class_mapping, comparison_val)
                except:
                    print('nooooooo')
                X_list.append(X_val)
                y_list.append(y_val)
    return np.array(X_list), y_list, valid_tickers 


def prepare_data_for_model(data, x_column_names, y_dict, time_steps, overlap_steps=None, print_missing_tickers=False):
    X_list = []
    y_list = []
    valid_tickers = {}
    grouped_data = data.groupby(by='ticker')
    for y_value in y_dict.keys():
        valid_tickers[y_value] = {'n_obs': [], 'tickers': []}
        tickers = y_dict[y_value]
        for ticker in tickers:
            try:
                tick_data = grouped_data.get_group(ticker)
                tick_data.sort_values('date', ascending=True)
                #time_steps_len = len(tick_data) - overlap_steps if overlap_steps else len(tick_data)
                num_observations = len(tick_data) // time_steps
                valid_tickers[y_value]['tickers'].append(ticker)
                valid_tickers[y_value]['n_obs'].append(num_observations)
            except:
                if print_missing_tickers:
                    print(f'{ticker} does not exist!')
                continue
            for i in range(num_observations):
                min_ind = i*time_steps #if i == 0 else i*time_steps - overlap_steps if overlap_steps else i*time_steps
                max_ind = (i+1)*time_steps #if i == 0 else (i+1)*time_steps - overlap_steps if overlap_steps else i*time_steps
                X_list.append(tick_data.iloc[min_ind: max_ind][x_column_names].values)
                y_list.append(y_value)
    return np.array(X_list), y_list, valid_tickers


def read_data():
    stock_prices_path = 'datasets\\historical_stock_prices.csv'
    historical_stocks_path = 'datasets\\historical_stocks.csv'
    stock_prices = pd.read_csv(stock_prices_path)
    historical_stocks = pd.read_csv(historical_stocks_path)
    return stock_prices, historical_stocks



def create_y_dict(historical_stocks, y_column, filter_min=None, filter_max=None):
    stock_dict = {}
    y_data = historical_stocks[y_column].value_counts()
    y_data_filtered = y_data.copy()
    if not filter_min is None:
        y_data_filtered = y_data_filtered[(y_data >= filter_min)]
    if not filter_max is None:
        y_data_filtered = y_data_filtered[(y_data <= filter_max)]
    filtered_data_keys = y_data_filtered.keys()
    for key in filtered_data_keys:
        stock_dict[key] = historical_stocks[historical_stocks[y_column] == key]['ticker'].values
    return stock_dict


def train_val_test_split(X, y, test_ratio):
    X = np.squeeze(X)
    #X = 
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_ratio, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def get_layer_description(model_description, dense=None, dropout=None, max_pool_size=None, strides='default',
                          batch_norm=None, filters=None, kernel=None, padding='default', activation='leaky_relu_alpha_0.3'):
    layer_description = None
    if dense:
        layer_description = {'dense': dense}
    if dropout:
        layer_description = {'dropout': dropout}
    if max_pool_size:
        layer_description = {'max_pool': max_pool_size, 'stride': strides, 'padding': padding}
    if batch_norm:
        layer_description = {'batch_norm': True}
    if filters:
        layer_description = {'filters': filters, 'kernel': kernel, 'padding': padding, 'strides': strides, 'activation': activation}
    model_description.append(layer_description)


def create_model(data_shape, class_num):
    """
    Model description example:
        CNN:
            get_layer_description(description, filters=16, kernel=2)

        BatchNormalization:
            get_layer_description(description, batch_norm=True)
        
        MaxPool:
            get_layer_description(description, max_pool_size=2, strides=1, padding='valid')
            
        Dropout:
            get_layer_description(description, dropout=0.2)
        
        Dense:
            get_layer_description(description, dense=64)
    
    """
    description = []
    #leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
    inputs = keras.Input(shape=data_shape)
    x = keras.layers.Conv1D(filters=32, kernel_size=2, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(inputs)
    get_layer_description(description, filters=32, kernel=2)
    
    #x = keras.layers.MaxPool1D(pool_size=2)(x)
    #get_layer_description(description, max_pool_size=2)

    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.MaxPool1D(pool_size=2)(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    get_layer_description(description, filters=64, kernel=3)
    
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2)(x)
    get_layer_description(description, max_pool_size=2)
    
    x = keras.layers.Conv1D(filters=124, kernel_size=5, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    get_layer_description(description, filters=124, kernel=5, padding='same')
    
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.MaxPool1D(pool_size=2)(x)
    
    #x = keras.layers.Conv1D(filters=128, kernel_size=5, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    #get_layer_description(description, filters=128, kernel=5)
    
    x = keras.layers.MaxPool1D(pool_size=2)(x)
    get_layer_description(description, max_pool_size=2)


    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.3)(x)
    get_layer_description(description, dropout=0.3)
    
    x = keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    get_layer_description(description, dense=256)

    x = keras.layers.Dropout(0.2)(x)
    get_layer_description(description, dropout=0.2)

    x = keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    get_layer_description(description, dense=64)

    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)

    output_activation = 'softmax' if class_num > 2 else 'sigmoid'
    output_class = class_num if class_num > 2 else 1
    output = keras.layers.Dense(output_class, activation=output_activation)(x)
    get_layer_description(description, dense=output_class)
    
    model = keras.Model(inputs=inputs, outputs=output)
    return model, description


def create_data_description(timesteps, timesteps_days, data_size, class_freq, overlap=None, overlap_days=None, predict_future_time_steps_days=None,
                            class_num=None, predict_future_time_steps=None, min_date=None, max_date=None,
                            description=None, y_column=None, y_tickers=None, file_name='my_models\\data_description.csv'):
    new_dict = {}
    data = pd.DataFrame()
    if os.path.exists(file_name):
        data = pd.read_csv(file_name)
    data_keys = data.columns
    new_num = len(data_keys) + 1 if len(data_keys) > 0 else 0
    data_dict = {'timesteps': timesteps,
                 'timesteps_days': timesteps_days,
                 'overlap': overlap,
                 'overlap_days': overlap_days,
                 'data_size': data_size,
                 'class_freq': class_freq,
                 'predict_future_time_steps': predict_future_time_steps,
                 'predict_future_time_steps_days': predict_future_time_steps_days,
                 'min_date': min_date,
                 'max_data': max_date,
                 'y_tickers': y_tickers,
                 'y_column': y_column,
                 'description': description
                 }
    new_dict[new_num] = data_dict
    #with open(file_name, 'wb') as handle:
    #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    data = data.append(pd.DataFrame.from_dict(new_dict))
    data.to_csv(file_name)


def get_dataset_log(file_name='my_models\\data_description.csv'):
    return pd.read_csv(file_name)


def calculate_auc(y_true, y_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
    #metrics.auc(fpr, tpr)
    return metrics.roc_auc_score(y_true, y_pred)


def create_model_logs(model_name, train_acc, val_acc, train_loss, val_loss, val_auc, epochs, train_time,
                      learning_rate, dummy_freq_score, dummy_random_score, model_description=None, data_description_num=None, batch_size=32,
                      model_status=0, file_name='my_models\\model_description.csv', model_save_name=None):
    row = {'name': model_name,
           'train_time': train_time,
           'epochs': epochs,
           'train_acc': train_acc,
           'val_acc': val_acc,
           'train_loss': train_loss,
           'val_loss': val_loss,
           'val_auc': val_auc,
           'learning_rate': learning_rate,
           'batch_size': batch_size,
           'dummy_freq_score': dummy_freq_score,
           'dummy_random_score': dummy_random_score,
           'model_status': model_status,
           'saved_model_name': model_save_name,
           'description': model_description,
           'data_mode': data_description_num}
    # Create pandas df with such information:
    df = pd.DataFrame()
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    df = df.append(row, ignore_index=True)
    df.to_csv(file_name, index=False)
    

def create_model_log_auto(model, model_name='binary_classifier', model_status=0, model_save_name=None): # X_train, y_train, X_test, y_test,
    if not multi_class:
        train_loss, train_acc = model.evaluate(X_train, y_train)
        test_loss, test_acc = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        roc = calculate_auc(y_test, y_pred)
    else:
        train_loss, train_acc = model.evaluate(X_train, y_train_OHE)
        test_loss, test_acc = model.evaluate(X_test, y_test_OHE)
        y_pred = model.predict(X_test)
        roc = calculate_auc(y_test_OHE, y_pred)

    print(f'AUC: {roc}')
    create_model_logs(model_name, train_acc, test_acc, train_loss, test_loss, roc, epochs, train_time, learning_rate,
                      dummy_freq_score, dummy_random_score, description, data_description_num, batch_size=batch_size, model_status=model_status,
                      model_save_name=model_save_name)


def prepare_freq_df(used_ticker_dict, column_name, time_step, mapping):
    key_list, ticker_count_list, total_obs_list, ticker_list, freq_ratio_list = [], [], [], [], []
    keys = used_ticker_dict.keys()
    mapping_values = list(mapping.values())
    mapping_keys = list(mapping.keys())
    for key in keys:
        some_key = key
        count = np.sum(used_ticker_dict[key]['n_obs'])
        ticker_count_list.append(count)
        total_obs_list.append(count * time_step)
        ticker_list.append(used_ticker_dict[key]['tickers'])
        if key in mapping_values:
            position = mapping_values.index(key)
            some_key = mapping_keys[position]
        key_list.append(some_key)
    
    total = np.sum(total_obs_list)
    
    df = pd.DataFrame.from_dict({f'{column_name}': key_list,
                                 'ticker_count': ticker_count_list,
                                 'num_obs': total_obs_list,
                                 'num_obs_total_prec': np.array(total_obs_list) / total * 100,
                                 'tickers': ticker_list
                                 })

    return df.sort_values(by='num_obs', ascending=False)


def plot_history(history):
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, len(accuracy) + 1)

    plt.rcParams['figure.figsize'] = [10, 5]
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss', color='red')
    plt.plot(epochs,val_loss , 'b', label='Validation loss', color='green')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'bo', label='Training acc', color='red')
    plt.plot(epochs, val_accuracy, 'b', label='Validation acc', color='green')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_model_data(path='my_models\\model_description.csv'):
    return pd.read_csv(path)


def load_h5_model(model_name):
    reconstructed_model = tf.keras.models.load_model(f'{model_name}.h5',
                                                     custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
    return reconstructed_model
# Find AUC for other classifiers

if __name__ == '__main__':
    
    ## Getting timesteps:
    # cat = stock_prices[stock_prices['ticker'] == 'CAT']
    # cat[:50]
    y_column = 'sector' #'industry' # 'sector'
    min_date = datetime.datetime(2010, 1, 1)
    max_date = None #datetime.datetime(2012, 12, 31)
    x_columns = ['adj_close']
    y_min = 2 #
    y_max = None #15000
    time_steps = 70
    predict_time_n_steps = 5
    overlap = 0
    test_ratio = 0.1
    model_name = os.path.join('my_models', 'eleventh.h5')
    use_sectors = None#['FINANCE', 'CONSUMER SERVICES', 'HEALTH CARE', 'TECHNOLOGY', 'CAPITAL GOODS']
    model_evaluation_dict = {'overfitting': 2, 'underfitting': 1, 'unknown': 0}
    class_mapping = None#{(-3, 3): 0, (0, 3): 1, (-3, 0): 2}
    multi_class = True

    stock_prices, historical_stocks = read_data()
    stock_prices, historical_stocks, y_column_adj, mapping = initial_data_preparation(stock_prices, historical_stocks, y_column,
                                                                                      date_min=min_date, date_max=max_date, use_sectors=use_sectors)
    
    # converting timesteps into days:
    cat = stock_prices[stock_prices['ticker'] == 'CAT']
    time_step_in_days = cat.iloc[time_steps]['date'] - cat.iloc[0]['date'] if time_steps is not None else 0
    predict_time_n_in_days = cat.iloc[predict_time_n_steps]['date'] - cat.iloc[0]['date'] if predict_time_n_steps is not None else 0
    overlap_in_days = cat.iloc[overlap]['date'] - cat.iloc[0]['date'] if overlap is not None else 0

    y_dict = create_y_dict(historical_stocks, y_column_adj, y_min, y_max)
    X, y, used_tickers = prepare_data_for_model(stock_prices, x_columns, y_dict, time_steps, overlap_steps=None)
    #X, y, used_tickers = prepare_data_for_price_classification(stock_prices, x_columns, y_dict, time_steps, predict_time_n_steps, overlap, class_mapping=class_mapping)
    print('Prepared data')
    X_train, X_test, y_train, y_test = train_val_test_split(X, y, test_ratio)
    y_train_OHE = tf.keras.utils.to_categorical(y_train, num_classes=len(y_dict))
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test_OHE = tf.keras.utils.to_categorical(y_test, num_classes=len(y_dict))
    print('Splitted data')
    #binary_model, description = create_model([X_train.shape[1], 1], 2)
    """model, description = create_model([X_train.shape[1], 1], len(class_mapping))
    learning_rate = 0.0001
    batch_size = 32
    epochs = 30"""
    #binary_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    #history = binary_model.fit(X_train, y_train, batch_size=batch_size, verbose=1, shuffle=True, epochs=epochs, validation_data=(X_test, y_test))
    """model = create_model([X_train.shape[1], 1], len(y_dict))
    print('Created model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    #binary_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    print('Compiled')

    y_train_OHE = tf.keras.utils.to_categorical(y_train, num_classes=len(y_dict))
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test_OHE = tf.keras.utils.to_categorical(y_test, num_classes=len(y_dict))
    history = model.fit(X_train, y_train_OHE, batch_size=32, verbose=1, shuffle=True, epochs=epochs, validation_data=(X_test, y_test_OHE))"""
    # model.save(model_name)
    # model.evaluate(X_test, y_test_OHE)
    
    #history = binary_model.fit(X_train, y_train, batch_size=32, verbose=1, shuffle=True, epochs=100, validation_data=(X_test, y_test))
    
    
    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(X_test, y_test)
    dummy_freq_score = clf.score(X_test, y_test)
    clf1 = DummyClassifier(strategy='uniform', random_state=42)
    clf1.fit(X_test, y_test)
    dummy_random_score = clf1.score(X_test, y_test)
    train_time = 0
    #start = time.process_time()
    #train_time = time.process_time() - start
    #print(time.process_time() - start)
    # creating dataset logs
    #create_data_description(time_steps, time_step_in_days, len(X), Counter(y), overlap=overlap, overlap_days=overlap_in_days,
    #                        predict_future_time_steps_days=predict_time_n_in_days, class_num=2, predict_future_time_steps=predict_time_n_steps,
    #                        min_date=min_date, max_date=max_date, description=None, y_column=y_column,
    #                        y_tickers=used_tickers)
    data_description_num = 2
    #sparse_categorical_crossentropy