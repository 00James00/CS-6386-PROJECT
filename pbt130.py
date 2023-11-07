import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import csv
import sys
import random
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import warnings
import time
import copy
import argparse
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler, MinMaxScaler, Binarizer, KBinsDiscretizer, PowerTransformer, QuantileTransformer
#from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler, MinMaxScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


import csv  # Import the csv module
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd

iterations_for_pop_size = {}
iterations_for_num_components = {}
results_table = []

def generate_dic_and_names(components=6):
    temp_preprocessors_dic = {
        "binarizer": Binarizer(),
        "standardizer": StandardScaler(),
        "normalizer": Normalizer(),
        "maxabs": MaxAbsScaler(),
        "minmax": MinMaxScaler(),
        "power_trans": PowerTransformer(),
        "quantile_trans": QuantileTransformer(random_state=0)
    }
    # temp_operator_names = ["binarizer", "standardizer", "normalizer", "maxabs", "minmax", "power_trans", "quantile_trans",     ]
    temp_operator_names = ["binarizer", "standardizer", "normalizer", "maxabs", "minmax", "power_trans", "quantile_trans", "binarizer", "standardizer", "normalizer", "maxabs", "minmax", "power_trans", "quantile_trans" ]


    binarizer_threshold = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    normalizer_norm = ['l1', 'l2', 'max']
    standard_with_mean = [True, False]
    power_standardize = [True, False]
    quantile_n_quantiles = [10, 100, 200, 500, 1000, 1200, 1500, 2000]
    quantile_distribution = ['uniform', 'normal']

    if components > 0:
        for item in binarizer_threshold:
            temp_operator_names.append(f"binarizer_threshold_{str(item)}")
            temp_preprocessors_dic[f"binarizer_threshold_{str(item)}"] = Binarizer(threshold=item)

    if components > 1:
        for item in normalizer_norm:
            temp_operator_names.append(f"normalizer_norm_{item}")
            temp_preprocessors_dic[f"normalizer_norm_{item}"] = Normalizer(norm=item)

    if components > 2:
        for item in standard_with_mean:
            temp_operator_names.append(f"standard_with_mean_{str(item)}")
            temp_preprocessors_dic[f"standard_with_mean_{str(item)}"] = StandardScaler(with_mean=item)

    if components > 3:
        for item in power_standardize:
            temp_operator_names.append(f"power_standardize_{str(item)}")
            temp_preprocessors_dic[f"power_standardize_{str(item)}"] = PowerTransformer(standardize=item)

    if components > 4:
        for item in quantile_n_quantiles:
            temp_operator_names.append(f"quantile_n_quantiles_{str(item)}")
            temp_preprocessors_dic[f"quantile_n_quantiles_{str(item)}"] = QuantileTransformer(n_quantiles=item, random_state=0)

    if components > 5:
        for item in quantile_distribution:
            temp_operator_names.append(f"quantile_distribution_{str(item)}")
            temp_preprocessors_dic[f"quantile_distribution_{str(item)}"] = QuantileTransformer(output_distribution=item, random_state=0)
        #second
    if components > 6:
        for item in binarizer_threshold:
            temp_operator_names.append(f"binarizer_threshold_{str(item)}")
            temp_preprocessors_dic[f"binarizer_threshold_{str(item)}"] = Binarizer(threshold=item)

    if components > 7:
        for item in normalizer_norm:
            temp_operator_names.append(f"normalizer_norm_{item}")
            temp_preprocessors_dic[f"normalizer_norm_{item}"] = Normalizer(norm=item)

    if components > 8:
        for item in standard_with_mean:
            temp_operator_names.append(f"standard_with_mean_{str(item)}")
            temp_preprocessors_dic[f"standard_with_mean_{str(item)}"] = StandardScaler(with_mean=item)

    if components > 9:
        for item in power_standardize:
            temp_operator_names.append(f"power_standardize_{str(item)}")
            temp_preprocessors_dic[f"power_standardize_{str(item)}"] = PowerTransformer(standardize=item)

    if components > 10:
        for item in quantile_n_quantiles:
            temp_operator_names.append(f"quantile_n_quantiles_{str(item)}")
            temp_preprocessors_dic[f"quantile_n_quantiles_{str(item)}"] = QuantileTransformer(n_quantiles=item, random_state=0)

    if components > 11:
        for item in quantile_distribution:
            temp_operator_names.append(f"quantile_distribution_{str(item)}")
            temp_preprocessors_dic[f"quantile_distribution_{str(item)}"] = QuantileTransformer(output_distribution=item, random_state=0)

    

    return temp_preprocessors_dic, temp_operator_names

# preprocessors_dic, operator_names = generate_dic_and_names(7)

# print(operator_names)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="name of dataset")
ap.add_argument("-c", "--classifier", required=True,help="name of classifier")
ap.add_argument("-max_time", "--max_time_limit", required=True,help="number of max_time_limit")
args = vars(ap.parse_args())

dataset = args['dataset']
classifier = args['classifier']
max_time_limit = int(args['max_time_limit'])

# [0, 42, 167,578,1440]
seed=1440
all_list = [i for i in range(0, 5000000)]
np.random.seed(seed)
top_choice_seeds = np.random.choice(all_list, 2000000, replace=False)
resample_seeds = np.random.choice(all_list, 2000000, replace=False)
mutate_way_seeds = np.random.choice(all_list, 2000000, replace=False)
op_seeds = np.random.choice(all_list, 2000000, replace=False)
pos_seeds = np.random.choice(all_list, 2000000, replace=False)


def load_data():
    train_data_dir = "/Users/cba/Desktop/cs_6386/research/Auto-FP-for-james/auto_fp/Exp_extended_space/PBT/" + dataset + "_train.csv"
    X_train = []
    y_train = []
    with open(train_data_dir) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if (not '?' in row[0: len(row) - 1]):
                X_train.append(list(map(lambda x: float(x), row[0: len(row) - 1])))
                y_train.append(float(row[-1]))
                
    valid_data_dir = "/Users/cba/Desktop/cs_6386/research/Auto-FP-for-james/auto_fp/Exp_extended_space/PBT/" + dataset + "_valid.csv"
    X_valid = []
    y_valid = []
    with open(valid_data_dir) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if (not '?' in row[0: len(row) - 1]):
                X_valid.append(list(map(lambda x: float(x), row[0: len(row) - 1])))
                y_valid.append(float(row[-1]))
    return X_train, X_valid, y_train, y_valid

X_train, X_valid, y_train, y_valid = load_data()

def get_model():
    if classifier == 'LR':
        model = LogisticRegression(random_state=0, n_jobs=1)
    elif classifier == 'XGB':
        model =  XGBClassifier(random_state=0, nthread=1, n_jobs=1)
    elif classifier == 'MLP':
        model = MLPClassifier(random_state=0)
    return model

def mutate(parent, mutate_way_seed, op_seed, pos_seed):
    mutations = []
    if (len(parent) == 1):
        mutations = ["add", "replace"]
    elif (len(parent) == max_len):
        mutations = ["delete", "replace", "switch"]
    else:
        mutations = ["add", "delete", "replace", "switch"]
    #random.seed(1440)
    np.random.seed(mutate_way_seed)
    mutate_type = np.random.choice(mutations)
    #print(mutate_type)

    child = []
    #print(parent)
    if (mutate_type == "add"):
        if len(parent) == 1:
            pos = 0
        else:
            np.random.seed(pos_seed)
            pos = np.random.randint(0, len(parent) - 1)
        np.random.seed(op_seed)
        op = np.random.choice(operator_names)
        for i in range(pos + 1):
            child.append(parent[i])
        child.append(op)
        for i in range(pos + 1, len(parent)):
            child.append(parent[i])
    elif (mutate_type == "delete"):
        if len(parent) == 1:
            pos = 0
        else:
            np.random.seed(pos_seed)
            pos = np.random.randint(0, len(parent) - 1)
        for i in range(pos):
            child.append(parent[i])
        for i in range(pos + 1, len(parent)):
            child.append(parent[i])
    elif (mutate_type == "replace"):
        if len(parent) == 1:
            pos = 0
        else:
            np.random.seed(pos_seed)
            pos = np.random.randint(0, len(parent) - 1)
        np.random.seed(op_seed)
        op = np.random.choice(operator_names)
        for i in range(pos):
            child.append(parent[i])
        child.append(op)
        for i in range(pos + 1, len(parent)):
            child.append(parent[i])
    elif (mutate_type == "switch"):
        np.random.seed(pos_seed)
        pos = np.random.choice(np.arange(len(parent)), size=2, replace=False)
        pos1 = min(pos)
        pos2 = max(pos)
        for i in range(pos1):
            child.append(parent[i])
        child.append(parent[pos2])
        for i in range(pos1 + 1, pos2):
            child.append(parent[i])
        child.append(parent[pos1])
        for i in range(pos2 + 1, len(parent)):
            child.append(parent[i])
    return child

def perturbation(top_pipe, resample_probablity,
                 resample_seed, mutate_way_seed, op_seed, pos_seed):
    result_pipe = []
    np.random.seed(resample_seed)
    prob = np.random.random()
    if prob < resample_probablity:
        ''' Do resampling '''
        np.random.seed(resample_seed)
        length = np.random.randint(1, 7)
        for i in range(length):
            result_pipe.append(operator_names[np.random.randint(1, len(operator_names)) - 1])
    else:
        ''' Do perturbation (small mutation) '''
        result_pipe = mutate(top_pipe, mutate_way_seed, op_seed, pos_seed)
    return result_pipe

def exploit_and_explore(population, bot_trial_info, top_trial_info, resample_probability,
                        resample_seed, mutate_way_seed, op_seed, pos_seed):
    result_pipe = []
    bot_index = population.index(bot_trial_info)
    top_index = population.index(top_trial_info)
    bot_pipe = bot_trial_info[0]
    top_pipe = top_trial_info[0]

    #for i in range(len(top_pipe)):
    #    operator = top_pipe[i]
    #    choices = operator_names
    #    ub, uv = len(choices) - 1, choices.index(operator) + 1
    #    lb, lv = 0, choices.index(operator) - 1
    #    idx = perturbation(choices, resample_probability, uv, ub, lv, lb, random_state)
    #    result_pipe.append(choices[idx])
    result_pipe = perturbation(top_pipe, resample_probability,
                               resample_seed, mutate_way_seed, op_seed, pos_seed)
    return result_pipe


population_size = 500
population = []
fraction = 0.2
resample_probability = 0.25
epochs = 1000000
max_len = 7

global_start = time.time()
time_limit_reached = False

def pbt_with_population_size(population_size, max_time_limit, num_components):
    # Begin PBT code
 
    global_start = time.time()
    time_limit_reached = False
    current_epoch = 0

    preprocessors_dic, operator_names = generate_dic_and_names(num_components)

    #Generate random pipelines, which equal to population size
    while (len(population) < population_size):
        rand_start = time.time()
        length = np.random.randint(1, max_len)
        temp_pipe = []
        for i in range(length):
            temp_pipe.append(operator_names[np.random.randint(1, len(operator_names)) - 1])
        rand_end = time.time()
        pick_time = rand_end - rand_start

        generate_pipe_start = time.time()
        pipe_str = ""
        if (length == 1):
            real_pipe = preprocessors_dic[temp_pipe[0]]
            pipe_str = temp_pipe[0]
        else:
            real_pipe = make_pipeline(preprocessors_dic[temp_pipe[0]], preprocessors_dic[temp_pipe[1]])
            pipe_str = temp_pipe[0] + "," + temp_pipe[1]
            for i in range(2, length):
                real_pipe = make_pipeline(real_pipe, preprocessors_dic[temp_pipe[i]])
                pipe_str += "," + temp_pipe[i]
        generate_pipe_end = time.time()

        model = get_model()
        prep_train_start, prep_train_end = 0, 0
        prep_valid_start, prep_valid_end = 0, 0
        train_start, train_end = 0, 0
        pred_start, pred_end = 0, 0
        eval_score_start, eval_score_end = 0, 0

        try:
            prep_train_start = time.time()
            X_train_new = real_pipe.fit_transform(X_train)
            prep_train_end = time.time()
            prep_valid_start = time.time()
            X_valid_new = real_pipe.transform(X_valid)
            prep_valid_end = time.time()

            train_start = time.time()
            model.fit(np.array(X_train_new), np.array(y_train))
            train_end = time.time()

            pred_start = time.time()
            y_pred = model.predict(np.array(X_valid_new))
            pred_end = time.time()

            eval_score_start = time.time()
            score = accuracy_score(y_valid, y_pred)
            eval_score_end = time.time()

            # print('The accuracy of prediction is:', score)
            f1 = f1_score(y_valid, y_pred, average='weighted')  # replace 'weighted' if needed
            if f1 >= 0.7:
                iterations_for_pop_size.setdefault(population_size, []).append(current_epoch)
                iterations_for_num_components.setdefault(len(temp_pipe), []).append(current_epoch)
                results_table.append((len(temp_pipe), current_epoch, population_size))
            
            # current_epoch += 1
      
            if f1 >= 0.7:
                iterations_for_pop_size[population_size] = current_epoch
                iterations_for_num_components[num_components] = current_epoch
                break
            
            current_epoch += 1
            global_mid = time.time()
            if (global_mid - global_start) >= max_time_limit:
                break

            

        except:
            score = 0
        global_mid = time.time()
        if (global_mid - global_start) >= max_time_limit:
            time_limit_reached = True
        f = open('results/pbt_pick_time_1.csv', 'a')
        f.write(str(pick_time) + "\n")
        f = open('results/pbt_wallock_1.csv', 'a')
        f.write(str(global_mid - global_start) + "\n")
        f = open(f'results/pbt_pipe_1.csv', 'a')
        f.write(pipe_str + "\n")
        f = open(f'results/pbt_score_1.csv', 'a')
        f.write(str(score) + "\n")
        f = open(f'results/pbt_eval_time_1.csv', 'a')
        f.write(str(generate_pipe_end - generate_pipe_start) + "," +
                str(prep_train_end - prep_train_start) + "," +
                str(prep_valid_end - prep_valid_start) + "," +
                str(train_end - train_start) + "," +
                str(pred_end - pred_start) + "," +
                str(eval_score_end - eval_score_start) + "\n")
        if score != 0:
            if time_limit_reached:
                sys.exit()
        else:
            if time_limit_reached:
                sys.exit()
            else:
                continue

        population.append((temp_pipe, score))
        



# First plot, various population size vs. number of iterations it takes to reach F1 score of 0.7
population_sizes = [10, 50, 100, 200, 500, 1000]
for pop_size in population_sizes:
    population = []  # Reset the population for each run
    pbt_with_population_size(pop_size, int(args['max_time_limit']), 7)  # Assuming 7 components for simplicity

# Plot the results for "number of iterations vs. population size"
plt.figure()
plt.plot(list(iterations_for_pop_size.keys()), list(iterations_for_pop_size.values()), marker='o')
plt.xlabel('Population Size')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations to Reach F1 >= 0.7 by Population Size')
plt.xticks(ticks=population_sizes)
plt.grid(True)
plt.show()



#second plot
# The following plot code shows the number of iterations vs. number of components used in pipeline
# iterations_for_num_components = {}  


avg_iterations_list = []

# # Iterate over the number of components.
for num_components in range(1, 12):  # Looping from 1 to 6 components
    iterations_for_pop_size = {}  # Reset for each iteration
    population = []  # Reset the population for each iteration
    pbt_with_population_size(population_size, 1000, num_components)
    
    # Calculate average iterations required for the current component count
    avg_iterations_for_pop_size = np.mean(iterations_for_pop_size[population_size])
    avg_iterations_list.append(avg_iterations_for_pop_size)

# Plot the results
plt.figure()
plt.scatter(range(1, 12), avg_iterations_list)
plt.xlabel('Number of Components')
plt.ylabel('Average Number of Iterations')
plt.title('Average Number of Iterations to Reach F1 >= 0.7 by Number of Components')
plt.xticks(ticks=range(1, 12))
plt.show()








preprocessors_dic, operator_names = generate_dic_and_names(7)

# Using PBT framework to do pipeline searching
current_epoch = 0
count = -1
while current_epoch < epochs:
    current_epoch += 1
    temp_population = []
    #sort population based on accuracy score
    population.sort(key=lambda x : x[1], reverse=True)
    cutoff = int(np.ceil(fraction * len(population)))
    tops = population[:cutoff]
    bottoms = population[len(population) - cutoff:]
    for bottom in bottoms:
        score = 0
        exploit_and_explore_pipe = []
        while score == 0:
            count += 1
            pick_start = time.time()
            np.random.seed(top_choice_seeds[count])
            top_idx = [idx for idx in range(len(tops))]
            top = tops[np.random.choice(top_idx)]
            exploit_and_explore_pipe = exploit_and_explore(population, bottom, top, resample_probability,
                                                           resample_seed=resample_seeds[count],
                                                           mutate_way_seed=mutate_way_seeds[count],
                                                           op_seed=op_seeds[count],
                                                           pos_seed=pos_seeds[count])
            pick_end = time.time()
            pick_time = pick_end - pick_start

            # begin evaluation
            generate_pipe_start = time.time()
            pipe_str = ""
            if (len(exploit_and_explore_pipe) == 1):
                real_pipe = preprocessors_dic[exploit_and_explore_pipe[0]]
                pipe_str = exploit_and_explore_pipe[0]
            else:
                real_pipe = make_pipeline(preprocessors_dic[exploit_and_explore_pipe[0]],
                                          preprocessors_dic[exploit_and_explore_pipe[1]])
                pipe_str = exploit_and_explore_pipe[0] + "," + exploit_and_explore_pipe[1]
                for i in range(2, len(exploit_and_explore_pipe)):
                    real_pipe = make_pipeline(real_pipe,
                                              preprocessors_dic[exploit_and_explore_pipe[i]])
                    pipe_str += "," + exploit_and_explore_pipe[i]
            generate_pipe_end = time.time()

            model = get_model()
            prep_train_start, prep_train_end = 0, 0
            prep_valid_start, prep_valid_end = 0, 0
            train_start, train_end = 0, 0
            pred_start, pred_end = 0, 0
            eval_score_start, eval_score_end = 0, 0

            try:
                prep_train_start = time.time()
                X_train_new = real_pipe.fit_transform(X_train)
                prep_train_end = time.time()

                prep_valid_start = time.time()
                X_valid_new = real_pipe.transform(X_valid)
                prep_valid_end = time.time()

                train_start = time.time()
                model.fit(np.array(X_train_new), np.array(y_train))
                train_end = time.time()

                pred_start = time.time()
                y_pred = model.predict(np.array(X_valid_new))
                pred_end = time.time()

                eval_score_start = time.time()
                score = accuracy_score(y_valid, y_pred)
                eval_score_end = time.time()
              

            except:
                score = 0
                
            global_mid = time.time()
            if (global_mid - global_start) >= max_time_limit:
                time_limit_reached = True
            f = open('results/pbt_pick_time_1.csv', 'a')
            f.write(str(pick_time) + "\n")
            f = open('results/pbt_wallock_1.csv', 'a')
            f.write(str(global_mid - global_start) + "\n")
            f = open('results/pbt_pipe_1.csv', 'a')
            f.write(pipe_str + "\n")
            f = open('results/pbt_score_1.csv', 'a')
            f.write(str(score) + "\n")
            f = open('results/pbt_eval_time_1.csv', 'a')
            f.write(str(generate_pipe_end - generate_pipe_start) + "," +
                    str(prep_train_end - prep_train_start) + "," +
                    str(prep_valid_end - prep_valid_start) + "," +
                    str(train_end - train_start) + "," +
                    str(pred_end - pred_start) + "," +
                    str(eval_score_end - eval_score_start) + "\n")
            if score != 0:
                if time_limit_reached:
                    sys.exit()
            else:
                if time_limit_reached:
                    sys.exit()
                else:
                    continue
        population[population.index(bottom)] = (exploit_and_explore_pipe, score)
