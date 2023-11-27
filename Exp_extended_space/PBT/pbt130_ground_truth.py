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


import numpy as np
import argparse
import time
import csv
from sklearn.metrics import f1_score
from sklearn.preprocessing import Binarizer, StandardScaler, Normalizer, MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import sys


def generate_dic_and_names():
    temp_preprocessors_dic = {#"isolation_forest": IsolationForest(random_state=0),
                     #"lof": LocalOutlierFactor(),
                     "binarizer": Binarizer(),
                     "standardizer": StandardScaler(),
                     "normalizer": Normalizer(),
                     "maxabs": MaxAbsScaler(),
                     "minmax": MinMaxScaler(),
                     "power_trans": PowerTransformer(),
                     "quantile_trans": QuantileTransformer(random_state=0)}
    temp_operator_names = ["binarizer", "standardizer", "normalizer", "maxabs", "minmax", "power_trans", "quantile_trans"]
    
    #binarizer_threshold = np.arange(0, 1.01, 0.05).tolist()
    binarizer_threshold = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    normalizer_norm = ['l1', 'l2', 'max']
    standard_with_mean = [True, False]
    power_standardize = [True, False]
    #quantile_n_quantiles = np.arange(10, 2001, 1).tolist()
    quantile_n_quantiles = [10, 100, 200, 500, 1000, 1200, 1500, 2000]
    quantile_distribution = ['uniform', 'normal']
    
    for item in binarizer_threshold:
        temp_operator_names.append(f"binarizer_threshold_{str(item)}")
        temp_preprocessors_dic[f"binarizer_threshold_{str(item)}"] = Binarizer(threshold=item)
        
    for item in normalizer_norm:
        temp_operator_names.append(f"normalizer_norm_{item}")
        temp_preprocessors_dic[f"normalizer_norm_{item}"] = Normalizer(norm=item)
    
    for item in standard_with_mean:
        temp_operator_names.append(f"standard_with_mean_{str(item)}")
        temp_preprocessors_dic[f"standard_with_mean_{str(item)}"] = StandardScaler(with_mean=item)
        
    for item in power_standardize:
        temp_operator_names.append(f"power_standardize_{str(item)}")
        temp_preprocessors_dic[f"power_standardize_{str(item)}"] = PowerTransformer(standardize=item)
        
    for item in quantile_n_quantiles:
        temp_operator_names.append(f"quantile_n_quantiles_{str(item)}")
        temp_preprocessors_dic[f"quantile_n_quantiles_{str(item)}"] = QuantileTransformer(n_quantiles=item, random_state=0)
    
    for item in quantile_distribution:
        temp_operator_names.append(f"quantile_distribution_{str(item)}")
        temp_preprocessors_dic[f"quantile_distribution_{str(item)}"] = QuantileTransformer(output_distribution=item, random_state=0)
        
    return temp_preprocessors_dic, temp_operator_names

preprocessors_dic, operator_names = generate_dic_and_names()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="name of dataset")
ap.add_argument("-c", "--classifier", required=True,help="name of classifier")
ap.add_argument("-max_time", "--max_time_limit", required=True,help="number of max_time_limit")
args = vars(ap.parse_args())

dataset = args['dataset']
classifier = args['classifier']
max_time_limit = int(args['max_time_limit'])

# [0, 42, 167,578,1440]
seed=0
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

    result_pipe = perturbation(top_pipe, resample_probability,
                               resample_seed, mutate_way_seed, op_seed, pos_seed)
    return result_pipe

def editdistance(s1, s2):
    if len(s1) < len(s2):
        return editdistance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]



all_pipelines = []
initial_pipelines = []

# Begin PBT code
population_size = 100
population = []
fraction = 0.2
resample_probability = 0.25
epochs = 1000000
max_len = 7

global_start = time.time()
time_limit_reached = False




with open('initial_pipelines.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Pipeline', 'F1 Score', 'Edit Distance with Best'])

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

            y_pred = model.predict(np.array(X_valid_new))
            score = f1_score(y_valid, y_pred, average='macro')


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
        
        population.append((temp_pipe, score))

        # Write initial pipeline and score to CSV (Edit Distance will be updated later)
        csv_writer.writerow([pipe_str, score, ''])

        if (score != 0 and time_limit_reached) or time_limit_reached:
            break
      
      



# Find the best pipeline and its F1 score
best_pipeline_info = max(population, key=lambda x: x[1])
best_pipeline = best_pipeline_info[0]
best_f1_score = best_pipeline_info[1]

# # Open the CSV file for writing
# with open('initial_pipelines.csv', 'w', newline='') as file:
#     csv_writer = csv.writer(file)
#     csv_writer.writerow(['Pipeline', 'F1 Score', 'Edit Distance with Best', 'Best Pipeline', 'Best F1 Score'])

#     # Write each pipeline, its F1 score, edit distance, best pipeline, and best F1 score
#     for pipe, score in population:
#         pipeline_str = ','.join(pipe)
#         distance = editdistance(pipeline_str.split(','), best_pipeline)
#         best_pipeline_str = ','.join(best_pipeline)  # Convert best pipeline list to string
#         csv_writer.writerow([pipeline_str, score, distance, best_pipeline_str, best_f1_score])


# for pipe, score in population:
#     all_pipelines.append((pipe, score))

for pipe, score in population:
    initial_pipelines.append((pipe, score))  # Append each evaluated pipeline to initial_pipelines
    all_pipelines.append((pipe, score))      # Also append to all_pipelines


# Record initial pipelines in initial_pipelines.csv
with open('initial_pipelines.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Pipeline', 'F1 Score', 'Edit Distance with Best'])
    for pipe, score in initial_pipelines:
        pipeline_str = ','.join(pipe)
        csv_writer.writerow([pipeline_str, score, ''])  # Leave 'Edit Distance with Best' empty for now


#second part



# Using PBT framework to do pipeline searching
current_epoch = 0
count = -1
time_limit_reached = False

with open('pbt_pipelines.csv', 'w', newline='') as pbt_file:
    pbt_csv_writer = csv.writer(pbt_file)
    pbt_csv_writer.writerow(['Pipeline', 'F1 Score', 'Edit Distance with Best'])
    while current_epoch < epochs and not time_limit_reached:
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
                #print("Generate pipe time: " + str(generate_pipe_end - generate_pipe_start))

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
                if (time.time() - global_start) >= max_time_limit:
                    time_limit_reached = True
                    break  # Break out of the inner loop

                if score != 0:
                    
                    all_pipelines.append((exploit_and_explore_pipe, score))
                    population[population.index(bottom)] = (exploit_and_explore_pipe, score)
                if time_limit_reached:
                  break  # Break out of the outer loop if time limit is reached




best_pipeline_info = max(all_pipelines, key=lambda x: x[1])
best_pipeline = best_pipeline_info[0]
best_f1_score = best_pipeline_info[1]

best_pipeline_str = ','.join(best_pipeline)

with open('pbt_pipelines.csv', 'w', newline='') as file:
    pbt_csv_writer = csv.writer(file)
    pbt_csv_writer.writerow(['Pipeline', 'F1 Score', 'Edit Distance with Best', 'Best Pipeline', 'Best F1 Score'])
    for pipe, score in all_pipelines:
        pipeline_str = ','.join(pipe)
        distance = editdistance(pipeline_str.split(','), best_pipeline)
        best_pipeline_str = ','.join(best_pipeline)
        pbt_csv_writer.writerow([pipeline_str, score, distance, best_pipeline_str, best_f1_score])




with open('initial_pipelines.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    # Write the headers
    csv_writer.writerow(['Pipeline', 'F1 Score', 'Edit Distance with Best', 'Best Pipeline', 'Best F1 Score'])
    
    # Iterate over each pipeline and write its details
    for pipe, score in initial_pipelines:
        pipeline_str = ','.join(pipe)
        distance = editdistance(pipeline_str.split(','), best_pipeline)
        # Write the complete row
        csv_writer.writerow([pipeline_str, score, distance, best_pipeline_str, best_f1_score])
