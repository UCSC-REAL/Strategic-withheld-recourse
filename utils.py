# files for helper functions

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression as LRG
from sklearn.model_selection import train_test_split
import random

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

#pip install cvxpy
#import cvxpy as cp

from sklearn.metrics import confusion_matrix

import pickle

def get_unique_elements(sublists):
    unique_elements = set()
    for sublist in sublists:
        unique_elements.update(sublist)
    return list(unique_elements) 

 ## Compute the full Cost Matrices for both Recourse and Manipulation (without subsidy)

 # precompute all distances
def get_all_costs(X_neg, X_pos, w_R, w_M):
    def cR(x, z, weight = None, p = 2, e = 0):
        #assert 0 <= subs <= 1
        
        w = np.ones(x.shape[0]) if weight is None else weight
        return (np.mean((w*(x-z))**p)**(1/p) + random.uniform(0, e))

    def cM(x, z, weight = None, p = 2, e = 0):
        #assert 0 <= subs <= 1
        
        w = np.ones(x.shape[0]) if weight is None else weight
        
        return (np.mean((w*(x-z))**p)**(1/p) + random.uniform(0, e))

    
    all_cR = {tuple(x): np.array([cR(x, X_pos[i], w_R, e = 0.) 
                                  for i in range(len(X_pos))]) for x in X_neg}

    all_cM = {tuple(x): np.array([cM(x, X_pos[i], w_M, e = 0.) 
                              for i in range(len(X_pos))]) for x in X_neg}
    
    
    return all_cR, all_cM


## Helper function to compute recourse ratio given any revealed_set

def compute_rec_man_ratio(X_neg, X_pos, revealed_set, input_X_pos_cR, input_X_pos_cM):
    #print(revealed_set)
    rec_cnt = 0
    man_cnt = 0
    for i in range(len(X_neg)):
        x = X_neg[i]
        
        # get the index of the minimum recourse action
        i_R = list(revealed_set)[np.argmin(input_X_pos_cR[tuple(x)][list(revealed_set)])]

        cR_i = input_X_pos_cR[tuple(x)][i_R]

        i_M = list(revealed_set)[np.argmin(input_X_pos_cM[tuple(x)][list(revealed_set)])]

        x_R = X_pos[i_R]
        x_M = X_pos[i_M]

        cM_i = input_X_pos_cM[tuple(x)][i_M]   
        
        if min(cR_i, cM_i) <= 1:
            if cR_i < cM_i:
                rec_cnt += 1
            else:
                man_cnt += 1
                
    recourse_ratio = rec_cnt/len(X_neg)
    manipulation_ratio = man_cnt/len(X_neg)
    
#     print("recourse_ratio", recourse_ratio)
#     print("manipulation_ratio", manipulation_ratio)    
            
    return recourse_ratio, manipulation_ratio



# Probabilistic Disclosure: Submodular Minimization

def submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, S):
    '''
    subset S: is the selected subset of recourse actions
    The function counts the number of Xm(xi, S) that overlaps with a particular set S
    Xm(xi, S) := {z: cM(xi, z) < cR(xi, zR)}
    ''' 
    
    if len(S) == 0:
        return 0
    else:
        total_overlap_count = 0
        # build the dictionary based on the revealed list so far
#         Xm_dict = {}
        for i in range(len(X_neg)):
            x = X_neg[i]         
            Xm_i = set()
            
            i_R = list(S)[np.argmin(input_X_pos_cR[tuple(x)][list(S)])]
            cR_i = input_X_pos_cR[tuple(x)][i_R]
            print(cR_i)


            for j in range(len(X_pos)): 
                if input_X_pos_cM[tuple(x)][j] < cR_i:
                    Xm_i.add(j)

            overlap = bool(S & Xm_i)
            total_overlap_count += overlap
        return total_overlap_count


def probablistic_submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, prob, S):
    '''
    prob: the probability
    subset S: is the selected subset of recourse actions
    The function counts the number of Xm(xi, S) that overlaps with a particular set S
    Xm(xi, S) := {z: cM(xi, z) < cR(xi, zR)}
    ''' 
    
    if len(S) == 0:
        return 0
    else:
        #total_overlap_count = 0
        total_probability_count = 0.0
        # build the dictionary based on the revealed list so far

        for i in range(len(X_neg)):
            x = X_neg[i]         
            Xm_i = set()
            
            i_R = list(S)[np.argmin(input_X_pos_cR[tuple(x)][list(S)])]
            cR_i = input_X_pos_cR[tuple(x)][i_R]


            for j in range(len(X_pos)): 
                if input_X_pos_cM[tuple(x)][j] < cR_i:
                    Xm_i.add(j)

            # compute the overlapping probability
            '''
            p(Xm(xi, S)) = 1 - \Pi_{z\in Xm(xi, S)}(1 - prob)  
            '''
            
            probability_overlap = 1 - (1 - prob)**(len(S.intersection(Xm_i)))
            #print(probability_overlap)
            
            total_probability_count += probability_overlap
            
            # overlap = bool(S & Xm_i)
            # total_overlap_count += overlap

        #return total_overlap_count
        return total_probability_count

# Greedy submodular function minimization

def greedy_submodular_minimization(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, submodular_function, k):
    '''
    Input: 

    submodular_function: defined above
    ground_set: X_pos

    k: the total number of recourse action revealed

    '''
    n = len(X_pos)
    selected_subset = []
    remaining_set = list(i for i in range(0, n))

    for iter in range(k):
        #print(iter)
        
        min_gain = float('inf')
        best_element = None
        
        for element in remaining_set:
            subset_candidate = selected_subset + [element]
            gain = submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, set(subset_candidate)) - submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, set(selected_subset))
            if gain < min_gain:
                min_gain = gain
                best_element = element

        selected_subset.append(best_element)
        remaining_set.remove(best_element)

     #   print(selected_subset)

      #  print("submodular function value", submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, set(selected_subset)))

        
         

    return selected_subset


# reference
def greedy_probablistic_submodular_minimization(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, probablistic_submodular_function, prob, k):
    '''
    Input: 

    submodular_function: defined above
    ground_set: X_pos

    k: the total number of recourse action revealed

    '''
    n = len(X_pos)
    selected_subset = []
    remaining_set = list(i for i in range(0, n))

    for iter in range(k):
        #print(iter)
        
        min_gain = float('inf')
        best_element = None
        
        for element in remaining_set:
            subset_candidate = selected_subset + [element]
            gain = probablistic_submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, prob, set(subset_candidate)) - probablistic_submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, prob, set(selected_subset))
            if gain < min_gain:
                min_gain = gain
                best_element = element

        selected_subset.append(best_element)
        remaining_set.remove(best_element)

     #   print(selected_subset)

      #  print("submodular function value", submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, set(selected_subset)))

        
         

    return selected_subset



def greedy_probablistic_pair_submodular_minimization(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, probablistic_submodular_function, prob, k):
    
    n = len(X_pos)
    selected_subset = []
    remaining_set = list(i for i in range(0, n))
    
    for t in range(k // 2):
        min_gain = float('inf')
        best_pair = None
        
        # Evaluate all pairs for the best decrease in the function value
        for i in range(len(remaining_set)):
            for j in range(i + 1, len(remaining_set)):
                
                current_value = probablistic_submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, prob, set(selected_subset))

                subset_candidate = selected_subset + [remaining_set[i], remaining_set[j]]
                   
                new_value = probablistic_submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, prob, set(subset_candidate))

                gain = new_value - current_value
            #    print("gain", gain)
                
                if gain < min_gain:
                    min_gain = gain
                    best_pair = (remaining_set[i], remaining_set[j])

        # add the two element to the selected_subset and remove them from remaining_set
        element1 = best_pair[0]
        element2 = best_pair[1]
        selected_subset.append(best_pair[0])
        selected_subset.append(best_pair[1])
        remaining_set.remove(best_pair[0])
        remaining_set.remove(best_pair[1])

     #   print(selected_subset)

      #  print("submodular function value", submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, set(selected_subset)))

        
        

    return selected_subset

# greedy probablistic 
def greedy_pair_submodular_minimization(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, submodular_function, k):
    
    n = len(X_pos)
    selected_subset = []
    remaining_set = list(i for i in range(0, n))
    
    for _ in range(k // 2):
        min_gain = float('inf')
        best_pair = None
        
        # Evaluate all pairs for the best decrease in the function value
        for i in range(len(remaining_set)):
            for j in range(i + 1, len(remaining_set)):
                # Access the pair as (remaining_list[i], remaining_list[j])
               # print(remaining_set[i], remaining_set[j])
                
                current_value = submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, set(selected_subset))

                subset_candidate = selected_subset + [remaining_set[i], remaining_set[j]]
                   
                new_value = submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, set(subset_candidate))

                gain = new_value - current_value
            #    print("gain", gain)
                
                if gain < min_gain:
                    min_gain = gain
                    best_pair = (remaining_set[i], remaining_set[j])

        # add the two element to the selected_subset and remove them from remaining_set
        element1 = best_pair[0]
        element2 = best_pair[1]
        selected_subset.append(best_pair[0])
        selected_subset.append(best_pair[1])
        remaining_set.remove(best_pair[0])
        remaining_set.remove(best_pair[1])

     #   print(selected_subset)

      #  print("submodular function value", submodular_function(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM, set(selected_subset)))

        
        

    return selected_subset


# compute the set of all optimal recourse actions given X_pos, and return the numpy array

def X_opt_set_cost(X_neg, X_pos, input_X_pos_cR, input_X_pos_cM):

    set_opt_rec = set()
    set_opt_rec_index = set()

    rec_list_array_dict = {}
    rec_list_index_dict = {}
    
    for i in range(len(X_neg)):
        x = X_neg[i]
        
        # index of best recourse action
        
        # Find the indices of the minimum values
        min_indices = np.where(input_X_pos_cR[tuple(x)] == np.min(input_X_pos_cR[tuple(x)]))[0]
        # Choose a random index from the minimum indices
        i_R = np.random.choice(min_indices)
      
        x_R = X_pos[i_R]

        if tuple(x_R) in rec_list_array_dict.keys():
            rec_list_array_dict[tuple(x_R)].append(i)
        else:
            rec_list_array_dict[tuple(x_R)] = [i]
        
        set_opt_rec_index.add(i_R)   
        set_opt_rec.add(tuple(x_R))

    X_opt = np.array(list(map(list, set_opt_rec)))
  #  print("rec_list_array_dict", rec_list_array_dict)

    
    # generate the sub-dictionary
    
    # Create a new dictionary to store the selected elements
    output_X_opt_cR = {}
    output_X_opt_cM = {}


    # Loop through the dictionary items and select elements based on the indices
    for key, value in input_X_pos_cR.items():
        selected_elements = value[list(set_opt_rec_index)]  # Use NumPy indexing
        output_X_opt_cR[key] = selected_elements
        
    for key, value in input_X_pos_cM.items():
        selected_elements = value[list(set_opt_rec_index)]  # Use NumPy indexing
        output_X_opt_cM[key] = selected_elements

    # map the array_dict to the index_dict

    rec_list_index_dict = {}
    for index, array_x in enumerate(X_opt):
   #     print("index", index)
   #     print("array_x", array_x)
        key = tuple(array_x)  # Convert numpy array to tuple for dictionary key
        if key in rec_list_array_dict:
           # print("key", key)
            rec_list_index_dict[index] = rec_list_array_dict[key]
    
    
    return X_opt, output_X_opt_cR, output_X_opt_cM, rec_list_index_dict




# system utility
def system_utility(revealed_set, X_opt, input_X_pos_cR, input_X_pos_cM, clf, X_neg, y_neg, X_pos, y_pos):
    
    
    # the primary goal is to generate new dataset after the recourse/manipulation interaction
    
    # create the result after recourse/manipulation 
    y_neg_after = np.copy(y_neg)
    X_neg_after = np.copy(X_neg)
    
          
    # compute among X_neg, their changed features and corresponding changed labels 
    for i in range(len(X_neg)):
        x = X_neg[i]    
        
        # Find the indices of the minimum recourse values
        i_R = list(revealed_set)[np.argmin(input_X_pos_cR[tuple(x)][list(revealed_set)])]
        
#         print(i_R)

        cR_i = input_X_pos_cR[tuple(x)][i_R]

        i_M = list(revealed_set)[np.argmin(input_X_pos_cM[tuple(x)][list(revealed_set)])]
       # print("i_M", i_M)

        x_R = X_opt[i_R]
        
        # find the corresponding y_R in X_pos
        # Find the row in X that matches the given x_value(s)
        matching_row_indices = np.where((X_pos == x_R).all(axis=1))[0]

        # Check if a matching row was found
        if matching_row_indices.shape[0] > 0:
            # Use the index of the matching row to find the corresponding y value
            y_R = y_pos[matching_row_indices[0]]
        
        
        x_M = X_opt[i_M]
    

        cM_i = input_X_pos_cM[tuple(x)][i_M]    
       # print("cM_i", cM_i)

        if cR_i < cM_i:
            # agent taking recourse, the new feature should be X_R, true label changes to y_R
            X_neg_after[i] = x_R
            y_neg_after[i] = y_R
            
        else:
            # agent perform manipulation,  the new feature should be X_M, true label is still the same
            X_neg_after[i] = x_M
          #  y_neg_after[i] = y_neg_after[i]
            
    # now compute the TPR, FPR on the new dataset: X_pos, y_pos, X_neg_after, y_neg_after   

    # Concatenate the positive and negative datasets
    X_combined = np.vstack((X_pos, X_neg_after))
    y_combined = np.concatenate((y_pos, y_neg_after))

    # Make predictions using the classifier
    y_pred = clf.predict(X_combined)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_combined, y_pred)

    # Extract the values from the confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    
    #print(tn, fp, fn, tp)
    
    # calculate tp - fp
    
    return tp - fp


def compute_utils(revealed_set, X_opt, input_X_pos_cR, input_X_pos_cM, clf, X_neg, y_neg, X_pos, y_pos, provide_rec):
    '''
    a helper function that takes an input revealed_set, compute the following quantities:
    
    recourse ratio
    manipulation ratio
    system's utility changed (measured by TP - FP afterwards)
    
    '''
    
    # for computing recourse ratio and manipulation ratio
    rec_cnt = 0
    man_cnt = 0
    
    
    # true qualification result after recourse/manipulation 
    y_neg_after = np.copy(y_neg)
    X_neg_after = np.copy(X_neg)
    
    # among X_neg, their changed features and corresponding changed labels 
    
    for i in range(len(X_neg)):
        
        x = X_neg[i]    
        
        # Find the indices of the minimum recourse values (there could be multiple min recourse actions)
        i_R = list(revealed_set)[np.argmin(input_X_pos_cR[tuple(x)][list(revealed_set)])]
        x_R = X_opt[i_R]
        cR_i = input_X_pos_cR[tuple(x)][i_R]
        
        # find the corresponding y_R in X_pos
        # Find the row in X that matches the given x_value(s)
        matching_row_indices = np.where((X_pos == x_R).all(axis=1))[0]

        # Check if a matching row was found
        if matching_row_indices.shape[0] > 0:
            # Use the index of the matching row to find the corresponding y value
            y_R = y_pos[matching_row_indices[0]]
               
        
        i_M = list(revealed_set)[np.argmin(input_X_pos_cM[tuple(x)][list(revealed_set)])]        
        x_M = X_opt[i_M]
        cM_i = input_X_pos_cM[tuple(x)][i_M]    


        if provide_rec[i] == 0:
            if cM_i < 1:
                man_cnt += 1
                # agent performs manipulation, the new feature should be X_M, true label is still the same
                X_neg_after[i] = x_M
                

        else:
            if cR_i < cM_i:
                # recourse count +1
                rec_cnt += 1
                # agent taking recourse, the new feature should be X_R, true label changes to y_R
                X_neg_after[i] = x_R
                y_neg_after[i] = y_R
                
                
            elif cM_i < 1:
                man_cnt += 1
                # agent performs manipulation, the new feature should be X_M, true label is still the same
                X_neg_after[i] = x_M
            
    # now compute the TP, FP on the new dataset: X_pos, y_pos, X_neg_after, y_neg_after   

    # Concatenate the positive and negative datasets
    X_combined = np.vstack((X_pos, X_neg_after))
    y_combined = np.concatenate((y_pos, y_neg_after))

    # Make predictions using the classifier
    y_pred = clf.predict(X_combined)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_combined, y_pred)

    # # Extract the values from the confusion matrix
    # tn, fp, fn, tp = conf_matrix.ravel()
    # Ensure the confusion matrix has the expected shape
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
        #print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
    else:
        # Handle cases where the confusion matrix does not have the expected shape
        #print("Unexpected shape of confusion matrix: ", conf_matrix.shape)
        if conf_matrix.size == 1:
            # Only one element, all predictions are the same class
            tn = fp = fn = tp = 0
            if y_true[0] == y_pred[0] == 0:
                tn = conf_matrix[0, 0]
            elif y_true[0] == y_pred[0] == 1:
                tp = conf_matrix[0, 0]
            #print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
        else:
            # For multi-class or unexpected shape, handle accordingly
            print("Multi-class or unexpected confusion matrix shape.")
    

    #rec_ratio = rec_cnt/len(X_neg)
    rec_ratio = rec_cnt/np.sum(provide_rec)
    man_ratio = man_cnt/len(X_neg)
    
    
    return rec_ratio, man_ratio, tp - fp

# def compute_utils(revealed_set, X_opt, input_X_pos_cR, input_X_pos_cM, clf, X_neg, y_neg, X_pos, y_pos):
#     '''
#     a helper function that takes an input revealed_set, compute the following quantities:
    
#     recourse ratio
#     manipulation ratio
#     system's utility (measured by TP - FP afterwards)
    
#     '''
    
#     # for computing recourse ratio and manipulation ratio
#     rec_cnt = 0
#     man_cnt = 0
    
    
#     # true qualification result after recourse/manipulation 
#     y_neg_after = np.copy(y_neg)
#     X_neg_after = np.copy(X_neg)
    
#     # among X_neg, their changed features and corresponding changed labels 
    
#     for i in range(len(X_neg)):
        
#         x = X_neg[i]    
        
#         # Find the indices of the minimum recourse values (there could be multiple min recourse actions)
#         i_R = list(revealed_set)[np.argmin(input_X_pos_cR[tuple(x)][list(revealed_set)])]
#         x_R = X_opt[i_R]
#         cR_i = input_X_pos_cR[tuple(x)][i_R]
        
#         # find the corresponding y_R in X_pos
#         # Find the row in X that matches the given x_value(s)
#         matching_row_indices = np.where((X_pos == x_R).all(axis=1))[0]

#         # Check if a matching row was found
#         if matching_row_indices.shape[0] > 0:
#             # Use the index of the matching row to find the corresponding y value
#             y_R = y_pos[matching_row_indices[0]]
               
        
#         i_M = list(revealed_set)[np.argmin(input_X_pos_cM[tuple(x)][list(revealed_set)])]        
#         x_M = X_opt[i_M]
#         cM_i = input_X_pos_cM[tuple(x)][i_M]    


#         if cR_i < cM_i:
#             # recourse count +1
#             rec_cnt += 1
#             # agent taking recourse, the new feature should be X_R, true label changes to y_R
#             X_neg_after[i] = x_R
#             y_neg_after[i] = y_R
            
            
#         else:
#             man_cnt += 1
#             # agent performs manipulation, the new feature should be X_M, true label is still the same
#             X_neg_after[i] = x_M
            
#     # now compute the TP, FP on the new dataset: X_pos, y_pos, X_neg_after, y_neg_after   

#     # Concatenate the positive and negative datasets
#     X_combined = np.vstack((X_pos, X_neg_after))
#     y_combined = np.concatenate((y_pos, y_neg_after))

#     # Make predictions using the classifier
#     y_pred = clf.predict(X_combined)

#     # Calculate the confusion matrix
#     conf_matrix = confusion_matrix(y_combined, y_pred)

#     # Extract the values from the confusion matrix
#     tn, fp, fn, tp = conf_matrix.ravel()
    
#     rec_ratio = rec_cnt/len(X_neg)
#     man_ratio = man_cnt/len(X_neg)
    
    
#     return rec_ratio, man_ratio, tp - fp
    
    
def social_cost(revealed_set, input_X_pos_cR, input_X_pos_cM, X_neg):

    cost_normal_sum = 0
    cost_mp_sum = 0
    
    # with strategic system, number of people taking recourse vs manipulation

    rec_cnt_mp = 0
    man_cnt_mp = 0


    for i in range(len(X_neg)):
        x = X_neg[i]

        # compute restricted recourse action and cost
        
        i_R_mp = list(revealed_set)[np.argmin(input_X_pos_cR[tuple(x)][list(revealed_set)])]
        cR_i_mp = input_X_pos_cR[tuple(x)][i_R_mp] 

        # compute the restricted manipulation action and cost
        i_M_mp = list(revealed_set)[np.argmin(input_X_pos_cM[tuple(x)][list(revealed_set)])]
        cM_i_mp = input_X_pos_cM[tuple(x)][i_M_mp] 

        if cR_i_mp < cM_i_mp:
            rec_cnt_mp += 1

            
            # compute normal recourse action and cost
            i_R_normal = np.argmin(input_X_pos_cR[tuple(x)])
            cR_i_normal = input_X_pos_cR[tuple(x)][i_R_normal]

            # print("cR_i_normal", cR_i_normal)
            # print("cR_i_mp", cR_i_mp)

            cost_normal_sum += cR_i_normal
            cost_mp_sum += cR_i_mp

        else:     
            man_cnt_mp += 1
            
    if rec_cnt_mp == 0:
        return 0
    
    
    else:
        # the average cost differences for agents who get recourse in the manipulation proof case
        # print(rec_cnt_mp)
        average_cost_normal = cost_normal_sum/rec_cnt_mp

        # the recourse/manipulation ratio
        average_cost_mp =  cost_mp_sum/rec_cnt_mp

        return average_cost_mp - average_cost_normal 


# # save the dictionary
# def save_dictionaries_as_pickle(dicts, dataset_name):
#     for dict_name, dict_data in dicts.items():
#         filename = f"{dataset_name}_logistic_regression_{dict_name}.pkl"
#         with open(filename, 'wb') as file:
#             pickle.dump(dict_data, file)
#         print(f"Saved: {filename}")



# this is for plotting the 6 quantities that we care about

def calculate_mean_std(data):
    # Determine the longest run
    max_length = max(len(value) for value in data.values())

    # Prepare structure for interpolated values
    interpolated_data = {}
    for key, values in data.items():
        iteration, sub = key
        if sub not in interpolated_data:
            interpolated_data[sub] = []

        # Extract x and y values
        x_values, y_values = zip(*values)

        # Interpolate y values
        f = interp1d(x_values, y_values, kind='linear', fill_value="extrapolate")
        common_x = np.linspace(min(x_values), max(x_values), max_length)
        interpolated_y = f(common_x)

        interpolated_data[sub].append((common_x, interpolated_y))
   # print(interpolated_data)

    # Calculate mean and std dev
    means = {}
    std_devs = {}
    for sub, xy_values in interpolated_data.items():
       # print("xy_values", xy_values)
        all_x, all_y = zip(*xy_values)
        #print("all_y", all_y)
        mean_x = np.mean(all_x, axis=0)
        # print("mean_x", mean_x)
        mean_y = np.mean(all_y, axis=0)
        # print("mean_y", mean_y)
        std_dev_y = np.std(all_y, axis=0)/ np.sqrt(len(mean_x))

        means[sub] = np.array(list(zip(mean_x, mean_y)))
        std_devs[sub] = std_dev_y

    return means, std_devs







def calculate_mean_std_recourse_rate_diff(data):
    # Extract unique subs and initialize data structure
    unique_subs = sorted({key[1] for key in data.keys()})
    num_percentages = len(data[next(iter(data))])

    # Initialize structure for means and std devs
    means = [[] for _ in range(num_percentages)]
    std_devs = [[] for _ in range(num_percentages)]

    # Organize data by subs
    organized_data = {sub: [] for sub in unique_subs}
    for (iteration, sub), values in data.items():
        organized_data[sub].append(values)

    # Calculate mean and std dev
    for sub_values in organized_data.values():
        arr = np.array(sub_values)
        for i in range(num_percentages):
            means[i].append(np.mean(arr[:, i]))
            std_devs[i].append(np.std(arr[:, i]))

    return means, std_devs

