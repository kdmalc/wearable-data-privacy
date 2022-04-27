import pandas as pd
import numpy as np
import random
from scipy.special import rel_entr


def run_n_attacks(df, training_df, num_attacks, ths=[0,2,5], k_vec=[1,5,10,20,30], sample_size=2):
    fields = 'FairlyActiveMinutes'
    my_cols = ['Trial', 'Test_IDs', 'Threshold', 'K', 'Precision', 'Recall', 'Accuracy']
    res_df = pd.DataFrame(columns=my_cols)

    all_IDs = df.Id.unique()
    num_users = len(all_IDs)

    user_dict = dict()
    for idx, user in enumerate(all_IDs):
        temp_df = training_df.loc[training_df['Id'] == user]
        user_dict[user] = temp_df[fields].values
    ##########################################################

    # LOOPING THROUGH THE NUMBER OF ATTACKS DESIRED
    for trial_num in range(num_attacks):
        #test_IDs = all_IDs[[0, 1]]
        test_IDs = random.sample(list(all_IDs), 2)
        n_attack_training = len(test_IDs)

        training_IDs = [ID for ID in all_IDs if ID not in test_IDs]
        shuffled_training_IDs = list(random.sample(training_IDs, len(training_IDs)))
        attack_IDs = list(test_IDs) + shuffled_training_IDs[0:n_attack_training]
        #attack_IDs = list(test_IDs) + list(training_IDs[0:2])
        print("SANITY CHECK:")
        for i in attack_IDs:
            print(f"{i} in training_IDs: {i in training_IDs}")

        print("TEST IDs:")
        print()
        print(test_IDs)
        for i in range(len(test_IDs)):
            print("TEST VALS " + str(i+1))
            print(user_dict[test_IDs[i]])
            print()
        ##########################################################
        # Max number of elements for any field
        max_elements = 31

        # Storage Initialization
        data_storage = np.zeros((len(attack_IDs), len(training_IDs), len(ths), max_elements, len(k_vec)))

        n_attack_training = len(attack_IDs) - len(test_IDs)
        true_positives = np.zeros((len(ths), len(k_vec)))
        false_negatives = np.zeros((len(ths), len(k_vec)))
        ##########################################################
        matching_matrix = np.zeros((len(attack_IDs), len(training_IDs), len(ths), len(k_vec))) + 9
        for i, attack_ID in enumerate(attack_IDs):
            v1 = user_dict[attack_ID]
            for k_idx, k in enumerate(k_vec):
                # POSSIBLE ISSUE
                # We shuffled the training IDs... so matching_matrix is thrown off... need to compensate for this somehow...
                for j, training_ID in enumerate(training_IDs): #shuffled_training_IDs[0:n_attack_training] #(training_IDs[:2]):
                    v2 = user_dict[training_ID]

                    for l, th in enumerate(ths):
                        if (isMatch(v1,v2,k,th)):
                            matching_matrix[i,j,l,k_idx] = 1
                        else:
                            matching_matrix[i,j,l,k_idx] = 0
        ##########################################################
        for attack_idx, attack_ID in enumerate(attack_IDs):
            if attack_ID in training_IDs:
                label_idx = training_IDs.index(attack_ID)
                for j in range(len(ths)):
                    for k_idx in range(len(k_vec)):
                        if matching_matrix[attack_idx, label_idx, j, k_idx] == 1:
                            true_positives[j, k_idx] += 1
                        else:
                            false_negatives[j, k_idx] += 1                    
        ##########################################################                 
        for j in range(len(ths)):
            for k_idx in range(len(k_vec)):
                if true_positives[j, k_idx] == n_attack_training:
                    # Note that this is not necessarily mean all of them were true positives
                    print(f"Th={ths[j]}, k={k_vec[k_idx]}: Correct number of positives found ({int(true_positives[j, k_idx])})!")
                else:
                    print(f"Th={ths[j]}, k={k_vec[k_idx]}: Incorrect number of positives... found ({int(true_positives[j, k_idx])}) vs expected ({n_attack_training})")
        ##########################################################            
        for j in range(len(ths)):
            for k_idx in range(len(k_vec)):
                all_positives = np.count_nonzero(matching_matrix[:,:,j,k_idx] == 1) 
                false_positives = all_positives - true_positives[j, k_idx]
                true_negatives = np.product(matching_matrix.shape) - all_positives - false_negatives[j, k_idx]

                precision = true_positives[j, k_idx] / (true_positives[j, k_idx] + false_positives)
                recall = true_positives[j, k_idx] / (true_positives[j, k_idx] + false_negatives[j, k_idx])
                accuracy = (true_positives[j, k_idx] + true_negatives) / np.product(matching_matrix.shape)

                new_row = [trial_num, test_IDs, ths[j], k_vec[k_idx], precision, recall, accuracy]
                new_df = pd.DataFrame([new_row], columns=my_cols)
                res_df = pd.concat([res_df, new_df])

        print("--------------------------------------------")
        print("END OF ATTACK NUMBER " + str(trial_num+1))
        print("--------------------------------------------")
        print()

    return res_df



def isMatch(v1,v2,k,th):
    '''
    Are a match is there is a common k-subsequence that is matched with L1 dist <= th
    '''
    
    # If len(v1)=31 and k=31, then this is range(0,1) ---- k <= len(v1)
    for i in range(0,len(v1)-k+1):
        #print(i)
        a = np.array([v1[i:i+k]])
        for j in range(0,len(v2)-k+1):
            #b = np.array([v2[i:i+k]])  # Original code, different type of match, 
            b = np.array([v2[j:j+k]])  # Not run yet, this uses any sliding window
            
            # Should I zero pad b, truncate a, or just return false?
            while (a.shape[1] > b.shape[1]):
                b = np.append(b, 0)
                b = np.reshape(b, (1, len(b)))
            
            if (np.linalg.norm((a - b), ord=1) <= th):
                return True
            
    return False