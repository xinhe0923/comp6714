## import modules here
import pandas as pd
import math as mt

################# Question 1 #################

def multinomial_nb(training_data, sms):# do not change the heading of the function
    category_dict = {}
    category_prob = {}
    word_list = []
    num_category = 0
    # if two categories are the same, their value should be added up
    for entry in training_data:
        category_name = entry[1]
        category_data = entry[0]
        num_category += 1

        if category_name not in category_dict:
            category_dict[category_name] = category_data
            # update category_prob
            category_prob[category_name] = 1
            # add words to word_list
            for word in category_data.keys():
                if word not in word_list:
                    word_list.append(word)
        else:
            # updata category_prob
            category_prob[category_name] += 1
            # merge two entries
            for key, value in category_data.items():
                if key in category_dict[category_name].keys():
                    category_dict[category_name][key] += value
                else:
                    category_dict[category_name][key] = value
                    # add words to word_list
                    if key not in word_list:
                        word_list.append(key)

    # number of words
    total_words = len(word_list)

    # step 2 training
    # P(ck)
    for key, value in category_prob.items():
    # print(key, value)
    # P(ck)
    for key, value in category_prob.items():
        # print(key, value)
        category_prob[key] = value / num_category

    trained_dict = {}
    for category_name, category_data in category_dict.items():
        # for each category, calculate the total number of all words
        num_words = 0
        for value in category_data.values():
            num_words += value

        category_trained = {}
        # P(tj|ck)
        for word in word_list:
            category_trained[word] = (category_data.get(word, 0) + 1) / (num_words + total_words)

        trained_dict[category_name] = category_trained

        # test all data and print them out if needed here
        #    print(category_dict)
    print(trained_dict)
    #    print(category_prob)

    # ----------------------------------------------------------
    # step 3 classifying
    prob_dict = {}
    # initialize probability log(P(ck))
    for category_name in trained_dict:
        prob_dict[category_name] = mt.log(category_prob[category_name])
        print(category_name, category_prob[category_name], prob_dict[category_name])
    for word in sms:
        if word in word_list:
            # sum log(P(tj|ck))
            for category_name, category_data in trained_dict.items():
                print(word, category_name, category_data[word], mt.log(category_data[word]))
                prob_dict[category_name] += mt.log(category_data[word])
    print('spam', prob_dict['spam'], mt.pow(mt.e, prob_dict['spam']))
    print('ham', prob_dict['ham'], mt.pow(mt.e, prob_dict['ham']))
    return (mt.pow(mt.e, prob_dict['spam']) / mt.pow(mt.e, prob_dict['ham']))