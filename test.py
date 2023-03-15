def function():
    file = open('twitter_train.txt')
    #f = open(in_output_probs_filename, "w")
    pair = file.readline()
    token_tag_counter_dict = {}
    tag_counter_dict = {}
    while pair:
        newpair = pair.split()
        token = newpair[0]
        tag = newpair[1]
        if token in token_tag_counter_dict.keys():
            if tag in token_tag_counter_dict[token].keys():
                count = token_tag_counter_dict[token][tag][0]
                token_tag_counter_dict[token][tag] = (count + 1, tag)
        else:
            token_tag_counter_dict[token][tag] = (1, tag)
        if tag in tag_counter_dict.keys():
            count = tag_counter_dict[tag][0]
            tag_counter_dict[tag] = (count + 1, tag)
        else:
            tag_counter_dict[tag] = (1, tag)
        pair = file.readline()
        if pair == '\n':
            pair = file.readline()
    
    prob_file = open(in_output_probs_filename, "w")
    for (key, value) in token_tag_counter_dict:
        max_pair = max(token_tag_counter_dict[key].values())
        tag = max_pair[1]
        prob = max_pair[0]/tag_counter_dict[tag]
    prob_file.write(key, "\t", tag, "\t", prob, "\n")




def naive_predict():
    # naive_predict(naive_output_probs.txt, twitter_dev_no_tag.txt, naive_predictions.txt)
    file = open('twitter_train.txt')
    #f = open(in_output_probs_filename, "w")
    pair = file.readline()
    token_tag_counter_dict = {}
    tag_counter_dict = {}
    while pair:
        newpair = pair.split()
        token = newpair[0]
        tag = newpair[1]
        if token in token_tag_counter_dict.keys():
            if tag in token_tag_counter_dict[token].keys():
                count = token_tag_counter_dict[token][tag][0]
                token_tag_counter_dict[token][tag] = (count + 1, tag)
            else:
                token_tag_counter_dict[token][tag] = (1, tag)
        else:
            token_tag_counter_dict[token] = {}
            token_tag_counter_dict[token][tag] = (1, tag)
        if tag in tag_counter_dict.keys():
            count = tag_counter_dict[tag][0]
            tag_counter_dict[tag] = (count + 1, tag)
        else:
            tag_counter_dict[tag] = (1, tag)
        pair = file.readline()
        if pair == '\n':
            pair = file.readline()
    '''
    prob_file = open(in_output_probs_filename, "a")
    for (key, value) in token_tag_counter_dict:
        max_pair = max(token_tag_counter_dict[key].values())
        tag = max_pair[1]
        prob = max_pair[0]/tag_counter_dict[tag][0]
        prob_file.write(f'{key}\t{tag}\t{prob}\n')
    prob_file.close()


    input_file = open(in_test_filename)
    input = input_file.readline()

    prediction_file = open(out_prediction_filename, "w")
    
'''
    return token_tag_counter_dict
    

naive_predict()
