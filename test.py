'''
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




file = open("naive_output_probs.txt", "x")
file = open("naive_predictions.txt", "x")
# Implement the six functions below
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
    
    prob_file = open("naive_output_probs.txt", "a")
    items = token_tag_counter_dict.items()
    lst = []
    for (key,value) in items:
        max_pair = max(token_tag_counter_dict[key].values())
        tag = max_pair[1]
        prob = max_pair[0]/tag_counter_dict[tag][0]
        if f'{key}\t{tag}\t{prob}\n' not in lst:
            lst.append(f'{key}\t{tag}\t{prob}\n')
    for elem in lst:
        prob_file.write(elem)
    prob_file.close()

    input_file = open('twitter_dev_no_tag.txt')
    num_unique_words = len(lst)
    prob_unknown_word = 0.01 / (0.01 * num_unique_words + 1)
    input = input_file.readline()
    prediction_file = open("naive_predictions.txt", "a")

    #print(token_tag_counter_dict.keys())
    print(input)
    while input:
        if input in token_tag_counter_dict.keys():
            max_pair = max(token_tag_counter_dict[input.strip()].values())
            tag = max_pair[1]
            print("Hello")
        #else:
        #    tag = '@'
        prediction_file.write(f'{tag}\n')
        input = input_file.readline()
        if (input == "" or input == "\n"):
            input = input_file.readline()
    prediction_file.close()
    input_file.close()
    

naive_predict()
'''
def count_lines(filename):
    file = open(filename)
    line = file.readline().strip()
    counter = 0
    while line:
        counter = counter + 1
        line = file.readline().strip()
        if line == '':
            line = file.readline().strip()
    return counter

print(count_lines('twitter_train.txt'))
        
