# GROUP: BUZZIN'
# Kelly Ng Kaiqing (A0240120H), 
# Lim Xiang Ling (A0238445Y), 
# Ng Yi Wei (A0238253E), 
# Tan Sin Ler (A0240651N)

# Implement the six functions below

##################### HELPER FUNCTIONS #######################

# placing possible tags into a list
def tags(file):
    tags_list = []
    with open(file) as f:
        for tag in f:
            tags_list.append(tag.strip())
    return tags_list

# tokens and their counts and returns a dictionary as follows:
# {token = x1: count of x1, token = x2: count of x2, ...}
def count_words(in_train_filename):
    freqs = {}
    with open(in_train_filename) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                token = temp_list[0]
                if token in freqs:
                    freqs[token] += 1
                else:
                    freqs[token] = 1
    return freqs

# tags and their counts and returns a dictionary as follows:
# {tag = y1: count of y1, tag = y2: count of y2, ...}
def count_tags(in_train_filename):
    freqs = {}
    with open(in_train_filename) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                tag = temp_list[1]
                if tag in freqs:
                    freqs[tag] += 1
                else:
                    freqs[tag] = 1
    return freqs

# returns a dictionary of dictionaries where count the number of token w associated with j 
# e.g. {tag = y1:{token = x1: count, token = x2: count}, tag = y2:{token = x1: count, token = x2: count}}
def count_tokens_tags(in_train_filename):
    freqs = {}
    with open(in_train_filename) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                token = temp_list[0]
                tag = temp_list[1]

                if tag in freqs:
                    tag_tokens = freqs[tag]
                    if token in tag_tokens:
                        tag_tokens[token] += 1
                    else:
                        tag_tokens[token] = 1
                else:
                    freqs[tag] = {}
                    tag_tokens = freqs[tag]
                    tag_tokens[token] = 1
    return freqs

# counting the number of unique words
def num_words(in_train_filename):
    freqs = {}
    with open(in_train_filename) as f:
        #loop through every tag
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                token = temp_list[0]
                if token not in freqs:
                    freqs[token] = 1
    return sum(freqs.values())

# calculates naive output probabilities
def calc_output_prob(in_train_filename):
    output_probabilities = {}
    DELTA = 0.1
    words = num_words(in_train_filename)
    tags_dict = count_tags(in_train_filename)
    tags_tokens_dict = count_tokens_tags(in_train_filename)
    tags_list = tags("twitter_tags.txt")

    output_probabilities["unseen_token_null"] = {}
    for tag in tags_list:
        tag_count = tags_dict[tag]
        num = DELTA
        den = tag_count + DELTA * (words + 1)
        output_probabilities["unseen_token_null"][tag] = num/den
    
    for tag, tags_count in tags_dict.items():
        tags_tokens = tags_tokens_dict[tag]
        for token, tokens_count in tags_tokens.items():
            numerator = tokens_count + DELTA
            denominator = tags_count + DELTA * (words + 1)

            if token in output_probabilities:
                token_prob_tag = output_probabilities[token]
                token_prob_tag[tag] = numerator/denominator
            else:
                output_probabilities[token] = {}
                token_prob_tag = output_probabilities[token]
                token_prob_tag[tag] = numerator/denominator
    return output_probabilities

# takes in the naive_output_probs txt file and output a dictionary
# in the following form: 
# {token=x1:{tag=y1:ouput_probability, tag=y2:ouput_probability}, token=x2:{tag=y1:ouput_probability, tag=y2:ouput_probability}}
def convert_to_dict(in_output_probs_filename):
    output_probabilities = {}
    with open(in_output_probs_filename) as probs_file:
        for line in probs_file:
            l = line.strip().split()
            if l:
                tag = l[0]
                token = l[1]
                prob = float(l[2])
                if token in output_probabilities:
                    token_tags = output_probabilities[token]
                    token_tags[tag] = prob
                else:
                    output_probabilities[token] = {}
                    token_tags = output_probabilities[token]
                    token_tags[tag] = prob
    return output_probabilities

################################### QUESTION 2A #####################################
# For this question, we chose a DELTA value of 0.1.
output_probabilities = calc_output_prob("twitter_train.txt")
with open('naive_output_probs.txt', 'w') as f:
    for token, tags_prob in output_probabilities.items():
        for tag, prob in tags_prob.items():
            f.write("{} \t {} \t {} \n ".format(tag, token, prob))
                    
################################### QUESTION 2B #####################################
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    naive_prediction = []
    naive_output_dict = convert_to_dict(in_output_probs_filename)
    with open(in_test_filename) as test_file:
        for word in test_file:
            token = word.strip()
            if (token):
                if token in naive_output_dict:
                    token_tags = naive_output_dict[token]
                    naive_prediction.append(max(token_tags, key=token_tags.get))       
                else:
                    unseen_token = naive_output_dict["unseen_token_null"]
                    naive_prediction.append(max(unseen_token, key=unseen_token.get))

    with open(out_prediction_filename, "w") as f:
        for prediction in naive_prediction:
            f.write(prediction)
            f.write('\n')

################################### QUESTION 2C #####################################
# The accuracy of our prediction is 65.3% (3 s.f.), where the number of correctly 
# predicted tags / number of predictions is 900/1378.


################################### QUESTION 3A #####################################
# Using Bayes' Theorem, P(y = j|x = w) = [P(x = w|y = j) * P(y = j)] / P(x = w) 
#                                      = [bj(w) * P(y = j)] / P(x = w).
# and since P(x = w) is a constant we can simply find the argmax of bj(w) * P(y = j).
# We can make use of bj(w) from question 2.

################################### QUESTION 3B #####################################
def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    naive_prediction = []
    tag_count_dict = count_tags(in_train_filename)
    naive_output_dict = convert_to_dict(in_output_probs_filename)
    total_words = sum(tag_count_dict.values())

    for token in naive_output_dict:
        token_tags = naive_output_dict[token]
        for tag, output_prob in token_tags.items():
            token_tags[tag] = output_prob * (tag_count_dict[tag]/total_words)

    with open(in_test_filename) as test_file:
        for word in test_file:
            token = word.strip()
            if (token):
                if token in naive_output_dict:
                    token_tags = naive_output_dict[token]
                    naive_prediction.append(max(token_tags, key=token_tags.get))
                else:
                    unseen_token = naive_output_dict["unseen_token_null"]
                    naive_prediction.append(max(unseen_token, key=unseen_token.get)) 
   
    with open(out_prediction_filename, "w") as f:
        for prediction in naive_prediction:
            f.write(prediction)
            f.write('\n')                         

################################### QUESTION 3C #####################################
# The accuracy of our prediction is 69.3% (3 s.f.), where the number of correctly 
# predicted tags / number of predictions is 955/1378.


# creating a dictionary to keep count of transitions
def count_transition_tags(train_filename):
    f = open(train_filename, "r")
    tag_list = tags("twitter_tags.txt") 
    transition_dict = {}
    transition_dict["START"] = {}
    for tag1 in tag_list:
        transition_dict["START"][tag1] = 0
        transition_dict[tag1] = {}
        transition_dict[tag1]["END"] = 0
    line = f.readline().strip().split()
    next_line = f.readline().strip().split()
    counter = 0
    transition_dict["START"][line[1]] += 1
    while next_line:
        tag = line[1]
        next_tag = next_line[1]
        
        if (tag in transition_dict.keys()):
            if (next_line == []):
                next_tag = "END"
            if (next_tag in transition_dict[tag].keys()):
                transition_dict[tag][next_tag] += 1
            else:
                transition_dict[tag][next_tag] = 1
        else:
            transition_dict[tag] = {}
            transition_dict[tag][next_tag] = 1
        line = next_line
        next_line = f.readline().strip().split()
        if (next_line == []):
            # need to indicate that the tweet ended
            # transition_dict[line[1]]["END"] += 1
            next_line = f.readline().strip().split()
            # and also need to indicate that another tweet has started
            if (next_line != []):
                transition_dict["START"][next_line[1]] += 1
    # counter: 16682
    # sum of values in dictionary: 17783
    return transition_dict

# print(count_transition_tags("twitter_train.txt"))

# to count tags, count_tags(in_train_filename)

def count_tags_with_start(in_train_filename):
    tag_dict = {}
    f = open(in_train_filename, "r")
    tag_list = tags("twitter_tags.txt")
    # print(tag_list)
    tag_dict["START"] = 1
    line = f.readline().strip().split()
    while line:
        tag = line[1]
        if tag in tag_dict.keys():
            tag_dict[line[1]] += 1
        else:
            tag_dict[line[1]] = 1
        line = f.readline().strip().split()
        if (line == []):
            tag_dict["START"] += 1
            line = f.readline().strip().split()
    # print(sum(tag_dict.values()))
    tag_dict["START"] -= 1
    # print(f'sum from start: {sum(tag_dict.values())}')
    return tag_dict

# print(count_tags_with_start("twitter_train.txt"))

# function to calculate transition prob
def transition_prob(train_filename):
    DELTA = 0.0001
    transition_dict = count_transition_tags(train_filename)
    tag_dict = count_tags(train_filename)
    tag_dict_start = count_tags_with_start(train_filename)
    num_words = sum(tag_dict_start.values())
    # num_words = sum(tag_dict.values())
    # print(num_words)
    # to confirm!! what is num_words??
    f = open("trans_probs.txt", "w")
    transition_dict_prob = {}
    transition_dict_count = {}
    # print(transition_dict)
    for (key, value) in transition_dict.items(): # key is initial tag and value is the dict of later tag and the count
        for (k, v) in value.items(): # k is later tag and v is count
            count = v
            #print(f'[{key}:{k} count: {count}]')
            # prob = (count) / tag_dict_start[key]
            prob = (count + DELTA) / (tag_dict_start[key] + DELTA * (num_words + 1))
            if (key in transition_dict_prob.keys()):
                transition_dict_prob[key][k] = prob
                transition_dict_count[key][k] = count
            else:
                transition_dict_prob[key] = {}
                transition_dict_prob[key][k] = prob
                transition_dict_count[key] = {}
                transition_dict_count[key][k] = count
            f.write(f'{key}\t\t{k}\t\t{prob}\n')
    # unknown_dict = {}
    for key1 in tag_dict:
        for key2 in tag_dict:
            if key2 not in transition_dict_prob[key1].keys():
                prob = (0 + DELTA) / (tag_dict_start[key1] + DELTA * (num_words + 1))
                if (key1 in transition_dict_prob.keys()):
                    transition_dict_prob[key1][key2] = prob
                else:
                    transition_dict_prob[key1] = {}
                    transition_dict_prob[key1][key2] = prob
                f.write(f'{key1}\t\t{key2}\t\t{prob}\n')
    # for (key, value) in tag_dict
    #print(f'tag_dict: {tag_dict}')
    sum_prob = {}
    sum_count = {}
    for key in transition_dict_prob.keys():
        sum_prob[key] = sum(transition_dict_prob[key].values())
        sum_count[key] = sum(transition_dict_count[key].values())
    print(sum_prob)
    #print(sum_count)
    #print(f'sum from transition: {sum(sum_count.values())}')
    return transition_dict_prob

#print(transition_prob("twitter_train.txt"))

def calculate_sum():
    f = open("trans_probs.txt", "r")
    line = f.readline().strip().split()
    sum_dict = {}
    tag_list = tags("twitter_tags.txt")
    sum_dict["START"] = 0
    for tag in tag_list:
        sum_dict[tag] = 0
    while line:
        tag1 = line[0]
        prob = float(line[2])
        # print(prob)
        if (tag1 in sum_dict.keys()):
            sum_dict[tag1] += prob
        else:
            sum_dict[tag1] = prob
        line = f.readline().strip().split()

    return sum_dict

# print(calculate_sum())

def output_prob(train_filename):
    output_probabilities = calc_output_prob(train_filename)
    DELTA = 0.1
    f = open("output_probs.txt", "w")
    tag_dict = count_tags(train_filename)
    num_words = sum(tag_dict.values())
    for token, tags_prob in output_probabilities.items():
        for tag, prob in tags_prob.items():
            f.write("{} \t {} \t {} \n ".format(tag, token, prob))
    return output_probabilities

# print(transition_prob("twitter_train.txt"))
# print(output_prob("twitter_train.txt"))

def viterbi(observations, tag_list, transition_dict_prob, output_prob):
    V = [{}]
    # output_prob
    # {word1: {tag1: prob1, tag2: prob}, word2: {tag1: prob1, ...}}
    # transition_dict_prob
    # {tag1: {tag1: prob1, tag2: prob2, ...}, tag2: {tag1: prob1, tag2: prob2, ...}}
    counter = 0
    for tag in tag_list:
        if (observations[counter] in output_prob.keys() and tag in output_prob[observations[counter]].keys()):
            V[counter][tag] = {"prob": transition_dict_prob["START"][tag] * output_prob[observations[counter]][tag], "prev": None}
        else:
            V[counter][tag] = {"prob": transition_dict_prob["START"][tag] * output_prob["unseen_token_null"][tag], "prev": None}
    counter += 1
    for i in range(1, len(observations)):
        V.append({})
        for tag in tag_list:
            max_trans_prob = V[counter - 1][tag_list[0]]["prob"] * transition_dict_prob[tag_list[0]][tag]
            prev_tag_selected = tag_list[0]
            for prev_tag in tag_list[1:]:
                tr_prob = V[counter - 1][prev_tag]["prob"] * transition_dict_prob[prev_tag][tag]
                if tr_prob > max_trans_prob:
                    max_trans_prob = tr_prob
                    prev_tag_selected = prev_tag
            if (observations[counter] in output_prob.keys() and tag in output_prob[observations[counter]].keys()):
                max_prob = max_trans_prob * output_prob[observations[counter]][tag]
            else:
                max_prob = max_trans_prob * output_prob["unseen_token_null"][tag]
            V[counter][tag] = {"prob": max_prob, "prev": prev_tag_selected}
        counter += 1
    
    opt = []
    max_prob = 0.0
    best_tag = None

    for tag, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_tag = tag
    opt.append(best_tag)
    previous = best_tag

    for t in range(len(V) - 2, -1 , -1):
        opt.insert(0, V[t+1][previous]["prev"])
        previous = V[t+1][previous]["prev"]

    return opt

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    # viterbi_predict("twitter_tags.txt", "trans_probs.txt", "output_probs.txt", "twitter_dev_no_tag.txt", 
    #                 "viterbi_predictions.txt")
    # observations are the words, while hidden states are the tags
    train_filename = "twitter_train.txt"
    tag_list = tags(in_tags_filename) # all the possible tags
    transition_dict_prob = transition_prob(train_filename) # getting all the transition probabilities
    output_prob = calc_output_prob(train_filename)  # getting all the ouptut probabilities
    test_file = open(in_test_filename, "r")
    f = open("viterbi_predictions.txt", "w")
    observations = []
    line = test_file.readline().strip()
    while line:
        observations.append(line)
        line = test_file.readline().strip()
        if (line == ""):
            line = test_file.readline().strip()
            opt = viterbi(observations, tag_list, transition_dict_prob, output_prob)
            for elem in opt:
                f.write(f'{elem}\n')
            observations.clear()
    

#print(viterbi_predict("twitter_tags.txt", "trans_probs.txt", "output_probs.txt", "twitter_dev_no_tag.txt","viterbi_predictions.txt"))

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    pass




def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '' #your working dir

    in_train_filename = f'{ddir}twitter_train.txt'

    naive_output_probs_filename = f'{ddir}naive_output_probs.txt'

    in_test_filename = f'{ddir}twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')


    naive_prediction_filename2 = f'{ddir}naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')
    
    trans_probs_filename =  f'{ddir}trans_probs.txt'
    output_probs_filename = f'{ddir}output_probs.txt'

    in_tags_filename = f'{ddir}twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')
    
    '''
    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}') 
    '''
    


if __name__ == '__main__':
    run()