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
# count number of transitions, returns a dictionary of dictionary which counts j from i
# e.g. {yt-1 = i1:{yt = j1: count, yt = j2:count}, yt-1 = i2:{yt = j1: count, yt = j2:count}}
def count_transition_tags(in_train_filename):
    freqs = {}
    tags_list = []
    with open(in_train_filename) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                tag = temp_list[1]
                tags_list.append(tag)
    for num in range(1, len(tags_list)):
        i = tags_list[num - 1]
        j = tags_list[num]
        if i in freqs:
            inital_to = freqs[i]
            if j in inital_to:
                inital_to[j] += 1
            else:
                inital_to[j] = 1
        else:
            freqs[i] = {}
            inital_to = freqs[i]
            inital_to[j] = 1
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

def calc_transition_prob(in_train_filename):
    trans_probabilities = {}
    DELTA = 0.1
    words = num_words(in_train_filename)
    tags_dict = count_tags(in_train_filename)
    transition_dict = count_transition_tags(in_train_filename)
    tags_list = tags("twitter_tags.txt")

    """ trans_prob["unseen_trans_null"] = {}
    for i in range(0, len(tags_list)):
        initial = tags_list[i]
        initial_count = tags_dict[initial]
        for j in range(0, len(tags_list)):
            to = tags_list[j]
            numerator = DELTA
            denomenator = initial_count + DELTA * (words + 1)
            trans_prob """
    # e.g. {yt-1 = i1:{yt = j1: trans, yt = j2:trans}, yt-1 = i2:{yt = j1: trans, yt = j2:trans}}
    for initial, finals in transition_dict.items():
        for final, count in finals.items():
            numerator = count
            denominator = tags_dict[initial]
            
            if initial in trans_probabilities:
                initial_to = trans_probabilities[initial]
                initial_to[final] = numerator / denominator
            
            else:
                trans_probabilities[initial] = {}
                initial_to = trans_probabilities[initial]
                initial_to[final] = numerator / denominator
    return trans_probabilities

"""     for tag in tags_list:
        tag_count = tags_dict[tag]
        num = DELTA
        den = tag_count + DELTA * (words + 1)
        output_probabilities["unseen_token_null"][tag] = num/den """



# takes in the naive_output_probs txt file and output a dictionary
# in the following form: 
# {token=x1:{tag=y1:ouput_probability, tag=y2:ouput_probability}, token=x2:{tag=y1:ouput_probability, tag=y2:ouput_probability}}
def output_convert_to_dict(in_output_probs_filename):
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

def trans_convert_to_dict(in_trans_probs_filename):
    trans_probabilities = {}
    with open(in_trans_probs_filename) as probs_file:
        for line in probs_file:
            l = line.strip().split()
            if l:
                initial = l[0]
                final = l[1]
                prob = float(l[2])
                if initial in trans_probabilities:
                    finals_probs = trans_probabilities[initial]
                    finals_probs[final] = prob
                else:
                    trans_probabilities[initial] = {}
                    finals_probs = trans_probabilities[initial]
                    finals_probs[final] = prob
    return trans_probabilities

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
    naive_output_dict = output_convert_to_dict(in_output_probs_filename)
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
    naive_output_dict = output_convert_to_dict(in_output_probs_filename)
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

################################### QUESTION 4A #####################################
trans_probabilities = calc_transition_prob("twitter_train.txt")
with open('trans_probs.txt', 'w') as f:
    for inital, finals in trans_probabilities.items():
        for final, prob in finals.items():
            f.write("{} \t {} \t {} \n ".format(inital, final, prob))

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    pass

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
    '''
    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

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