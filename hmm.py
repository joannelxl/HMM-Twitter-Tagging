# Implement the six functions below
def get_token_tag_count_from_file(train_data_filename):
    file = open(train_data_filename)
    pair = file.readline().strip()
    token_tag_counter_dict = {}
    # getting no of times a token appears with the associated tag
    while pair:
        new_pair = pair.split()
        token = new_pair[0]
        tag = new_pair[1]
        if token in token_tag_counter_dict.keys():
            if tag in token_tag_counter_dict[token].keys():
                count = token_tag_counter_dict[token][tag][0]
                token_tag_counter_dict[token][tag] = (count + 1, tag)
            else:
                token_tag_counter_dict[token][tag] = (1, tag)
        else:
            token_tag_counter_dict[token] = {}
            token_tag_counter_dict[token][tag] = (1, tag)
        pair = file.readline()
        if pair == '\n':
            pair = file.readline()
    file.close()
    return token_tag_counter_dict

def get_tag_count_from_file(train_data_filename):
    file = open(train_data_filename)
    pair = file.readline().strip()
    tag_counter_dict = {}
    # getting no of times a token appears with the associated tag
    while pair:
        new_pair = pair.split()
        token = new_pair[0]
        tag = new_pair[1]
        if tag in tag_counter_dict.keys():
            count = tag_counter_dict[tag][0]
            tag_counter_dict[tag] = (count + 1, tag)
        else:
            tag_counter_dict[tag] = (1, tag)
        pair = file.readline()
        if pair == '\n':
            pair = file.readline()
    file.close()
    return tag_counter_dict

def get_num_unique_words(filename):
    d = get_token_tag_count_from_file(filename)
    return len(d)

def add_stuff_in_prob_file(in_output_probs_filename, token_tag_counter_dict, tag_counter_dict):
    prob_file = open(in_output_probs_filename, "a")
    items = token_tag_counter_dict.items()
    num_unique_words = len(token_tag_counter_dict)
    max_lst = []
    lst = []
    final_dict = {}
    token_tag_prob_dict = {}
    SIGMA = 0.01
    for (key,value) in items: # key is token and value is a dict of tag and count
        token_tag_prob_dict[key] = {}
        for (k,v) in value.items(): # k is tag and v is a tuple of count and tag
            prob_pair = v
            tag = prob_pair[1]
            prob = (prob_pair[0] + SIGMA)/(tag_counter_dict[tag][0] + SIGMA * (num_unique_words + 1))
            lst.append(f'{key}\t{tag}\t{prob}\n')
            token_tag_prob_dict[key][k] = (prob, tag)
        max_pair = max(token_tag_prob_dict[key].values())
        tag = max_pair[1]
        prob = max_pair[0]
        if f'{key}\t{tag}\t{prob}\n' not in max_lst:
            max_lst.append(f'{key}\t{tag}\t{prob}\n')
            final_dict[key] = tag
    unknown_dict = {}
    for (key, value) in tag_counter_dict.items():
        prob = (0 + SIGMA) / (tag_counter_dict[key][0] + SIGMA * (num_unique_words + 1))
        unknown_dict[key] = (prob, key)
    max_pair = max(unknown_dict.values())
    tag = max_pair[1]
    prob = max_pair[0]
    final_dict['unknown'] = tag
    max_lst.append(f'unknown\t{tag}\t{prob}\n')
    for elem in lst:
        prob_file.write(elem)
    prob_file.close()
    return final_dict
    
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    # naive_predict(naive_output_probs.txt, twitter_dev_no_tag.txt, naive_predictions.txt)
    file = open("naive_output_probs.txt", "w")
    file = open("naive_predictions.txt", "w")

    token_tag_counter_dict = get_token_tag_count_from_file('twitter_train.txt')
    tag_counter_dict = get_tag_count_from_file('twitter_train.txt')
    num_unique_words = get_num_unique_words('twitter_train.txt')
    
    # final_dict is a dict of token and predicted tag
    final_dict = add_stuff_in_prob_file(in_output_probs_filename, token_tag_counter_dict, tag_counter_dict)

    input_file = open(in_test_filename)
    input = input_file.readline().strip()
    prediction_file = open(out_prediction_filename, "a")

    #print(token_tag_counter_dict)
    while input:
        # if input in final_dict.keys():
        final_tag = final_dict.get(input)
        # else: 
            # final_tag = '@'
        prediction_file.write(f'{final_tag}\n')
        input = input_file.readline().strip()
        if (input == "" or input == "\n"):
            input = input_file.readline()
    prediction_file.close()
    input_file.close()
    pass


def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    #naive_predict2('naive_output_probs.txt', 'twitter_train.txt', 'twitter_dev_no_tag.txt', 'naive_predictions2.txt')
    file = open("naive_predictions2.txt", "w")
    train_file = open(in_train_filename)
    pair = train_file.readline().strip()
    tag_dict = {}
    token_dict = {}
    # getting prob of tag and token from train set
    while pair:
        new_pair = pair.split()
        token = new_pair[0]
        tag = new_pair[1]
        if tag not in tag_dict.keys():
            tag_dict[tag] = 1
        else:
            count = tag_dict[tag]
            tag_dict[tag] = count + 1
        if token not in token_dict.keys():
            token_dict[token] = 1
        else:
            count = token_dict[token]
            token_dict[token] = count + 1
        pair = train_file.readline().strip()
        if pair == "":
            pair = train_file.readline().strip()
    train_file.close() # done with train set

    total_word_count = sum(tag_dict.values())
    prob_file = open(in_output_probs_filename)
    prediction_file = open(out_prediction_filename, "a")
    # transforming the probability
    triplet = prob_file.readline().strip().split()

    tag_token_counter_dict = {}
    while triplet:
        token = triplet[0]
        tag = triplet[1]
        prob = float(triplet[2])
        prob_tag = tag_dict[tag] / total_word_count
        prob_token = token_dict[token] / total_word_count
        if token in tag_token_counter_dict.keys():
            tag_token_counter_dict[token][tag] = ((prob_token * prob / prob_tag), tag)
        else:
            tag_token_counter_dict[token] = {}
            tag_token_counter_dict[token][tag] = ((prob_token * prob / prob_tag), tag)
        triplet = prob_file.readline().strip().split()
        if triplet == '':
            triplet = prob_file.readline().strip().split()
    prob_file.close()
    items = tag_token_counter_dict.items()
    final_dict = {}
    for (key, value) in items:
        max_pair = max(tag_token_counter_dict[key].values())
        prob = max_pair[0]
        tag = max_pair[1]
        final_dict[key.strip()] = tag
    input_file = open(in_test_filename)
    input = input_file.readline().strip()
    while input: 
        if input in final_dict.keys():
            pred_tag = final_dict[input]
        else:
            pred_tag = '@'
        prediction_file.write(f'{pred_tag}\n')
        input = input_file.readline().strip()
        if (input == "" or input == "\n"):
            input = input_file.readline().strip()
    prediction_file.close()
    input_file.close()

    pass

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
