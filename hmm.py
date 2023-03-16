file = open("naive_output_probs.txt", "w")
file = open("naive_predictions.txt", "w")
# Implement the six functions below
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    # naive_predict(naive_output_probs.txt, twitter_dev_no_tag.txt, naive_predictions.txt)
    file = open('twitter_train.txt')
    #f = open(in_output_probs_filename, "w")
    pair = file.readline()
    token_tag_counter_dict = {}
    tag_counter_dict = {}

    # getting no of times a tag appeared, and no of times a token appeared (tgt w the associated tag)
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
    file.close()


    prob_file = open(in_output_probs_filename, "a")
    items = token_tag_counter_dict.items()
    max_lst = []
    lst = []
    final_dict = {}
    for (key,value) in items:
        for (k,v) in value.items():
            prob_pair = v
            tag = prob_pair[1]
            prob = prob_pair[0]/tag_counter_dict[tag][0]
            lst.append(f'{key}\t{tag}\t{prob}\n')
        max_pair = max(token_tag_counter_dict[key].values())
        tag = max_pair[1]
        prob = max_pair[0]/tag_counter_dict[tag][0]
        if f'{key}\t{tag}\t{prob}\n' not in max_lst:
            max_lst.append(f'{key}\t{tag}\t{prob}\n')
            final_dict[key] = tag
    for elem in lst:
        prob_file.write(elem)
    prob_file.close()

    input_file = open(in_test_filename)
    num_unique_words = len(lst)
    prob_unknown_word = 0.01 / (0.01 * num_unique_words + 1)
    input = input_file.readline().strip()
    prediction_file = open(out_prediction_filename, "a")

    print(input)
    while input:
        if input in final_dict.keys():
            final_tag = final_dict.get(input)
        else: 
            final_tag = '@'
        prediction_file.write(f'{final_tag}\n')
        input = input_file.readline().strip()
        if (input == "" or input == "\n"):
            input = input_file.readline()
    prediction_file.close()
    input_file.close()
    pass

file = open("naive_predictions2.txt", "w")
def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    #naive_predict2('naive_output_probs.txt', 'twitter_train.txt', 'twitter_dev_no_tag.txt', 'naive_predictions2.txt')
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
            tag_token_counter_dict[token][tag] = ((prob_tag * prob / prob_token), tag)
        else:
            tag_token_counter_dict[token] = {}
            tag_token_counter_dict[token][tag] = ((prob_tag * prob / prob_token), tag)
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
