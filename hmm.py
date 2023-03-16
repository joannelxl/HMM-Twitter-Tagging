# Implement the six functions below
import json
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    naive_prediction = []
    with open(in_output_probs_filename) as probs_file:
        data = probs_file.read()
        naive_output_dict = json.loads(data)
        with open(in_test_filename) as test_file:
            for word in test_file:
                token = word.strip()
                if (token):
                    if token in naive_output_dict:
                        token_tags = naive_output_dict[token]
                        naive_prediction.append(max(token_tags, key=token_tags.get))       
                    else:
                        unseen_token = naive_output_dict["unseen token"]
                        naive_prediction.append(max(unseen_token, key=unseen_token.get))

    with open(out_prediction_filename, "w") as f:
        for prediction in naive_prediction:
            f.write(prediction)
            f.write('\n')
                        

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    naive_prediction = []
    word_count_dict = count_words(in_train_filename)
    tag_count_dict = count_tags(in_train_filename)
    
    total_tokens = sum(tag_count_dict.values())
    tag_probs = {}
    token_probs = {}
    for key,value in tag_count_dict.items():
        tag_probs[key] = value/total_tokens

    for k,v in word_count_dict.items():
        token_probs[k] = v/total_tokens

    with open(in_output_probs_filename) as probs_file:
        data = probs_file.read()
        naive_output_dict = json.loads(data)
        with open(in_test_filename) as test_file:
            for word in test_file:
                token = word.strip()
                if (token):
                    if token in naive_output_dict:
                        token_tags = naive_output_dict[token]
                        token_prob = token_probs[token]
                        for tag, output_prob in token_tags.items():
                            token_tags[tag] = (output_prob * token_prob)/tag_probs[tag]
                        naive_prediction.append(max(token_tags, key=token_tags.get))
                    else:
                        unseen_token = naive_output_dict["unseen token"]
                        for tag, output_prob in unseen_token.items():
                            #can multiply by 0??
                            unseen_token[tag] = output_prob/tag_probs[tag]
                        naive_prediction.append(max(unseen_token, key=unseen_token.get))  
    with open(out_prediction_filename, "w") as f:
        for prediction in naive_prediction:
            f.write(prediction)
            f.write('\n')                         



def count_words(file):
    freqs = {}
    with open(file) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = []
                for word in line.split():
                    temp_list.append(word)
                token = temp_list[0]

                if token in freqs:
                    freqs[token] += 1
                else:
                    freqs[token] = 1
    print(freqs)
    return freqs

def count_tags(file):
    freqs = {}
    with open(file) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = []
                for t in line.split():
                    temp_list.append(t)
                tag = temp_list[1]

                if tag in freqs:
                    freqs[tag] += 1
                else:
                    freqs[tag] = 1
    print(freqs)
    return freqs



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
"""
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
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}') """
    


if __name__ == '__main__':
    run()