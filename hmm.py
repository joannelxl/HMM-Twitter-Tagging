# Q2(a)
import pandas as pd

def denomenator(file):
    trainData = open(file, encoding = "utf8")
    #{tag:count}
    tags_freq = {}

    for line in trainData:
        newLine = line.strip()
        #check that line is not empty
        if newLine:
            token_and_tag = newLine.split()
            #print(token_and_tag[0])
            word = token_and_tag[0]
            tag = token_and_tag[1]
            if tag not in tags_freq:
                tags_freq[tag] = 1
            else:
                tags_freq[tag] += 1
                
    return tags_freq


#print(denomenator('twitter_train.txt'))


def numerator(file):
    trainData = open(file, encoding = "utf8")
    #{tag:{"Tokens": {Token:count}}
    association_freq = {}

    for line in trainData:
        newLine = line.strip()
        #check that line is not empty
        if newLine:
            token_and_tag = newLine.split()
            token = token_and_tag[0]
            tag = token_and_tag[1]

            if tag not in association_freq:
                association_freq[tag] = {"Tokens":{token:1}, "count":1}
            else:
                association_freq[tag]["count"] += 1
                if token not in association_freq[tag]["Tokens"]:
                    association_freq[tag]["Tokens"][token]= 1
                else:
                    association_freq[tag]["Tokens"][token] += 1
    return association_freq

#print(numerator('twitter_train.txt'))

#number of unique words in the training data
def num_words(file):
    trainData = open(file, encoding = "utf8")
    freqs = {}
    for line in trainData:
        l = line.strip()
        if l:
            tempList = l.split()
            token = tempList[0]
            if token not in freqs:
                freqs[token] = 1
    return sum(freqs.values())
        

#print(num_words('twitter_train.txt')) #ans: 5763

def output_probability(train_file, no_tag_file, output_file):
    smoothing = 0.1
    results = {"token": [], "tag": [], "prob": []}
    numWords = num_words(train_file)
    #tag, tokens (count)
    tag_and_tokens = numerator(train_file)
    #tag and count
    tag_and_counts = denomenator(train_file)

    for tag, tokens in tag_and_tokens.items():
        yj = tag_and_tokens[tag]["count"]
        denom = yj + smoothing * (numWords + 1)
        tag_and_tokens[tag]["Tokens"]["UnseenWord"] = smoothing / denom

        for word in tag_and_tokens[tag]["Tokens"]:
            yj_xw = tag_and_tokens[tag]["Tokens"][word]

            numer = yj_xw + smoothing
            denom1 = yj + smoothing * (numWords + 1)
            b_jw = numer/denom1
            tag_and_tokens[tag]["Tokens"][word] = b_jw

    for tag, tokens in tag_and_tokens.items():
        for word in tag_and_tokens[tag]["Tokens"]:
            b_jw = tag_and_tokens[tag]["Tokens"][word]
            results["token"].append(word)
            results["tag"].append(tag)
            results["prob"].append(b_jw)

    output_probabilities = pd.DataFrame.from_dict(results)
    #output_probabilities.to_csv(output_file, index = False)

    return output_probabilities

final = output_probability('twitter_train.txt', 'twitter_train_no_tag.txt',
                   'naive_output_probs.txt')
final.to_csv('naive_output_probs.txt', index = False, header = False)

#-----------------------------------Given codes-------------------------------------#

# Implement the six functions below

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    ### Get output_probs from naive output probs.txt
    pass


    





def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
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

    ddir = ''  # your working dir

    in_train_filename = f'{ddir}twitter_train.txt'

    naive_output_probs_filename = f'{ddir}naive_output_probs.txt'

    in_test_filename = f'{ddir}twitter_dev_no_tag.txt'
    in_ans_filename = f'{ddir}twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}naive_predictions.txt'
    naive_predict(naive_output_probs_filename,
                  in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename,
                   in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename = f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(
        viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 = f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(
        viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')


if __name__ == '__main__':
    run()
