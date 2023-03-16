# Implement the six functions below
import re

# question 2a


def naive_MLE():
    # initialise the dictionaries
    tag_count = {}
    tokens_for_tag = {}
    DELTA = 0.1

    # count number of unique words
    with open('twitter_train.txt', encoding="utf8") as unique:
        unique_words_count = set(unique.readlines())
        unique_count = len(unique_words_count)
        # print(unique_count) #6128

    with open('twitter_train.txt', encoding="utf8") as tags_file:
        # separate tokens and tags
        for line in tags_file:
            words = re.split(r'\s{1,}', line)
            # turn words to all lowercase
            token = words[0]
            tag = words[1]

            if tag in tag_count:
                tag_count[tag] += 1
                if token in tokens_for_tag[tag]:
                    tokens_for_tag[tag][token] += 1
                else:
                    tokens_for_tag[tag][token] = 1
            else:
                tag_count[tag] = 1
                tokens_for_tag[tag] = {}
                tokens_for_tag[tag][token] = 1
        # extra space due to the way i clean data so remove from dict keys
        del tag_count['']
        del tokens_for_tag['']

        # write to file

        file_name_submit = 'naive_output_probs.txt'
        with open(file_name_submit, "w", encoding="utf-8") as f:
            f.write("Token" + "\t" + "Tag" + "\t" +
                    "Probability of token given tag" + "\n" + "\n")

            for tag, token in tokens_for_tag.items():
                # print(tag)
                for key, value in token.items():
                    # print(tag)
                    # print(key)

                    # print(value)
                    probability = (
                        value + DELTA)/(tag_count.get(tag) + DELTA * (unique_count + 1))
                    # print(probability)
                    f.write(str(key) + "\t" + str(tag) +
                            "\t" + str(probability) + '\n')

            # prob of unknown tag
            prob_unknown = (0 + DELTA)/(0 + DELTA * (unique_count + 1))
            f.write("Unknown" + "\t" + "unk" + "\t" + str(prob_unknown))


naive_MLE()


def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    dic = {}
    with open(in_output_probs_filename, encoding="utf8") as in_prob_output:
        next(in_prob_output)
        next(in_prob_output)
        for prob_line in in_prob_output:
            prob_line = prob_line.strip()
            words = re.split(r'\t', prob_line)
            token = words[0].strip()
            tag = words[1].strip()
            probability = float(words[2])
            if token in dic:
                if tag in dic[token]:
                    dic[token][tag] += 1
                else:
                    dic[token][tag] = 1
            else:
                dic[token] = {}
                dic[token][tag] = 1

    # print(dic['thanks'])
    input_file = open(in_test_filename, encoding="utf-8")
    input = input_file.readline().strip()
    text = ''
    while input:
        # print(input)
        if input in dic:
            pred_token = max(dic[input])
            text += pred_token + '\n'

        else:
            pred_token = max(dic["Unknown"])
            text += pred_token + '\n'
            # print(input)
        input = input_file.readline().strip()
        if (input == "" or input == "\n"):
            input = input_file.readline()
        # print(pred_token)

    input_file.close()
    with open(out_prediction_filename, "w", encoding="utf-8") as pred:
        pred.write(text)
        #accuracy is only 45.6%


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
        predicted_tags = [l.strip()
                          for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip()
                             for l in fin.readlines() if len(l.strip()) != 0]

        assert len(predicted_tags) == len(ground_truth_tags)
        correct = 0
        for pred, truth in zip(predicted_tags, ground_truth_tags):
            if pred == truth:
                correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)
# evaluate('naive predictions.txt','twitter_dev_ans.txt')


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
