# GROUP: BUZZIN'
# Kelly Ng Kaiqing (A0240120H),
# Lim Xiang Ling (A0238445Y),
# Ng Yi Wei (A0238253E),
# Tan Sin Ler (A0240651N)

# Implement the six functions below

##################### HELPER FUNCTIONS #######################
import numpy as np
import re

# placing possible tags into a list
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
    with open(in_train_filename, encoding="utf8") as f:
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
    with open(in_train_filename, encoding="utf8") as f:
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
    with open(in_train_filename, encoding="utf8") as f:
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
    with open(in_train_filename, encoding="utf8") as f:
        # loop through every tag
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
    DELTA = 0.001
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
    with open(in_output_probs_filename, encoding="utf8") as probs_file:
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
with open('naive_output_probs.txt', 'w', encoding="utf8") as f:
    for token, tags_prob in output_probabilities.items():
        for tag, prob in tags_prob.items():
            f.write("{} \t {} \t {} \n ".format(tag, token, prob))

################################### QUESTION 2B #####################################


def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    naive_prediction = []
    naive_output_dict = convert_to_dict(in_output_probs_filename)
    with open(in_test_filename, encoding="utf8") as test_file:
        for word in test_file:
            token = word.strip()
            if (token):
                if token in naive_output_dict:
                    token_tags = naive_output_dict[token]
                    naive_prediction.append(
                        max(token_tags, key=token_tags.get))
                else:
                    unseen_token = naive_output_dict["unseen_token_null"]
                    naive_prediction.append(
                        max(unseen_token, key=unseen_token.get))

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

    with open(in_test_filename, encoding="utf8") as test_file:
        for word in test_file:
            token = word.strip()
            if (token):
                if token in naive_output_dict:
                    token_tags = naive_output_dict[token]
                    naive_prediction.append(
                        max(token_tags, key=token_tags.get))
                else:
                    unseen_token = naive_output_dict["unseen_token_null"]
                    naive_prediction.append(
                        max(unseen_token, key=unseen_token.get))

    with open(out_prediction_filename, "w") as f:
        for prediction in naive_prediction:
            f.write(prediction)
            f.write('\n')

################################### QUESTION 3C #####################################
# The accuracy of our prediction is 69.3% (3 s.f.), where the number of correctly
# predicted tags / number of predictions is 955/1378.

################################### QUESTION 4A #######################################

###########  TRANSITION PROBABILITY #############
# put tags in a list in order


def count_tweets(file):
    count_tweets = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.split()
            if len(line) == 2:
                continue
            else:
                count_tweets += 1
    return count_tweets


def tags_list(file):
    tags_list = []
    count_tweets = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.split()
            if len(line) == 2:
                tags_list.append(line[1].strip())
            else:
                tags_list.append('END'.strip())
                count_tweets += 1
    return tags_list

# transition dictionary: {(tag y - 1,tag y): count, (tag y, tag y+1): count}


def put_tags_in_list(file):
    tag_count_dictionary = count_tags(file)
    ls = []
    for i in tag_count_dictionary.keys():
        ls.append(i)
    return ls


def transition_dic(file):
    tag_count_dictionary = count_tags(file)
    transition_dic = {}
    tag_list = tags_list(file)
    unique_tag_list = put_tags_in_list(file)

    for idx in range(len(tag_list)):
        # handle first index
        if idx == 0:
            temp_tuple = ("START", tag_list[idx].strip())
            transition_dic[temp_tuple] = 1
            if tag_list[idx] in unique_tag_list:
                unique_tag_list.remove(tag_list[idx])

        # if it is not start or stop state
        elif tag_list[idx - 1] != 'END' and tag_list[idx] != 'END' and tag_list[idx + 1] != 'END':
            temp_tuple = (tag_list[idx].strip(), tag_list[idx + 1].strip())
            if temp_tuple not in transition_dic:
                transition_dic[temp_tuple] = 1
            else:
                transition_dic[temp_tuple] += 1

        # handle the distinction between 2 tweets
        elif tag_list[idx] == 'END':
            continue

        # if tag has both start and stop state
        elif tag_list[idx - 1] == 'END' and tag_list[idx] != 'END' and tag_list[idx + 1] == 'END':
            if tag_list[idx] in unique_tag_list:
                unique_tag_list.remove(tag_list[idx])

            temp_tuple_start = ("START", tag_list[idx].strip())
            if temp_tuple_start not in transition_dic:
                transition_dic[temp_tuple] = 1
            else:
                transition_dic[temp_tuple] += 1

            temp_tuple_end = (tag_list[idx].strip(), "STOP")
            if temp_tuple_end not in transition_dic:
                transition_dic[temp_tuple] = 1
            else:
                transition_dic[temp_tuple] += 1

        # if start state only:
        elif tag_list[idx] != 'END' and tag_list[idx - 1] == 'END':
            if tag_list[idx] in unique_tag_list:
                unique_tag_list.remove(tag_list[idx])

            temp_tuple_start = ("START", tag_list[idx].strip())
            if temp_tuple_start not in transition_dic:
                transition_dic[temp_tuple_start] = 1
            else:
                transition_dic[temp_tuple_start] += 1

            temp_tuple_transition = (
                tag_list[idx].strip(), tag_list[idx + 1].strip())
            if temp_tuple_transition not in transition_dic:
                transition_dic[temp_tuple_transition] = 1
            else:
                transition_dic[temp_tuple_transition] += 1

        # if stop state only:
        elif tag_list[idx] != 'END' and tag_list[idx + 1] == 'END':
            temp_tuple_end = (tag_list[idx].strip(), 'STOP')
            if temp_tuple_end not in transition_dic:
                transition_dic[temp_tuple_end] = 1
            else:
                transition_dic[temp_tuple_end] += 1

    # handle start states for unseen tags
    for tag in unique_tag_list:
        temp_tuple_start = ('START', tag)
        transition_dic[temp_tuple_start] = 0
    # print(transition_dic)
    return transition_dic

# compute transition probability:


def transition_probability(file):
    DELTA = 10
    words = num_words(file)
    tag_count_dictionary = count_tags(file)
    transition_dictionary = transition_dic(file)
    tweet_count = count_tweets(file)

    text = "{} \t {} \t {} \n".format("Yt-1", "Yt", "Transition Probability")
    trans_output_file = 'trans_output_file.txt'

    trans_sum_before_normalise = {}
    trans_output_before_normalise = {}
    # with open(trans_output_file, 'w', encoding="utf8") as trans_output_file:
    # knowns
    for tup, count in transition_dictionary.items():
        start_state = tup[0]
        next_state = tup[1]
        if start_state == "START":
            trans_prob = (count + DELTA) / \
                (tweet_count + DELTA * (words + 1))
        else:
            count_yt_1 = tag_count_dictionary[start_state]
            trans_prob = (count + DELTA) / \
                (count_yt_1 + DELTA * (words + 1))

        # adding to dictionary for normalising later
        if start_state not in trans_sum_before_normalise:
            trans_sum_before_normalise[start_state] = trans_prob
        else:
            trans_sum_before_normalise[start_state] += trans_prob

        # adding to trans_output_before_normalise
        trans_output_before_normalise[tup] = trans_prob

        # text += "{} \t {} \t {} \n ".format(start_state,
        #                                     next_state, trans_prob)
        # unknown tokens
    tags_seen = []
    for tag in tag_count_dictionary.keys():
        if tag not in tags_seen:
            count_yt_1 = tag_count_dictionary[tag]
            trans_prob = (DELTA) / (count_yt_1 + DELTA * (words + 1))

            if tag not in trans_sum_before_normalise:
                trans_sum_before_normalise[tag] = trans_prob
            else:
                trans_sum_before_normalise[tag] += trans_prob
                # text += "{} \t {} \t {} \n ".format(tag,
                #                                     "Unseen", trans_prob)
            tags_seen += tag
            trans_output_before_normalise[(tag, "Unseen")] = trans_prob
    print(trans_sum_before_normalise)
    print(trans_output_before_normalise)
    with open(trans_output_file, 'w', encoding="utf8") as trans_output_file:
        for tup, prob in trans_output_before_normalise.items():
            start_state = tup[0]
            next_state = tup[1]
            final_prob = prob / trans_sum_before_normalise[start_state]
            text += "{} \t {} \t {} \n ".format(start_state,
                                                next_state, final_prob)

        trans_output_file.write(text)

        # print(transition_probability)
        # trans_output_file.write(text)


###############  OUTPUT_PROBABILITY ###################
output_probabilities = calc_output_prob("twitter_train.txt")
with open('output_probs.txt', 'w', encoding="utf8") as f:
    for token, tags_prob in output_probabilities.items():
        for tag, prob in tags_prob.items():
            f.write("{} \t {} \t {} \n ".format(tag, token, prob))

twt_file = 'C:\\Users\\user\\Documents\\NUS\\Academics\\Y2S2\\BT3102\\project\\project_q4_5\\HMM-Twitter-Tagging\\twitter_train.txt'
transition_probability(twt_file)

##################################### QUESTION 4B #######################################


def count_lines(file):
    with open(file, 'r', encoding="utf8") as fp:
        num_lines = sum(1 for line in fp if line.rstrip())
        return num_lines


def put_line_in_list(file):
    ls = []
    with open(file, 'r', encoding="utf8") as fp:
        for line in fp:
            l = line.strip()
            if l not in ls:
                ls.append(l)
    return ls


def viterbi(observations, states, start_p, trans_p, emit_p, end_p):
    V = [{}]
    # print(emit_p)
    for st in states:
        if observations[0] not in emit_p[st]:
            V[0][st] = {"prob": start_p[st] * emit_p[st]
                        ['unseen_token_null'], "prev": None}
        else:
            V[0][st] = {"prob": start_p[st] * emit_p[st]
                        [observations[0]], "prev": None}

    for t in range(1, len(observations)):
        V.append({})

        for st in states:
            if st not in trans_p[states[0]]:
                max_tr_prob = V[t - 1][states[0]]["prob"] * \
                    trans_p[states[0]]['Unseen']

            else:
                max_tr_prob = V[t - 1][states[0]]["prob"] * \
                    trans_p[states[0]][st]

            prev_st_selected = states[0]

            for prev_st in states[1:]:
                if st not in trans_p[prev_st]:
                    tr_prob = V[t - 1][prev_st]["prob"] * \
                        trans_p[prev_st]['Unseen']
                else:
                    tr_prob = V[t - 1][prev_st]["prob"] * \
                        trans_p[prev_st][st]

                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            if observations[t] not in emit_p[st]:
                max_prob = max_tr_prob * emit_p[st]['unseen_token_null']
            else:
                max_prob = max_tr_prob * emit_p[st][observations[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    # # to account for stop states:
    # V.append({})
    # for st in states:
    #     max_tr_prob = V[-2][st]["prob"] * end_p[st]
    #     V[-1][st] = {"prob": max_tr_prob, "prev": st}
    # print(V[1]["@"])
    opt = []
    max_prob = 0.0
    best_st = None
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    return opt


def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    ans_list = []

    #### prep data ####

    # start_p {tag1: prob, tag2: prob}
    start_p = {}

    # trans_p {tag1: {tag1: prob2, tag2: prob2}}
    trans_p = {}

    # output_p {tag1: {token1: prob1}, tag2: {token2, prob2}}
    output_p = {}

    # end_p {tag1: prob, tag2: prob}
    end_p = {}

    # all unique tags in list
    unique_tag_list = put_line_in_list(in_tags_filename)

    # to account for missing stop states:
    stop_state_list = put_line_in_list(in_tags_filename)

    # make start, end and trans dictionaries
    with open(in_trans_probs_filename, encoding="utf8") as in_trans_probs_f:
        next(in_trans_probs_f)
        for prob_line in in_trans_probs_f:
            prob_line = prob_line.strip()
            if prob_line:
                words = re.split(r'\t', prob_line)
                yt_1 = words[0].strip()
                yt = words[1].strip()
                probability = float(words[2])

                if yt_1 == "START":
                    # adding into start_p
                    start_p[yt] = probability
                elif yt == "STOP":
                    end_p[yt_1] = probability
                    if yt_1 in stop_state_list:
                        stop_state_list.remove(yt_1)
                else:
                    # adding into trans_p
                    if yt_1 in trans_p:
                        trans_p[yt_1][yt] = probability
                    else:
                        trans_p[yt_1] = {}
                        trans_p[yt_1][yt] = probability
    # print(trans_p["@"])
    summing = 0
    # print(len(unique_tag_list))
    for tag in unique_tag_list:
        for i in trans_p[tag].values():
            summing += i
        # print(tag)
        # print(summing)
        summing = 0

    # account for missing stop states:
    for tag in stop_state_list:
        # print(trans_p[tag])
        end_p[tag] = trans_p[tag]["Unseen"]

    # make output_p dictionary
    with open(in_output_probs_filename, encoding="utf8") as in_output_prob_f:
        for prob_line in in_output_prob_f:
            prob_line = prob_line.strip()
            if prob_line:
                words = re.split(r'\t', prob_line)
                tag = words[0].strip()
                token = words[1].strip()
                probability = float(words[2])

                # adding into output_p
                if tag in output_p:
                    output_p[tag][token] = probability
                else:
                    output_p[tag] = {}
                    output_p[tag][token] = probability

    print(output_p["@"]["unseen_token_null"])

    # print(output_p["@"])

    #### Do actual viterbi on each tweet ####
    with open(in_test_filename, encoding="utf8") as f:
        one_tweet = []
        for line in f:
            if line.strip():
                one_tweet.append(line.strip())
            else:
                # print(one_tweet)
                ans_list.extend(
                    viterbi(one_tweet, unique_tag_list, start_p, trans_p, output_p, end_p))
                one_tweet.clear()

    # write into output prediction file
    with open(out_predictions_filename, 'w', encoding="utf8") as output_pred_f:
        for tag in ans_list:
            output_pred_f.write(tag + '\n')

    # print(output_p)


# def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
#                     out_predictions_filename):

#     num_hidden_states = count_lines(in_tags_filename)

#     with open(in_test_filename, 'w', encoding="utf8") as in_test_f:
#         for line in in_test_f:
#             if line.strip():

    # pi_matrix = np.empty((sequence_length, num_hidden_states))
    # bp = np.empty((sequence_length, num_hidden_states))

    #################################### QUESTION 5 ###########################################


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


def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = 'C:\\Users\\user\\Documents\\NUS\\Academics\\Y2S2\\BT3102\\project\\project_q4_5\\HMM-Twitter-Tagging'

    in_train_filename = f'{ddir}\\twitter_train.txt'
    naive_output_probs_filename = f'{ddir}\\naive_output_probs.txt'

    in_test_filename = f'{ddir}\\twitter_dev_no_tag.txt'
    in_ans_filename = f'{ddir}\\twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}\\naive_predictions.txt'
    naive_predict(naive_output_probs_filename,
                  in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename,
                   in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename = f'{ddir}\\trans_output_file.txt'
    output_probs_filename = f'{ddir}\\output_probs.txt'

    in_tags_filename = f'{ddir}\\twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}\\viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(
        viterbi_predictions_filename, in_ans_filename)
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
