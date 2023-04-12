# GROUP: BUZZIN'
# Kelly Ng Kaiqing (A0240120H), 
# Lim Xiang Ling (A0238445Y), 
# Ng Yi Wei (A0238253E), 
# Tan Sin Ler (A0240651N)

# Implement the six functions below
DELTA = 0.1
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
    tags_list.append("")
    with open(in_train_filename) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                tags_list.append(temp_list[1])
            else:
                tags_list.append("")

    for num in range(1, len(tags_list)):
        i = tags_list[num - 1]
        j = tags_list[num]
        if (i == ""):
            if ("START" in freqs):
                inital_to = freqs["START"]
                if (j in inital_to):
                    inital_to[j] += 1
                else:
                    inital_to[j] = 1
            else:
                freqs["START"] = {}
                inital_to = freqs["START"]
                inital_to[j] = 1

        elif (j == ""):
            if (i in freqs):
                inital_to = freqs[i]
                if ("STOP" in inital_to):
                    inital_to["STOP"] += 1
                else:
                    inital_to["STOP"] = 1
            else:
                freqs[i] = {}
                inital_to = freqs[i]
                inital_to["STOP"] = 1
        else:
            if (i in freqs):
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
def calc_naive_output_prob(in_train_filename):
    output_probabilities = {}
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



################################### QUESTION 2A #####################################
# For this question, we chose a DELTA value of 0.1.
naive_output_probabilities = calc_naive_output_prob("twitter_train.txt")
with open('naive_output_probs.txt', 'w') as f:
    for token, tags_prob in naive_output_probabilities.items():
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

################################### QUESTION 4 ######################################

########################## ADDITIONAL HELPER FUNCTIONS ##############################

# calculates the output probabilities for 4a
def calc_output_prob(in_train_filename):
    output_probs_dict = calc_naive_output_prob(in_train_filename)
    tags_list = tags("twitter_tags.txt")
    for token in output_probs_dict:
        for tag in tags_list:
            if (tag not in output_probs_dict[token]):
                prob = output_probs_dict["unseen_token_null"][tag]
                output_probs_dict[token][tag] = prob
    return output_probs_dict

# calculates the transition probabilities for 4a
def calc_transition_prob(in_train_filename, in_tags_filename):
    trans_probabilities = {}
    words = num_words(in_train_filename)
    transition_dict = count_transition_tags(in_train_filename)
    tags_list = tags(in_tags_filename)

    # e.g. {yt-1 = i1:{yt = j1: trans, yt = j2:trans}, yt-1 = i2:{yt = j1: trans, yt = j2:trans}}
    for initial, finals in transition_dict.items():
        for final, count in finals.items():
            numerator = count + DELTA
            denominator = sum(transition_dict[initial].values()) + DELTA * (words + 1) # smoothing
            
            if (initial in trans_probabilities):
                initial_to = trans_probabilities[initial]
                initial_to[final] = numerator / denominator
            
            else:
                trans_probabilities[initial] = {}
                initial_to = trans_probabilities[initial]
                initial_to[final] = numerator / denominator
    
    for i in trans_probabilities:
        for tag in tags_list:
            if (tag not in trans_probabilities[i]):
                num = DELTA
                den = sum(transition_dict[i].values()) + DELTA * (words + 1) # smoothing
                trans_probabilities[i][tag] = num / den
        if (i != "START" and "STOP" not in trans_probabilities[i]):
            num = DELTA
            den = sum(transition_dict[i].values()) + DELTA * (words + 1) # smoothing
            trans_probabilities[i]["STOP"] = num / den

    # Normalise so that row probabilties will sum up to 1
    sum_dict = {}
    for i in trans_probabilities:
        sum_dict[i] = sum(trans_probabilities[i].values())  
    for i, trans_probs in trans_probabilities.items():
        for j, trans_prob in trans_probs.items():
            trans_probabilities[i][j] = trans_prob/sum_dict[i]
    return trans_probabilities

# Taking probabilities from trans_probs.txt and converting them into a dictionary
# in the form of {yt-1 = i1:{yt = j1: trans, yt = j2:trans}, yt-1 = i2:{yt = j1: trans, yt = j2:trans}}
def trans_convert_to_dict(in_trans_probs_filename):
    trans_probabilities = {}
    with open(in_trans_probs_filename) as probs_file:
        next(probs_file)
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

################################### QUESTION 4A #####################################

output_probabilities = calc_output_prob("twitter_train.txt")
with open('output_probs.txt', 'w') as f:
    for token, tags_prob in output_probabilities.items():
        for tag, prob in tags_prob.items():
            f.write("{} \t {} \t {} \n ".format(tag, token, prob))

trans_probabilities = calc_transition_prob("twitter_train.txt", "twitter_tags.txt")
with open('trans_probs.txt', 'w') as f:
    f.write("t \t\t t+1 \t prob \n ")
    for inital, finals in trans_probabilities.items():
        for final, prob in finals.items():
            f.write("{} \t {} \t {} \n ".format(inital, final, prob))

################################### QUESTION 4B #####################################

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    tags_list = tags(in_tags_filename)
    output_dict = output_convert_to_dict(in_output_probs_filename)
    trans_dict = trans_convert_to_dict(in_trans_probs_filename)
    viterbi_prediction = []
    testWords = []
    testWords.append(" ")
    with open(in_test_filename) as test_file:
        for word in test_file:
            w = word.strip()
            if (w):
                testWords.append(w)
            else:
                testWords.append(" ")

    pi = {}
    bp = {}
    step = 1
    for num in range(1, len(testWords)):
        # Initial step
        if (testWords[num-1] == " "):
            pi[step] = {}
            bp[step] = {}
            
            for tag in tags_list:
                if (testWords[num] in output_dict):
                    pi[step][tag] = trans_dict["START"][tag] * output_dict[testWords[num]][tag]
                else:
                    pi[step][tag] = trans_dict["START"][tag] * output_dict["unseen_token_null"][tag]
                bp[step][tag] = 0
            step += 1
        
        # Final step
        elif (testWords[num] == " "):
            temp_dict = {}
            temp_list = []
            prev = step - 1
            for t, tag_prob in pi[prev].items():
                temp_dict[t] = tag_prob * trans_dict[t]["STOP"]
            maxProb = max(temp_dict.values())
            finalBP = max(temp_dict, key=temp_dict.get)
            temp_list.append(finalBP)
            curr_step = max(bp)
            curr_bp = bp[curr_step][finalBP]
            while curr_bp != 0:
                temp_list.append(curr_bp)
                curr_step -= 1
                curr_bp = bp[curr_step][curr_bp]
            temp_list = list(reversed(temp_list))
            viterbi_prediction.extend(temp_list)
            step = 1
            pi = {}
            bp = {}
            
        # Recursion steps
        else:
            pi[step] = {}
            bp[step] = {}
            prev = step - 1
            
            for j in tags_list:
                temp = {}
                for i, prev_prob in pi[prev].items():

                    if (testWords[num] in output_dict):
                        temp[i] = prev_prob * trans_dict[i][j] * output_dict[testWords[num]][j]
                    else:
                        temp[i] = prev_prob * trans_dict[i][j] * output_dict["unseen_token_null"][j]
                
                pi[step][j] = max(temp.values())
                bp[step][j] = max(temp, key=temp.get)

            step += 1

    with open(out_predictions_filename, "w") as f:
        for prediction in viterbi_prediction:
            f.write(prediction)
            f.write('\n')

################################### QUESTION 4C #####################################
# The accuracy of our prediction is 75.2% (3 s.f.), where the number of correctly 
# predicted tags / number of predictions is 1036/1378.


################################### QUESTION 5 ######################################

################################## QUESTION 5A ######################################
""" 
# Meaningful groups

We observed that tokens starting with '@USER_' are mostly associated with the tag "@", and tokens starting with 'http' or 'www.' (i.e. links) 
are mostly associated with the tag "U" in 'twitter_train_txt', regardless of the remaining characters in the token. Hence, when computing 
the viterbi algorithm, we clustered the tokens that followed these patterns into meaningful groups. We did it by automatically assigning 
emission probabilities of 1 to tokens that start with '@USER_' for tag "@" and 0 to all other tags. Similarly, we assigned emission probabilities 
of 1 to tokens that start with 'http' or 'www.' for tag "U", and 0 for all other tags. 

The purpose of this is to ensure that the predicted tag for all users will only be "@", and the predicted tag for all links will only be "U". 


# Unseen words - lower case

We have also converted all tokens in the training and test files into lower case. This is to account for cases where unseen words have actually 
been seen in the training data set, for example the token 'hello' and 'Hello' are essentially the same word. If the word ‘hello’ is seen in the 
training data set but ‘Hello’ is not seen, we would have predicted ‘Hello’ as an unseen token. However, the predicted tag should be the same in 
both cases.


# Hapax legomenon

Tokens that appeared only once in the entire train set may not have the correct tag prediction as there are not enough instances to predict the 
tags accurately. Hence the probability calculation may not be accurate. To counter this problem, our group extracted these tokens out, calculated 
the hapax_probability and scaled the output_probs probability by this value.

We first created a dictionary, hapax_dic = {hapax_tag: count} to keep track of the tags with tokens only appearing once in the entire train data.

We then extracted the words that only appeared once in the entire train data. For these tokens, we identified the corresponding tag and added it 
to the count in the hapax_dic.

Afterwards, we calculated the probability of each hapax tag: hapax_tag_count / total_hapax_count 
total_hapax_count is the total number of tokens that appeared only once in the train set. 

Finally, we incorporated this in the output probability calculation. For those tags that are in hapax_dic, we multiplied delta = 0.1 by the hapax_prob[tag].
 """
########################## ADDITIONAL HELPER FUNCTIONS ##############################

# Converting tokens to lower case and counting the number of tokens with respect to a tag
# {tag: {token: count}}
def count_tag_token_lower(in_train_filename):
    freqs = {}
    with open(in_train_filename) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                token = temp_list[0].lower()
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

# Counting the number of unqiue tokens when the tokens are in lower case
def num_words_lower(in_train_filename):
    freqs = {}
    with open(in_train_filename) as f:
        #loop through every tag
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                token = temp_list[0].lower()
                if token not in freqs:
                    freqs[token] = 1
    return sum(freqs.values())

# Total count of tokens in lower case
def count_tokens_lower(in_train_filename):
    freqs = {}
    with open(in_train_filename) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                token = temp_list[0].lower()
                if token in freqs:
                    freqs[token] += 1
                else:
                    freqs[token] = 1
    return freqs

# Counting the number of tags with respect to a token
# {token: {tag: count}}
def count_token_tag_lower(in_train_filename):
    freqs = {}
    with open(in_train_filename) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = l.split()
                token = temp_list[0].lower()
                tag = temp_list[1]

                if token in freqs:
                    tag_tokens = freqs[token]
                    if tag in tag_tokens:
                        tag_tokens[tag] += 1
                    else:
                        tag_tokens[tag] = 1
                else:
                    freqs[token] = {}
                    tag_tokens = freqs[token]
                    tag_tokens[tag] = 1
    return freqs

# Calculating the new ouput probability with hapax and tokens in lower case
def calc_new_output_prob(in_train_filename):
    output_probabilities = {}
    words = num_words_lower(in_train_filename)
    tags_dict = count_tags(in_train_filename)
    tags_tokens_dict = count_tag_token_lower(in_train_filename)
    token_tag_dict = count_token_tag_lower(in_train_filename)
    tags_list = tags("twitter_tags.txt")
    count_token_dict = count_tokens_lower(in_train_filename)


    hapex_count_per_tag = {}
    total_hapex_count = 0
    hapex_prob = {}

    for token, count in count_token_dict.items():
        if count == 1:
            for tag, count in token_tag_dict[token].items():
                if tag not in hapex_count_per_tag:
                    hapex_count_per_tag[tag] = 1
                else:
                    hapex_count_per_tag[tag] += 1
                total_hapex_count += 1
    for tag, count in hapex_count_per_tag.items():
        hapex_prob[tag] = count / total_hapex_count

    output_probabilities["unseen_token_null"] = {}
    for tag in tags_list:
        tag_count = tags_dict[tag]
        num = DELTA
        den = tag_count + DELTA * (words + 1) # smoothing
        output_probabilities["unseen_token_null"][tag] = num/den

    for tag, tags_count in tags_dict.items():
        tags_tokens = tags_tokens_dict[tag]
        for token, tokens_count in tags_tokens.items():
            if tag in hapex_prob:
                numerator = tokens_count + DELTA * hapex_prob[tag] 
                denominator = tags_count + (DELTA * hapex_prob[tag]) * (words + 1) # smoothing
            else:
                numerator = tokens_count + DELTA 
                denominator = tags_count + DELTA * (words + 1) # smoothing

            if token in output_probabilities:
                token_prob_tag = output_probabilities[token]
                token_prob_tag[tag] = numerator/denominator
            else:
                output_probabilities[token] = {}
                token_prob_tag = output_probabilities[token]
                token_prob_tag[tag] = numerator/denominator

    for token in output_probabilities:
        for tag in tags_list:
            if (tag not in output_probabilities[token]):
                prob = output_probabilities["unseen_token_null"][tag]
                output_probabilities[token][tag] = prob
    return output_probabilities

# Calculating the new transition probability with new unique token count
def calc_new_transition_prob(in_train_filename, in_tags_filename):
    trans_probabilities = {}
    words = num_words_lower(in_train_filename)
    transition_dict = count_transition_tags(in_train_filename)
    tags_list = tags(in_tags_filename)

    # e.g. {yt-1 = i1:{yt = j1: trans, yt = j2:trans}, yt-1 = i2:{yt = j1: trans, yt = j2:trans}}
    for initial, finals in transition_dict.items():
        for final, count in finals.items():
            numerator = count + DELTA
            denominator = sum(transition_dict[initial].values()) + DELTA * (words + 1) # smoothing
            
            if (initial in trans_probabilities):
                initial_to = trans_probabilities[initial]
                initial_to[final] = numerator / denominator
            
            else:
                trans_probabilities[initial] = {}
                initial_to = trans_probabilities[initial]
                initial_to[final] = numerator / denominator
    
    for i in trans_probabilities:
        for tag in tags_list:
            if (tag not in trans_probabilities[i]):
                num = DELTA
                den = sum(transition_dict[i].values()) + DELTA * (words + 1) # smoothing
                trans_probabilities[i][tag] = num / den
        if (i != "START" and "STOP" not in trans_probabilities[i]):
            num = DELTA
            den = sum(transition_dict[i].values()) + DELTA * (words + 1) # smoothing
            trans_probabilities[i]["STOP"] = num / den

    # Normalise so that row probabilties will sum up to 1
    sum_dict = {}
    for i in trans_probabilities:
        sum_dict[i] = sum(trans_probabilities[i].values())  
    for i, trans_probs in trans_probabilities.items():
        for j, trans_prob in trans_probs.items():
            trans_probabilities[i][j] = trans_prob/sum_dict[i]
    return trans_probabilities

################################### QUESTION 5B #####################################

new_output_probabilities = calc_new_output_prob("twitter_train.txt")
with open('output_probs2.txt', 'w') as f:
    for token, tags_prob in new_output_probabilities.items():
        for tag, prob in tags_prob.items():
            f.write("{} \t {} \t {} \n ".format(tag, token, prob))


new_trans_probabilities = calc_new_transition_prob("twitter_train.txt", "twitter_tags.txt")
with open('trans_probs2.txt', 'w') as f:
    f.write("t \t\t t+1 \t prob \n ")
    for inital, finals in new_trans_probabilities.items():
        for final, prob in finals.items():
            f.write("{} \t {} \t {} \n ".format(inital, final, prob))

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
                     
    tags_list = tags(in_tags_filename)
    output_dict = output_convert_to_dict(in_output_probs_filename)
    trans_dict = trans_convert_to_dict(in_trans_probs_filename)
    viterbi_prediction = []
    testWords = []
    testWords.append(" ")
    with open(in_test_filename) as test_file:
        for word in test_file:
            w = word.strip()
            if (w):
                testWords.append(w.lower())
            else:
                testWords.append(" ")

    pi = {}
    bp = {}
    step = 1
    for num in range(1, len(testWords)):

        # Initial step
        if (testWords[num-1] == " "):
            
            pi[step] = {}
            bp[step] = {}
            
            for tag in tags_list:

                # Meaningful grouping based on `@user_` tagging to `@`
                if (testWords[num][0:6] == "@user_"):
                    if (tag == "@"):
                        pi[step][tag] = trans_dict["START"][tag] * 1
                    else:
                        pi[step][tag] = trans_dict["START"][tag] * 0

                # Meaningful grouping based on `http` and `www.` tagging to `U`
                elif (testWords[num][0:4] == "http" or testWords[num][0:4] == "www."):
                    if (tag == "U"):
                        pi[step][tag] = trans_dict["START"][tag] * 1
                    else:
                        pi[step][tag] = trans_dict["START"][tag] * 0
                else:
                    if (testWords[num] in output_dict):
                        pi[step][tag] = trans_dict["START"][tag] * output_dict[testWords[num]][tag]
                    else:
                        pi[step][tag] = trans_dict["START"][tag] * output_dict["unseen_token_null"][tag]
                bp[step][tag] = 0
            step += 1
        
        # Final step
        elif (testWords[num] == " "):
            temp_dict = {}
            temp_list = []
            prev = step - 1
            for t, tag_prob in pi[prev].items():
                temp_dict[t] = tag_prob * trans_dict[t]["STOP"]
            maxProb = max(temp_dict.values())
            finalBP = max(temp_dict, key=temp_dict.get)
            temp_list.append(finalBP)
            curr_step = max(bp)
            curr_bp = bp[curr_step][finalBP]
            while curr_bp != 0:
                temp_list.append(curr_bp)
                curr_step -= 1
                curr_bp = bp[curr_step][curr_bp]
            temp_list = list(reversed(temp_list))
            viterbi_prediction.extend(temp_list)
            step = 1
            pi = {}
            bp = {}
        
        # Recursion steps
        else:
            pi[step] = {}
            bp[step] = {}
            prev = step - 1
            
            for j in tags_list:
                temp = {}
                for i, prev_prob in pi[prev].items():
                    if (testWords[num][0:6] == "@user_"):
                        if (j == "@"):
                            temp[i] = prev_prob * trans_dict[i][j] * 1
                        else:
                            temp[i] = trans_dict[i][j] * 0
                    elif (testWords[num][0:4] == "http" or testWords[num][0:4] == "www."):
                        if (j == "U"):
                            temp[i] = prev_prob * trans_dict[i][j] * 1
                        else:
                            temp[i] = prev_prob * trans_dict[i][j] * 0
                    else:
                        if (testWords[num] in output_dict):
                            temp[i] = prev_prob * trans_dict[i][j] * output_dict[testWords[num]][j]
                        else:
                            temp[i] = prev_prob * trans_dict[i][j] * output_dict["unseen_token_null"][j]
                pi[step][j] = max(temp.values())
                bp[step][j] = max(temp, key=temp.get)

            step += 1

    with open(out_predictions_filename, "w") as f:
        for prediction in viterbi_prediction:
            f.write(prediction)
            f.write('\n')

################################### QUESTION 5C #####################################
# The accuracy of our improved viterbi is 81.2% (3 s.f.), where the number of correctly 
# predicted tags / number of predictions is 1119/1378 (increase of 83 tags).


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

    trans_probs_filename2 =  f'{ddir}trans_probs2.txt'
    output_probs_filename2 = f'{ddir}output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}') 
    


if __name__ == '__main__':
    run()