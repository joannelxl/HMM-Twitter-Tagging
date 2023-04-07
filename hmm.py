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
    with open(in_train_filename, encoding="utf-8") as f:
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
    with open(in_train_filename, encoding="utf-8") as f:
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
    with open(in_train_filename, encoding="utf-8") as f:
        for line in f:
            #.strip removes spaces at beginning and end of line
            l = line.strip()
            #if line is not empty
            if l:
                #.split separates words by comma
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
    with open(in_train_filename, encoding="utf-8") as f:
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
    DELTA = 0.1
    words = num_words(in_train_filename)
    #tag and its count, number of times the tag appeared
    tags_dict = count_tags(in_train_filename)
    #tag and no. of times each token is associated w this tag
    tags_tokens_dict = count_tokens_tags(in_train_filename)
    #full set of tags 
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

def calc_output_prob(in_train_filename):
    naive_ouput_probs = calc_naive_output_prob(in_train_filename)
    #print(naive_ouput_probs)
    all_tags = tags("twitter_tags.txt")
    for word in naive_ouput_probs:
        for tag in all_tags:
            #check that word does not have the associated tag
            if (tag not in naive_ouput_probs[word]):
                prob = naive_ouput_probs["unseen_token_null"][tag]
                naive_ouput_probs[word][tag] = prob
    return naive_ouput_probs

#print(calc_output_prob("twitter_train.txt"))


# takes in the naive_output_probs txt file and output a dictionary
# in the following form: 
# {token=x1:{tag=y1:ouput_probability, tag=y2:ouput_probability}, token=x2:{tag=y1:ouput_probability, tag=y2:ouput_probability}}
def convert_to_dict(in_output_probs_filename):
    output_probabilities = {}
    with open(in_output_probs_filename, encoding="utf-8") as probs_file:
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
output_probabilities = calc_naive_output_prob("twitter_train.txt")
with open('naive_output_probs.txt', 'w', encoding="utf-8") as f:
    for token, tags_prob in output_probabilities.items():
        for tag, prob in tags_prob.items():
            f.write("{} \t {} \t {} \n ".format(tag, token, prob))
                    
################################### QUESTION 2B #####################################
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    naive_prediction = []
    naive_output_dict = convert_to_dict(in_output_probs_filename)
    with open(in_test_filename, encoding="utf-8") as test_file:
        for word in test_file:
            token = word.strip()
            if (token):
                if token in naive_output_dict:
                    token_tags = naive_output_dict[token]
                    naive_prediction.append(max(token_tags, key=token_tags.get))       
                else:
                    unseen_token = naive_output_dict["unseen_token_null"]
                    naive_prediction.append(max(unseen_token, key=unseen_token.get))

    with open(out_prediction_filename, "w", encoding="utf-8") as f:
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

    with open(in_test_filename, encoding="utf-8") as test_file:
        for word in test_file:
            token = word.strip()
            if (token):
                if token in naive_output_dict:
                    token_tags = naive_output_dict[token]
                    naive_prediction.append(max(token_tags, key=token_tags.get))
                else:
                    unseen_token = naive_output_dict["unseen_token_null"]
                    naive_prediction.append(max(unseen_token, key=unseen_token.get)) 
   
    with open(out_prediction_filename, "w", encoding="utf-8") as f:
        for prediction in naive_prediction:
            f.write(prediction)
            f.write('\n')                         

################################### QUESTION 3C #####################################
# The accuracy of our prediction is 69.3% (3 s.f.), where the number of correctly 
# predicted tags / number of predictions is 955/1378.



############################# HELPER FUNCTIONS FOR Q4&5 ###############################

#q4a output probabilities
output_probabilities = calc_naive_output_prob("twitter_train.txt")
with open('output_probs.txt', 'w', encoding = "utf-8") as f:
    for token, tags_prob in output_probabilities.items():
        for tag, prob in tags_prob.items():
            f.write("{} \t {} \t {} \n ".format(tag, token, prob))

#q4a transition probabilities
#numerator (count transition tags)

#iterate through the train.txt file, compare current and next
# freq = {transition = {i:j} : count = freq of this transition }

def transition_tags(in_train_filename):
    freq = {}
    tags = []
    startCount = 0
    with open(in_train_filename, encoding = "utf-8") as f:
        for line in f:
            l = line.split()
            if len(l) == 2:
                tag = l[1].strip()
                tags.append(tag)
            else:
                tags.append('END')
                tags.append('START')
                startCount += 1

    #print(tags)
         
    #print(startCount) #Ans:1101    
    #dont consider the last one since its a false 'START'    
    for x in range(len(tags)-1):
        i = tags[x-1]
        j = tags[x]
        temp = (i,j)
        if temp in freq:
            freq[temp] += 1
        else:
            freq[temp] = 1

    freq.pop(("END","START"))
    return freq

#print(transition_tags("twitter_train.txt"))

#freq = {tag i:count}
def count_tag_i(in_train_filename):
    freq = count_tags(in_train_filename)
    freq['START'] = 1101
    return freq
#print(count_tag_i("twitter_train.txt"))

#need to normalise 


def calc_transition_prob(in_train_filename, out_prob_filename):
    #transition_probs = { transition=(i, j} : prob = num }
    transition_probs = {}
    word_count = num_words(in_train_filename)
    transition_dict = transition_tags(in_train_filename)
    tag_count = count_tag_i(in_train_filename)
    all_tags = tags("twitter_tags.txt")
    all_tags.append('START')
    all_tags.append('END')
    DELTA = 0.1
       
    for transition, count in transition_dict.items():
        numerator = count
        denominator = tag_count[transition[0]]
        tag_i = transition[0]
        tag_j = transition[1]
        temp = (tag_i, tag_j)

        numer = numerator + DELTA
        denom = denominator + DELTA * (word_count+1)
        
        trans_prob = numer / denom
        transition_probs[temp] = trans_prob

    dictCopy = transition_probs.copy()

    for transition_tuple, prob in dictCopy.items():
        for i in all_tags:
            start_tag = transition_tuple[0]
            next_tag = transition_tuple[1]
            transition = (start_tag, i)
            #check if start tag --> all other tags exist
            if transition not in transition_probs:
                num = DELTA
                den = tag_count[transition[0]] + DELTA * (word_count + 1)
                prob = num/den
                transition_probs[transition] = prob
    #return transition_probs


    #normalise

    sum_dict = count_transition_sum(transition_probs)
    
    for transition_tuple, prob in transition_probs.items():
        start_tag = transition_tuple[0]
        for i in sum_dict:
            if (start_tag == i):
                prob_before_normalising = transition_probs[transition_tuple]
                prob_after_normalising = prob_before_normalising / sum_dict[i]
                transition_probs[transition_tuple] = prob_after_normalising
    return transition_probs
    
                
            
        

def count_transition_sum(trans_probs):
    #print(trans_probs)
    temp = {}
    for transition, prob in trans_probs.items():
        if transition[0] in temp:
            #print(prob)
            temp[transition[0]] += prob
        else:
            temp[transition[0]] = prob
    return temp

#print(count_transition_sum(calc_transition_prob("twitter_train.txt", "trans_probs.txt")))
            


#q4b initilise viterbi

def calc_viterbi(tags, transition_probs, output_probs, tweet):
    pi_and_bp = {}
    #word num is the step count
    word_num = 1

    for i in range(len(tweet)):
        word = tweet[i]
        for tag in tags:
            #check for first word of tweet
            if (i == 0):
                if (word in output_probs):
                    pi = transition_probs['START', tag] * output_probs[word][tag]
                    if (word_num not in pi_and_bp): 
                        pi_and_bp[word_num] = {}
                        pi_and_bp[word_num][tag] = [pi,'START']
                    else:
                        pi_and_bp[word_num][tag] = [pi,'START']

                else:
                    #if word is not in output_probs
                    pi = transition_probs['START', tag] * output_probs['unseen_token_null'][tag]
                    if (word_num not in pi_and_bp): 
                        pi_and_bp[word_num] = {}
                        pi_and_bp[word_num][tag] = [pi,'START']
                    else:
                        pi_and_bp[word_num][tag] = [pi,'START']


            else:
                #prev pi * transition * output 
                #store all pies, find max, then push into dict
                #so pi_and_bp will alw contain the max pi and backpointer
                #{pi:BP}
                tempDict = {}
                prev = word_num - 1

                for prevTag, pi in pi_and_bp[prev].items():
                    transition = (prevTag, tag)
                    if (word in output_probs):
                        new_pi = pi[0] * transition_probs[transition] * output_probs[word][tag]
                        tempDict[new_pi] = prevTag
                    else:
                        new_pi = pi[0] * transition_probs[transition] * output_probs['unseen_token_null'][tag]
                        tempDict[new_pi] = prevTag

                sorted_dict = sorted(tempDict.items())
                #print(sorted_dict)
                
                max_pi = sorted_dict[-1][0]
                max_bp = sorted_dict[-1][1]
                if (word_num not in pi_and_bp):
                    pi_and_bp[word_num] = {}
                    pi_and_bp[word_num][tag] = [max_pi, max_bp]
                else:
                    pi_and_bp[word_num][tag] = [max_pi, max_bp]
        
        word_num += 1
        

    #last number, all multiply by transition from tag to stop
    #print(pi_and_bp)
    for i in pi_and_bp:
        if (i == len(tweet)):
            for tag in pi_and_bp[i]:
                for max_pi_and_bp in pi_and_bp[i][tag]:
                    #update with new value
                    transition = (tag, 'END')
                    curr_prob = pi_and_bp[i][tag][0]
                    pi_and_bp[i][tag][0] = transition_probs[transition] * curr_prob      

    sorted_pi_and_bp = dict(sorted(pi_and_bp.items(),reverse=True))

    #find max, store BP
    final_max = dict(sorted(sorted_pi_and_bp[len(tweet)].items(), key = lambda item: item[1], reverse = True))
    
    finalBP = next(iter((final_max.items())))[1][1]
    allBackPointers = {len(tweet):finalBP}

    for i in sorted_pi_and_bp:
        for tag, pi_bp in sorted_pi_and_bp[i].items():
            if (tag == allBackPointers[i]):
                newBP = pi_bp[1]
                backPointer = newBP
                #append curr tag to allBackPointers
                allBackPointers[i-1] = backPointer

    sequence = []
    allBackPointers = dict(sorted(allBackPointers.items()))

    for i in allBackPointers:
        if (i != 0):
            sequence.append(allBackPointers[i])

    return(sequence)


################################# QUESTION 4A ################################
transition_probabilities = calc_transition_prob("twitter_train.txt", "trans_probs.txt")
with open('trans_probs.txt', 'w', encoding="utf-8") as f:
    for i_to_j, trans_prob in transition_probabilities.items():
        f.write("{} \t {} \t {} \n ".format(i_to_j[0], i_to_j[1], trans_prob))

output_probabilities = calc_output_prob("twitter_train.txt")
with open('output_probs.txt', 'w', encoding = "utf-8") as f:
    for token, tags_prob in output_probabilities.items():
        for tag, prob in tags_prob.items():
            f.write("{} \t {} \t {} \n ".format(tag, token, prob))

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    output_probs = convert_to_dict(in_output_probs_filename)
    transition_probs = calc_transition_prob("twitter_train.txt", "trans_probs.txt")
    #read the tags
    tags = []
    with open(in_tags_filename, encoding = "utf-8") as f:
        for line in f:
            tags.append(line.strip())

    #read the tweets
    all_tweets = []
    with open(in_test_filename, encoding = "utf-8") as f:
        tweet = []
        for line in f:
            l = line.strip()
            if l:
                tweet.append(l)
            else:
                #once it encounters an empty line,
                #append tweet to all tweets, then empty tweet again
                all_tweets.append(tweet)
                tweet = []

    viterbi_dict = {}
    
    #iterate through all tweets, each time, calculate viterbi for it

    for i in all_tweets:
        temp_seq = calc_viterbi(tags, transition_probs, output_probs, i)
        viterbi_dict[all_tweets.index(i)+1] = temp_seq
    ##return viterbi_dict


    with open(out_predictions_filename, "w") as f:
        for tweet in viterbi_dict:
            for prediction in viterbi_dict[tweet]:
                f.write(prediction)
                f.write('\n')

                    
print(viterbi_predict("twitter_tags.txt", 'trans_probs.txt', 'output_probs.txt', 'twitter_dev_ans.txt',
                      'viterbi_predictions.txt')) 


      
def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    pass




def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename, encoding="utf-8") as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename, encoding="utf-8") as fin:
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

    in_train_filename = 'twitter_train.txt'

    naive_output_probs_filename = 'naive_output_probs.txt'

    in_test_filename = 'twitter_dev_no_tag.txt'
    in_ans_filename  = 'twitter_dev_ans.txt'
    naive_prediction_filename = 'naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')


    naive_prediction_filename2 = 'naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')
    
    trans_probs_filename =  'trans_probs.txt'
    output_probs_filename = 'output_probs.txt'

    
    in_tags_filename = 'twitter_tags.txt'
    viterbi_predictions_filename = 'viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(
        viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    '''
    trans_probs_filename2 = f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}') 
    '''
    
    


if __name__ == '__main__':
    run()
