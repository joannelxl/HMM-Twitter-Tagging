import csv
import pandas as pd

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    outputprobs = open(in_output_probs_filename, encoding = "utf8")
    output_probs = pd.read_csv(outputprobs)
    #print(type(output_probs)) --> io.TextIOWrapper
    out_pred_file = open(out_prediction_filename, encoding = "utf8")
    testData = open(in_test_filename, encoding = "utf8")

    allTweets = []
    oneTweet = []

    for line in testData:
        print(line)
        #check that line is not empty
        #line contains a token
        if line:
            #change everyth to lower case
            word = line.lower()
            if '@user' in word:
                word = '@user'
            elif 'http' in word:
                word = 'http'
            word = word
            oneTweet.append(word)
            #print(oneTweet)
        #when empty line is encountered
        else:
            print("it came here")
            allTweets.append(oneTweet)
            print(allTweets)
            oneTweet = []

    #print(oneTweet)

    #to handle UnseenToken in output_probs from train set
    UnseenTokensProb = output_probs[output_probs["token"] == "UnseenToken"].drop("token", axis = 1).set_index("tag")


    for tweet in allTweets:
        for token in tweet:
            allTags = UnseenTokensProb.copy()

            probFromTrain = output_probs[output_probs.token==token].set_index("tag")
            allTags.update(probFromTrain)
            #highest likelihood
            maxTag = allTags.idxmax().prob
            f.write(maxTag + "\n")
        #add line after every tweet
        f.write("\n")
    
            
                
           


print(naive_predict('naive_output_probs.txt', 'twitter_dev_no_tag.txt', 'naive_predictions.txt'))
