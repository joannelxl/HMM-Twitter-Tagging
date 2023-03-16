def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    output_probs = open(in_output_probs_filename, encoding = "utf8")
    #print(type(output_probs)) --> io.TextIOWrapper
    out_pred_file = open(out_prediction_filename, encoding = "utf8")
    testData = open(in_test_filename, encoding = "utf8")
    allTweets = []
    oneTweet = []

    for line in testData:
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
        #when empty line is encountered
        else:
            allTweets.append(oneTweet)
            oneTweet = []

    #to handle UnseenToken in output_probs from train set
    UnseenTokensProb = output_probs[output_probs["Tokens"] == "UnseenToken"].setIndex("tag")
    


##    for tweet in allTweets:
##        for token in tweet:
##            
    
            
                
           


naive_predict('naive_output_probs.txt', 'twitter_dev_no_tag.txt', 'naive_predictions.txt')
