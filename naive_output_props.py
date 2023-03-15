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
    uniqueWords = []

    #line is type string
    for line in trainData:
        #line is actly the word!! since got no tag
        if line not in uniqueWords:
            uniqueWords.append(line)
        else:
            pass

    return len(uniqueWords) #ans: 5764         

#print(num_words('twitter_train_no_tag.txt'))


