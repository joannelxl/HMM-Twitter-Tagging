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

def output_probability(train_file, no_tag_file, output_file):
    smoothing = 0.1
    results = {"token": [], "tag": [], "prob": []}
    numWords = num_words(no_tag_file)
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
