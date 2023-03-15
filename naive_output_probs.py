import json
def tags(file):
    tags_list = []
    with open(file) as f:
        for tag in f:
            tags_list.append(tag.strip())
    return tags_list
#returns a dictionary where count the frequency of tags
#e.g. {tag = y1: count, tag = y2: count}
def count_tags(file):
    freqs = {}
    with open(file) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = []
                for word in line.split():
                    temp_list.append(word)
            tag = temp_list[1]

            if tag in freqs:
                freqs[tag] += 1
            else:
                freqs[tag] = 1
    return freqs

#returns a dictionary of dictionaries where count the number of token w associated with j 
#e.g. {tag = y1:{token = x1: count, token = x2: count}, tag = y2:{token = x1: count, token = x2: count}}
def count_tokens_tags(file):
    freqs = {}
    with open(file) as f:
        for line in f:
            l = line.strip()
            if l:
                temp_list = []
                for word in line.split():
                    temp_list.append(word)
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

def num_words(file):
    freqs = {}
    with open(file) as f:
        #loop through every tag
        for line in f:
            tag = line.strip()
            #if word do not exist in dict add 1 to word count
            if tag not in freqs:
                freqs[tag] = 1
    return sum(freqs.values())

def calc_output_prob(tokens_file, tokens_tags_file):
    output_probabilities = {}
    sigma = 1
    words = num_words(tokens_file)
    tags_dict = count_tags(tokens_tags_file)
    tags_tokens_dict = count_tokens_tags(tokens_tags_file)
    tags_list = tags("twitter_tags.txt")

    output_probabilities["unseen token"] = {}
    for tag in tags_list:
        tag_count = tags_dict[tag]
        num = sigma
        den = tag_count + sigma * (words + 1)
        output_probabilities["unseen token"][tag] = num/den
    
    for tag, tags_count in tags_dict.items():
        tags_tokens = tags_tokens_dict[tag]
        for token, tokens_count in tags_tokens.items():
            numerator = tokens_count + sigma
            denominator = tags_count + sigma * (words + 1)

            if token in output_probabilities:
                token_prob_tag = output_probabilities[token]
                token_prob_tag[tag] = numerator/denominator
            else:
                output_probabilities[token] = {}
                token_prob_tag = output_probabilities[token]
                token_prob_tag[tag] = numerator/denominator
            #output_probabilities[(tag, token)] = numerator/denominator

    return output_probabilities

output_prob_dict = calc_output_prob("twitter_train_no_tag.txt", "twitter_train.txt")

with open('naive_output_probs.txt', 'w') as f:
    f.write(json.dumps(output_prob_dict))
