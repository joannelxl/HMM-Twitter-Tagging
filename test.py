def function():
    file = open('twitter_train.txt')
    #f = open(in_output_probs_filename, "w")
    pair = file.readline()
    token_tag_counter_dict = {}
    tag_counter_dict = {}
    while pair:
        newpair = pair.split()
        token = newpair[0]
        tag = newpair[1]
        if (token, tag) in token_tag_counter_dict.keys():
            count = token_tag_counter_dict[(token, tag)]
            token_tag_counter_dict.update({(token, tag): count + 1})
        else:
            token_tag_counter_dict[(token, tag)] = 1
        if tag in tag_counter_dict.keys():
            count = tag_counter_dict[tag]
            tag_counter_dict.update({tag: count + 1})
        else:
            tag_counter_dict[tag] = 1
        pair = file.readline()
        if pair == '\n':
            pair = file.readline()
    print("tag_counter_dict: ")
    print(tag_counter_dict)
    print("token_tag_counter_dict: ")
    print(token_tag_counter_dict)

print(function())
