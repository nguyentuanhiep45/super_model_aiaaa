from random import choices, randint

with open("wordlist.txt") as f:
    word_list = f.readlines()

def random_sentence():
    return ' '.join(map(lambda w : w[0:-1], choices(word_list, k = randint(1, 10))))