# finding synonyms for words
# WordNet is corpus reader for NLTK, a lexical database for English.
# Used to find synonym and antonym of words

#wordnet used in major search engine

# synonym (words having the same meaning)
# hypernyms (The generic term used to designate a class of specifics (i.e., meal is a breakfast), hyponyms (rice is a meal)
# holonyms (proteins, carbohydrates are part of meal)
# meronyms (meal is part of daily food intake)
# with the help of WordNet we can create  spelling checking, language translation, Spam detection and many more

from nltk.corpus import wordnet

print(dir(wordnet))  # list all features of wordnet

# find synonum of a word, give colletion of synonym words
syns = wordnet.synsets('dog')
print(syns)

synonyms = []
antonyms = []

for syn in wordnet.synsets("dog"):
    for l in syn.lemmas():
        print('\n', l.name())
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))
