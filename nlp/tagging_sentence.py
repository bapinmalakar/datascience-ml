#solved tagging problem
#tagging is process to label each word with pos tag based on the context of the sentence
#this process is very complicated

#some time tagging is not possible because same word can have different different meaning based on sentence(Ambiguity).
from nltk import pos_tag, sent_tokenize, word_tokenize

text = "Hello Guru99, You have to build a very good site, and I love visiting your   site."

#pos_tag need of list of word, means we need to do tokenize

#breake paragrapg sentence wise
sentence = sent_tokenize(text)

for sen in sentence:
    print(pos_tag(word_tokenize(sen)))

#in corups two type pos-tagger present
# 1. rule-based pos tagger: for ambiguous words, this technique do tagging based on the meaning or context of the information
# applied or present. For that it will check and analyze preceding words following words. So it will tagged the based on
#grammatical rules of a language

# 2. Stochastic pos taggers: here frequency or probability applied. Here word tag depend on also previous tagged
# It will calculate the probability of frequency of taggs in the sentence. Tagg selected based on higest probabilty
# of a particular word tagg

#Hidden Markov model
#tagging problem can be modeled by HMM
#Goal of HMM is find hidden state sequence
# Here tokend are Obserable sequece and tags are hiddn states
#HMM perform join distribution P(x,y), x is token sequence and y is tag sequence