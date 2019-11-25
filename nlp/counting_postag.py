#count pos_tags are tagged in the words
#Its useful for text classification and then decide your operating on different classes

from collections import Counter
import nltk

textIs = "Guru99 is one of the best sites to learn WEB, SAP, Ethical Hacking and much more online."

lowercaseText = textIs.lower()

regexTokenizer = nltk.RegexpTokenizer(r'\w+')

tokensAre = regexTokenizer.tokenize(textIs)

tokenWithTags = nltk.pos_tag(tokensAre)

counts = Counter(tag for word, tag in tokenWithTags)

print(counts)

#find number of time a word occuring in document. Done by FreqDistclass, define in nltk.probability module
#Count Method: freq_dist.count(word), return number of time given word appear
#Frequency Method: freq_dist.freq(word), return frequency of given word 
paraGraph = "A paragraph is a self-contained unit of a discourse in writing dealing with a particular point or idea. A paragraph consists of one or more sentences. Though not required by the syntax of any language, paragraphs are usually an expected part of formal writing, used to organize longer prose"

frequencyOfWords = nltk.FreqDist(nltk.word_tokenize(paraGraph))
print('\n\n', frequencyOfWords.most_common())

for wd in frequencyOfWords:
    print("\n\nFrequency of word {} is {}".format(wd,frequencyOfWords.freq(wd) ))

# frequencyOfWords.plot() #for grapg representation of frequency

#Collaction
# are pair of words occuring together many time in document like CT Scan, Infrared Ray etc. these type word occure together
# So collaction is require for finding the frequency of words
# Collaction are two type:
# Bigrams: Collaction of two words
# Trigrams: Collaction of three words

tokensOfPara = nltk.word_tokenize(paraGraph)
biGramAre = nltk.bigrams(tokensOfPara)

print(list(biGramAre))

triGrams = nltk.trigrams(tokensOfPara)



