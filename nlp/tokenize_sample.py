#tokenize is the process of dividing large quantity of text in small parts, called tokens

#NLP primary usecase
#1. text classification
#2. intelligent chatboat
#3. sentimental analysis
#4. language translation

#tokens are used to finding pattern and its a base step
#tokenize can classified in 2 sub-modules
#1. Word tokenize: word_tokenize() for split a sentence into words, it doesnot remove punchation
#2. Sentence tokenize: sent_tokenize() for split text into sentence

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

sample_text = 'A paragraph is a self-contained unit of a discourse in writing dealing with a particular point or idea. A paragraph consists of one or more sentences! though not required by the syntax of any language, paragraphs are usually an expected part of formal writing, used to organize longer prose.'

sentenceAre = sent_tokenize(sample_text)

for sen in sentenceAre:
    print('Sentence is: \n', sen)
    tokensAre = word_tokenize(sen) #not remove punchuation
    print('\nWords are: \n', tokensAre)
    tokenizerRegex = RegexpTokenizer(r'\w+')
    wordsAfterRemovePunchuation = tokenizerRegex.tokenize(sen) #remove punctuation and generate words
    print('Now wrods are:: \n', wordsAfterRemovePunchuation)