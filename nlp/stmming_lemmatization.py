# stemming
# Kind of normalization for words
# Normalization is technique where set of words in a sentence are converted into sequence of words based on same meaning but
# have some variation according to the context
# means we can find root word of any variation
# like eat, eats, eation are variation of Eat(root word)
# simply we can say stemming use to grouping of same words by finding root word
# use PorterStemmer function

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

words_variation = ['wait', 'waitting', 'waiting', 'waited', 'waits']

ps = PorterStemmer()

for wd in words_variation:
    root_word = ps.stem(wd)
    print('Root wrod ', wd, ' is ', root_word)

# so using stemming we remove redundancy in the sentence

# exaple find root word for each word present in sentence
sentence = "Hello Guru99, You have to build a very good site and I love visiting your site."
regexTokenizer = RegexpTokenizer(r'\w+') # for make tokens from the sentence
all_words = regexTokenizer.tokenize(sentence)
print('Words are: ', all_words)

unique_words = {}

for wd in all_words:
    rootWord = ps.stem(wd)
    if(rootWord in unique_words):
        unique_words[rootWord] = unique_words[rootWord].append(wd)
    else:
        print('else execute: ', wd)
        unique_words[rootWord] = [wd]

print('root words details\n')
print(unique_words)

#lemmatization
#algorithm process to find lemma of a word depending their meaning
#used for morphological analysis
#lemmatization help to return base and dictionary form of a word, which known as leema

#Lemmatization and Stemming are not same
# Stemming works by removing siffix or prefiix of a word to find root word
# Other hand lemmatization try to find morphological words to find its base and dictionary form
# WordNetLemmatizer function

text2 = "studies studying cries cry"
text2_token = regexTokenizer.tokenize(text2)
wordNetLemmatization = WordNetLemmatizer()

for wd in text2_token:
    rootWord = ps.stem(wd)
    lemmaIs = wordNetLemmatization.lemmatize(wd)
    print("Root word  of {} is {}".format(wd, rootWord))
    print("Lemmatize of word {} is {}".format(wd, lemmaIs))
