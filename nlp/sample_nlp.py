from nltk.tokenize import RegexpTokenizer
#RegexpTokenizer
# remove all expression, symbol, character, numeric or any things whatever you want
tokenize = RegexpTokenizer(r'\w+')
#RegexpTokenizer(r'\w+') passed regular expression to RegexpTokenizer
sentenceIs = "Hello Guru99, You have build a very good site and I love visiting your site."

fliterText = tokenize.tokenize(sentenceIs)
#get token from the sentence


print(fliterText)