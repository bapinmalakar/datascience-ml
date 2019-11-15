#sentence is collection  of words and each words denote prats of speech
#here we define parts of speech tag to each words

#Input: Everything to permit us.
#Output: [('Everything', NN),('to', TO), ('permit', VB), ('us', PRP)]

#steps
#1. tokenize: make sentence into words (word_tokenize)
#2. apply pos_tag in the output of step-1(pos_tag(tokenize_text))


# Abbreviation	Meaning
# CC	        coordinating conjunction
# CD	        cardinal digit
# DT	        determiner
# EX	        existential there
# FW	        foreign word
# IN	        preposition/subordinating conjunction
# JJ	        adjective (large)
# JJR	        adjective, comparative (larger)
# JJS	        adjective, superlative (largest)
# LS	        list market
# MD	        modal (could, will)
# NN	        noun, singular (cat, tree)
# NNS	        noun plural (desks)
# NNP	        proper noun, singular (sarah)
# NNPS	        proper noun, plural (indians or americans)
# PDT	        predeterminer (all, both, half)
# POS	        possessive ending (parent\ 's)
# PRP	        personal pronoun (hers, herself, him,himself)
# PRP$	        possessive pronoun (her, his, mine, my, our )
# RB	        adverb (occasionally, swiftly)
# RBR	        adverb, comparative (greater)
# RBS	        adverb, superlative (biggest)
# RP	        particle (about)
# TO	        infinite marker (to)
# UH	        interjection (goodbye)
# VB	        verb (ask)
# VBG	        verb gerund (judging)
# VBD	        verb past tense (pleaded)
# VBN	        verb past participle (reunified)
# VBP	        verb, present tense not 3rd person singular(wrap)
# VBZ	        verb, present tense with 3rd person singular (bases)
# WDT	        wh-determiner (that, what)
# WP	        wh- pronoun (who)
# WRB	        wh- adverb (how)

#chunking used, to give more structure to sentence by following part of speech(POS) tagging.
#Also known as Shallow parsing
#resulting group of word is known as chunks
#In shallow parsing maximum one level between root to leaf
#Shallow parsing also called Light parsing and chunking
#primary usage is make group of Noun Phrases

#ruls
#no predidifned ruls, we can use according to our requirement
# exam grouping: noun, verb, adjective and coordinating junction
#chunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}
#., any character except new line
#*, match 0 or more occurabce
#?, Match 0 or 1 repetitions

from nltk import pos_tag #part of spech tagging
from nltk import RegexpParser
from nltk.tokenize import RegexpTokenizer

smaple_text = "learning NLP now and continue learing untill NLP is not finish!!"
regex_tokenizer = RegexpTokenizer(r'\w+')
wordsAre = regex_tokenizer.tokenize(smaple_text)

print('Words are:\n', wordsAre)

assign_tags = pos_tag(wordsAre)

print('Words with taggs\n', assign_tags)

pattern = """mychunk:{<NNP.?>*<VBG.?>*<VBZ>?}"""

chunker = RegexpParser(pattern)

print('After regex: ', chunker)

output = chunker.parse(assign_tags)

print('After chunking ', output)