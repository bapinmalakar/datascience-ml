from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

print('Setup done!!')

#our input folder is stories, which have n number of text docume

#process for taking all file path and save for process
title = "./../input/stories"
alpha = 0.3

folders = [x[0] for x in os.walk(title)]
folders[0] = folders[0][:len(folders[0])-1]

print(folders)