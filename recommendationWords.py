import fasttext
import pandas as pd
import os

os.system("wget -O model_dictionary_doYouMean.bin --no-check-certificate \'https://drive.google.com/uc?export=download&id=1WElNnOuACT7cCJaNsZoUieTFqRZpevyi\'")

m=fasttext.load_model("model_dictionary_doYouMean.bin")

#input text
#texts='''
#
#
#'''

with open("articles.txt") as f:
    texts=f.read()

texts=texts.replace('\n',''). replace ('\r','')

label,prob=m.predict(texts,k=53)



doYouOffering=[]

doYouOffering = pd.DataFrame({'interests':label})

doYouOffering['probability'] = pd.DataFrame({'probability':prob})


# clearance zones prefix and suffix 
doYouOffering['interests'] = doYouOffering['interests'].str.replace('__label__','')

doYouOffering['interests'] = doYouOffering['interests'].str.replace(',"To','')

print (doYouOffering)
