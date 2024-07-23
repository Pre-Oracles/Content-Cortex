#Normalised - check out end points
# Data structures 
#check data - finding missing value
#parameter 
#Errow/skew in code

#Json -> python script to normalise -> csv
#MongoDB, Django

import pandas as pd
tb = pd.read_csv("example.csv")

#changing and identifying column names/normalise end points
tb.rename(columns=  {"ip":"IP"}, inplace=True)
print(tb.columns) 

#removing null entities
print(tb.isnull().sum())
tb.dropna(how ='any',axis=0)

#renaming exisitng code
tb.loc[tb.Department=='footwear','Department'] = 'shoes'

#Normalize the data? Normal distribution
tb['score'] = (tb['score'] - tb['score'].min()) / (tb['score'].max() - tb['score'].min())