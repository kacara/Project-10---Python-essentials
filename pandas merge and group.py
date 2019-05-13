        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 18:31:10 2018

@author: caser
"""

#1 import libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import datetime
#import seaborn as sns
#from scipy import stats 
#from scipy import io as spio
#from sklearn import datasets


#2 create 2 df's
df1 = pd.DataFrame({'A' : [1., np.nan, 3., 5., np.nan],'B' : [np.nan, 2., 3., np.nan, 6.]})
df2 = pd.DataFrame({'A' : [5., 2., 4., np.nan, 3., 7.], 'B' : [np.nan, np.nan, 3., 4., 6., 8.]})
df3 = pd.DataFrame({'C' : [5., 2., 4., np.nan, 3., 7., 8.], 'B' : [2, np.nan, 3., 7., 8., np.nan, 12.]})
df4 = pd.DataFrame({'A' : [1., np.nan, 3., 5., np.nan, 3],'B' : [np.nan, 2., 3., np.nan, 6., 6.], 'C' : [5., 2., 4., np.nan, 3.,  8.]})
df5 = pd.DataFrame({
             'Branch' : 'A A A A A A A B'.split(),
             'Buyer': 'Carl Mark Carl Carl Joe Joe Joe Carl'.split(),
             'Quantity': [1,3,5,1,8,1,9,3],
             'Date' : [
                 datetime.datetime(2013,1,1,13,0),
                 datetime.datetime(2013,1,1,13,5),
                 datetime.datetime(2013,10,1,20,0),
                 datetime.datetime(2013,10,2,10,0),
                 datetime.datetime(2013,10,1,20,0),
                 datetime.datetime(2013,10,2,10,0),
                 datetime.datetime(2013,12,2,12,0),
                 datetime.datetime(2013,12,2,14,0),
                 ]
             })

#contents
#4 merge --> apply mapping and join 2 df's w/ many options including outer, left, etc.
#5 pd.merge_ordered --> merge with optional filling/interpolation designed for ordered data like time series data
#6 map func --> replace df1 values w/ mapped values from a series
#8 'join' is same as merge with index=true
    
#11 concat
#13 append --> 'append' concatenates along axis=0, the index
#15 combine_first function --> keep df1, repalece any missing column or index by df2
#18 drop_duplicates

#20 groupby --> seems more general but requires agg functions
#25 pivot and pivot_table --> seems more visual and can be used w/o agg func


#https://pandas.pydata.org/pandas-docs/stable/merging.html#database-style-dataframe-joining-merging

#        'outer': take the union of the indexes (default)
#        'left': use the calling object’s index
#        'right': use the passed object’s index
#        'inner': intersect the indexes

#pd.concat takes an Iterable as its argument. Hence, it cannot take DataFrames directly as its argument. Also Dimensions of the DataFrame should match along axis while concatenating.
#
#pd.merge can take DataFrames as its argument, and is used to combine two DataFrames with same columns or index, which can't be done with pd.concat since it will show the repeated column in the DataFrame.
#
#Whereas join can be used to join two DataFrames with different indices.


#4 'merge'  applies mapping and joins several columns with many options
pd.merge(df1, df3, on='B', how='inner')


#5 pd.merge_ordered --> merge with optional filling/interpolation designed for ordered data like time series data
pd.merge_ordered(df1, df3, fill_method='ffill', left_by='B')


#6 'map' is very useful as a merge op for linking columns together in 1 df and 1 series 
s = pd.Series(['six', 'seven', 'six', 'seven', 'six'], index=['a', 'b', 'c', 'd', 'e']) #create a series
t = pd.Series({'six' : 6., 'seven' : 7.}) #create another series
s.map(t) #map t onto s, means replace column values in s with mapped values from t
#map alternative by merge for 2 df's
mule1 = pd.merge(df1, df2, on='col1', how='left')

#8 'join' disregards indexes on df's
df1.join(df3, how='outer')
# which is same as merge with index=true
pd.merge(df1, df3, left_index=True, right_index=True, how='outer')



#11 'concat' can also join columns
pd.concat([df1, df3], axis=1, ignore_index=True)


#13 'append' concatenates along axis=0, the index
df1.append([df2, df3], ignore_index=True)


#15 combine_first
df1.combine_first(df3)  #add df3 columns and indexes onto df1 if df1 value is blank; if df1 is not blank, keep df1 value



#18 drop_duplicates 
pd.concat([df1, df2]).drop_duplicates(keep='first')



#20 groupby  --> seems more general but requires agg functions. False cancels indexing
grouped4=df4.groupby('A', as_index=False)  #group according to A, 
grouped4.size()             # show how many sub-group exist
grouped4.head(2)            #show first 2 rows from each group
grouped4.tail(2)            #show last 2 rows from each group
grouped4.nth(0, dropna='any') #check documentation for description
grouped4.describe()         #run some statistics    
grouped4.first()           #show group, remove duplicate indexes after 1st
grouped4.sum()              # apply sum agg funct.
grouped4.get_group(3.0)     # show all duplicate indexes at once for 1 index variable
grouped4.groups             #show all group structure and dtypes
len(grouped4)               #show number of rows
grouped4.ngroup()           #obtain a numeric representation of a series when all that matters is identifying distinctpandas.DataFrame.pivot_table¶ values. 

grouped4=df4.groupby(['B', 'A']) #group in multiple columns like a pivot
grouped4=df4.groupby([pd.Grouper(level=0), 'A']) #group with index and column 
grouped5=df5.groupby([pd.Grouper(freq='1M',key='Date'),'Buyer']).sum() #group by date
TTC_org=df2.groupby(['Approver', 'SSO', 'Name'], as_index=False).size().reset_index(drop=True) #pivot like operation

#group according to a function (should be index value)
df6=df5.set_index('Quantity')
grouped6=df6.groupby( lambda x: x%3)
grouped7=grouped6.filter(lambda x: x['B'].mean() > 3.) #filter out some values from group


df4.groupby(df4.index // 5).std() #take std dev of every 5 rows in order
df4.groupby('A').apply(np.sum)    #can take in 'apply' func.
grouped4.agg(['sum', 'mean'])       #can take agg functions
grouped4.agg({'C': 'sum', 'D': 'std'}) #can take different agg functions per column
grouped4.agg({'B': np.sum, 'C': lambda x: np.std(x, ddof=1)}) #can take different agg functions. lambda function is a transfer function

df4.groupby('A').sum().stack()
df4.groupby('A').sum().unstack()

#iteration of group elements
for name, group in grouped4: 
    print(name) 
    print(group)
    


# melt




#25 pivot and pivot_table --> seems more visual and can be used w/o agg func
mule1 = df1.pivot_table(index=['Quarter', 'Week'], columns='Project',values='Charged_USD')
a= mule1.unstack()

spent = df1.pivot_table(index=['Quarter', 'Week'], columns='Project',
                        values='Charged_USD', aggfunc='sum').reset_index()

df3 = df1.pivot_table(index=['Similars','Fullpath']).reset_index()



