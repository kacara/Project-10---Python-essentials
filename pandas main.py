#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:13:38 2018

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
import os


#2 set the working directory
os.chdir('/media/caser/Warehouse/Home2/Documents/Python/Project 32 - RMD data 1')
#os.getcwd() #check the working dir


#3 import data from csv file 
df=pd.read_csv("Meteorite_Landings.csv", parse_dates=['year'], encoding='UTF-8')


#contents
#3 check data content
#4 data statistics
#5 selecting and filtering
#8 sorting
#10 calculations
#12 applying pipelines, aggregiation and using out-pandas formulas
#13 map function and single element(cell) ops.
#15 indexing
#18 creating dataframes
#20 data types in detail
#21 string methods for Series
#23 definining functions
#24 printing values in print()



#3 check data content
df.shape
df.columns
df.index
df.index.values
#or
df.axes
df.head(15)

df.dtypes
df['name'].dtype #not types
df.get_dtype_counts()   #counts of unique dtypes in this object.
df.values           #print all values in table
df.factorize()  #obtain a numeric representation of a series when all that matters is identifying distinct values. 

#4 data statistics
df.describe()                   #numbers only
df.describe(percentiles=[.05, .25, .75, .997])
df.describe(include='all')      #include strings, etc also
df_filtered = df['recclass'].unique()                      #find uniques
df_filtered = df['recclass'].nunique(axis=0, dropna=True)  #find number of uniques
df_filtered = df['recclass'].duplicated() #return boolean if complete row is duplicate
#alternative
df.drop_duplicates()
df_filtered=df.isnull().any()  #any columns where all values are null #have index option
df_filtered=df.isnull().all()  #any columns where at least one value is null #have index option
df.nlargest(3, 'reclong') #select 10 rows with biggest values in recclass 

df['mass (g)'].idxmin(axis=0)   #show index posn. of min value
df.count()                      #Number of non-NA observations in column
df['mass (g)'].value_counts()   #value counts, 1 column only, different than groupby.count()
#or
df.mode()                       #this one shows only highest count but for all columns

pd.cut(df['reclong'],3)        #cluster into 3 groups (discretization)
#or
pd.cut(df['reclong'],[-165.953,  0,  100,  354.473]) #cluster into 3 groups, define borders (quantiling)
#or
pd.qcut(df['reclong'],[0,  .33,  .66,  1])          #cluster into 3 groups, by percentage,
#or
df.quantile(.8)   #compute values at the given quantile

#5 calling, selecting and filtering 
#5.1 calling specific columns or indexes
df.name     #select column
#or
df['name']  #select column
df[['name','name','recclass']]  #select multiple column as list
df.columns[5]   #select column # 5 as a name
df.loc[3]       #select index 3, put in brackets if text
df.loc[list(range(2,15))]       #select per index in the list


del df['name']  #delete column by name
#or
df.drop(df.columns[5], axis=1, inplace=True) #delete column by order no 

df[:3]          #select index from 0 to 3 excluding 3
df[::4]         #select index no 0 4 8 ...end 
        
df[df.columns[5]] #select column # 5 
df.groupby(df.index // 5).std() #take std dev of every 5 rows in order

df.select_dtypes(include=['number', 'bool'], exclude=['object']) #select by d.type


df['A'] = pd.Series(list(range(len(df))))                  #create a new column for indexing
df_filtered=df[ df['id']>20000 ]                           #simple int filter
df_filtered=df[ df['name'] == 'Aachen' ]                   #simple boolean filter
df_filtered2=df[ (df['id']>20000) & (df['reclong']==0.0) ] #filter by 2 columns
df_filtered3=df.loc[ df['id']>20000 , ['id','recclass'] ]    #filter and select columns
#or by df.query
df_filtered4=df.query('id>20000')
df_filtered5=df.loc[df.column1 >= 5,['column2','column3']] = 555 #if column1 value >= 5 write 555 to column2 and column3

df.loc[5,'id']      #select by index and column name
df.loc[5:15,'id']   #select range of index
df.loc[5:15]=0      #select range of index (simplified) and assign a value
df.loc[5:15,['id','recclass']]  #select range of index, and specific columns

df.iloc[5, 3]            #select by index and column number
df.iloc[(0,2,4),3:7]     #select specific index and column range by tuple
df.iloc[[0,2,4],3:7]     #select specific index and column range by list


#8 sorting
df.sort_index() #sort by index
df.sort_index(axis=1, inplace=True) #sort columns 
df.sort_values(by=['reclat','reclong'],ascending=False,na_position='last',inplace=True)  #sort by values in given column order
df['name'].searchsorted(['c'],side='left') #in a column/series find indices where elements should be inserted to maintain order


#10 calculations
df_x=df.iloc[5:10,:]
df_y=df.iloc[5:10,5:10]
df_z=df_x+df_y #sum values where cells coincide, if not, cell is 'nan'
sum_=df_x.sum() #sum column values, same as axis=0
sum_=df_x.sum(axis=1) #sum index values
sum_=df_x.cumsum() #sum of index values, keeping each index
sum_=df_x.max(axis=1) #max of index values
mean_=df_x.mean()   # mean of column
std_= df_x.std()    #std of column


#12 applying pipelines, aggregiation and using out-pandas formulas
#12.1 pipe
df_x=df.iloc[5:10,:]   #create a simplified df 
(df_x.pipe(np.mean))                 #apply out-pandas formula to a df
(df_x.pipe(np.mean).pipe(np.sum))    #apply 2 out-pandas formula to a df as pipeline

#12.2 'apply' method which is simpler than pipe
df.iloc[5:10,4].apply(np.mean)  #apply out-pandas formula to a column 
df.apply(pd.Series.max)         #apply series method 'max' to each column sperately
df.apply(lambda x: x.max() - x.min(), axis=0) #along column subtract min from max

#12.3 agg - Aggregating with multiple functions
df.agg(['sum', 'mean'])   #New in version 0.20.0.
df.agg(['count', 'mean', 'std', 'min', 'median',  'max'])   #similar to built in 'describe' method   
df.aggregate([np.sum, np.mean, np.std]) #use of np functions

#Function 	Description
#mean() 	Compute mean of groups
#sum() 	    Compute sum of group values
#size() 	Compute group sizes
#count() 	Compute count of group
#std() 	    Standard deviation of groups
#var() 	    Compute variance of groups
#sem() 	    Standard error of the mean of groups
#describe() 	Generates descriptive statistics
#first() 	Compute first of group values
#last() 	Compute last of group values
#nth() 	    Take nth value, or a subset if n is a list
#min() 	    Compute min of group values
#max()      Compute max of group values

#12.4 transform - multiple operations at the same time rather than one-by-one as in 'agg'
df.transform(np.abs)   #New in version 0.20.0.
df.transform([np.abs, lambda x: x+1])
df.transform({'A': np.abs, 'B': lambda x: x+1}) #Passing a dict of functions will allow selective transforming per column.


#13 map function and single element(cell) ops.
f = lambda x: len(str(x))   #define a func
df.applymap(f)              #run each df element through the func.
df['name'].map(f)           #column/series version of 'map'
df['strng length']=df['name'].map(f)  #create a new column with executed data
# 'map' is very useful as a merge op for linking columns together in 2 different df's 
s = pd.Series(['six', 'seven', 'six', 'seven', 'six'], index=['a', 'b', 'c', 'd', 'e']) #create a series
t = pd.Series({'six' : 6., 'seven' : 7.}) #create another series
s.map(t) #map t onto s


#15 indexing
df3=df.loc[5:15,'id']
df3.reindex(list(range(len(df3.index)))) #reindex from 0
df3.reindex(df.index)   #reindex df3 same as index of df
df3.set_index('id')     # set 'id' as index, del old index
df.reset_index(inplace=True) 
df.loc[(1,),] #multiindex filter where first index value=1
df.loc[(1,'A'),'column_']


#18 creating dataframes
#create df from dict
df = pd.DataFrame({'x': [1, 2, 3], 'y': [3, 4, 5]})
#or
dict_ = {'x': [1, 2, 3], 'y': [3, 4, 5]}      #create a dict
df = pd.DataFrame(dict_)
#create df from list
df = [ [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], ]
[[row[i] for row in df] for i in range(4)]
#or
list_=['i', 't', 'e', 'm', 's']  #create a list
df=pd.DataFrame(list(enumerate(list_)))
#or complex one
df = pd.DataFrame({'one' : pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
                   'two' : pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
                   'three' : pd.Series(np.random.randn(3), index=['b', 'c', 'd'])})


#20 data types in detail
#By default integer types are int64 and float types are float64
df_types = pd.DataFrame({'string': list('abc'),
                    'int64': list(range(1, 4)),
                    'uint8': np.arange(3, 6).astype('u1'),
                    'float64': np.arange(4.0, 7.0),
                    'bool': [True, False, True],
                    'dates': pd.date_range('now', periods=3).values,
                    'category': pd.Series(list("ABC")).astype('category')})
df_types['tdeltas'] = df_types.dates.diff()
df_types['uint64'] = np.arange(3, 6).astype('u8')
df_types['tz_aware_dates'] = pd.date_range('20130101', periods=3, tz='US/Eastern')

df.dtypes
df['name'].dtype #not types
df.get_dtype_counts()   #counts of unique dtypes in this object.
df.select_dtypes(include=['number', 'bool'], exclude=['object']) #select by d.type

#changing data type
df2=df[['reclat','reclong']].astype('int64', copy=True, raise_on_error=False) #up to pandas 0.19
df2=df[['reclat','reclong']].astype('int64', copy=True, errors='ignore') #after 0.19

df.infer_objects()  #soft convert to the correct type

m = ['1.1', 2, 3]
pd.to_numeric(m, downcast='integer')    #to numeric as int, if errors='coerce', np.nan

m = ['2016-07-09', datetime.datetime(2016, 3, 2)]
pd.to_datetime(m)    #to_datetime, if errors='coerce', pd.NaT 

m = ['5us', pd.Timedelta('1day')]
pd.to_timedelta(m)    #to_timedelta, if errors='coerce', pd.NaT 
#'apply' can be used to run in all df


#21 string methods for Series
#example
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str[0:2] #extract first 2 letters
s.str.split('_') #split by "_"
s.str.split('_').str.get(0) #get the first part of seperated text


#https://pandas.pydata.org/pandas-docs/stable/text.html#text-string-methods


#23 definining functions
zscore = lambda x: (x - x.mean()) / x.std() #lambda function
#or
def my_func(x):
    zscore=(x - x.mean()) / x.std()
    return zscore

my_func(df.id)  #run id column for defined func.

# if clause with exception
x = 10
if x > 5:
    raise Exception('x should not exceed 5. The value of x was: {}'.format(x))
else
	#blabla

#define a try funct with exception
try:
    import nltk
except ImportError:
    print "you should install nltk before continuing"

# https://realpython.com/python-exceptions/

#24 printing values in print()
# https://www.python-course.eu/python3_formatted_output.php
print("trial for number: %10.3e"% (356.08977)) # % sign will be replaced by text
print("trial for number:/ %10.3e"% (356.08977)) # / sign creates a new line
print("First argument: {0}, second one: {1}".format(47,11)) #new format to replace %