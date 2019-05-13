#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 20:01:30 2018

@author: caser
"""


#1 import numpy and pandas
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import datetime
#import seaborn as sns
#from scipy import stats 
#from scipy import io as spio
#from sklearn import datasets

#2 import data from csv file 
df=pd.read_csv("Meteorite_Landings.csv", parse_dates=['year'], encoding='UTF-8')

#data structures - list (use during filtering etc.)
list_=['i', 't', 'e', 'm', 's']   #list is in brackets
list2_=list('items of a list and some number 6415611869')    #convert to list
list3_=list(range(10))                                       #convert to list alternative
list4_=list2_+list3_                                         #can be nested
boolean_result='i' in list_       #boolean type check
type(boolean_result)              #check data type of a variable
list2_[2:10:3]                    #slice range by steps of 2 
len(list2_)    
max(list2_)  
list2_.count('i')                 #number of occurrences of 'i'

my_list = ['apple', 'banana', 'grapes', 'pear']
counter_list = list(enumerate(my_list, 1)) #enumerate returns index and value
print(counter_list)
# Output: [(1, 'apple'), (2, 'banana'), (3, 'grapes'), (4, 'pear')]
#or
for i, color in enumerate(colors): #enumerate usage in a loop 
    ...

list2_.sort()                     #sort contents and revise variable order in place 
list_.append('x')                 #add 'x' to end of list   
list2_.insert(5, 'x')             #add 'x' to 5th posn.
list_.remove('x')                 #remove all 'x' from list   
list2_.pop(3)                     #remove 3rd item from list  
del list2_[2:4]                   #remove a range from list 


#data structures - dict (use during filtering etc.)
dict_ = {'x': [1, 2, 3], 'y': [3, 4, 5]}      #create a dict method1
dict_ = dict(sape=4139, guido=4127, jack=4098)  #create a dict method2.1
dict_ = dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])   #create a dict method2.2
dict_ = {x: x**2 for x in (2, 4, 6)}   #create a dict method3
dict_['guido'] = 4127      #add a new key and value
del dict_ ['sape']         #delete key 
list(dict_)                #list keys
sorted(dict_)              #sort in place
dict_.items()              #list all elements

#lists and dictionaries are mutable , meaning you can change their content without changing their identity. Other objects like integers, floats, strings and tuples are objects that can not be changed.


#data structures - tuple (use during filtering etc.)
t = 12345, 54321, 'hello!'      #separated by commas and no paranthesis
u = t, (1, 2, 3, 4, 5)          #can be nested
u[1]                            #find by index no


#data structures - pd.series() (use during filtering etc.)
series_ = pd.Series(list_)  #create series
series_ = pd.Series(list('abcde'), index=[0,3,2,5,4])  #create series 2
series_.sort_index()   #sort index and values
series_.sort_index().loc[1:6]   #sort index and values in range 

#Boolean ops
(df > 0).all()              #check all column values if all >0 , true
(df.iloc[:,1:5]>0).all()    #check only specific columns and indx range
(df > 0).any()              #check all column values if any >0 , true
pd.Index(['foo', 'bar', 'baz']) == 'foo' #check which indexes are foo


# deleting variables
del list_