#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 11:13:11 2018

@author: caser
"""

#1 importnumpy and pandas
import numpy as np
import pandas as pd
import datetime as dt
#import seaborn as sns

#2 import data from csv file 
df=pd.read_csv("Meteorite_Landings.csv", parse_dates=['year'], encoding='UTF-8')

# create date
pd.date_range(start='2018-04-24', end='2018-04-27', periods=3)  #create range
pd.date_range(start='1/1/2018', periods=5, freq='3M')           #create range with 3M frq,
#or
pd.to_datetime([1, 2, 3], unit='D', origin=pd.Timestamp('1960-01-01'))
dt.datetime.now()         #current time
pd.to_datetime(['2005/11/23', '2010.12.31']) #convert to date

# date operations
df1['1_period']=df1['1_period'].dt.total_seconds() #convert timedelta to seconds