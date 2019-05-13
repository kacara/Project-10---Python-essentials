#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 22:16:42 2018

@author: caser
"""

#1 import libraries
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#import datetime
import seaborn as sns
sns.set()
#from scipy import stats 
#from scipy import io as spio
#from sklearn import datasets

#2 import data from csv file 
df1=pd.read_csv("Meteorite_Landings.csv", parse_dates=['year'], encoding='UTF-8')
df2 = pd.DataFrame(np.random.randn(100, 4), index=pd.date_range('1/1/2000', periods=100), columns=list('ABCD'))
df2=df2.cumsum().cumsum()


#contents
#1 pandas plotting  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
#2 matplotlib pyplot
#3 seaborn




#1 pandas plotting
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
plt.figure() 
df2.plot()  #plot each numerical column agains index on 1 chart
df[['reclat','reclong']].boxplot() #boxplot of a df
df2.plot.scatter(x='A', y='B', ylim=(0,500)) #plot x vs y as scatter where y_axis limit is 500 


#2 matplotlib pyplot

#2.1 simplest plot
plt.plot(df2['B'],df2['A']) #plot x=B and y=A
plt.plot(df2['A'])  #plot x=index and y=A

#2.1.2 single figure single plot
fig = plt.figure(figsize=(9,15))
plt.title( 'T48SPR' + ' vs ' + '0_date' + ' scatter' ) 
plt.ylabel('T48SPR')
plt.xlabel('0_date')
plt.grid(True)
plt.minorticks_on()
plt.plot_date('0_date', 'T48SPRD_VAL0', data=df3, xdate=True, linestyle ='solid', linewidth=1, markersize=2.5)
plt.savefig('single scatter plots.pdf', facecolor="0.9" )


#2.2 complex plot
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html
fig1 = plt.figure(figsize=(8, 12))
plt.suptitle('Categorical Plotting') #main title, figure
plt.title('Easy as 1, 2, 3') #sub title, axes
plt.ylabel('y axis') # and plt.xlabel('x axis')
plt.axis([0, 6, 0, 20]) #set x axis from 0 to 6
#or
plt.xlim(0, 6)  #  set the x limits of the current axes.
plt.xticks(np.arange(0, 1, step=0.2)) #set the current tick locations and labels of the x-axis.
plt.minorticks_on() # or plt.minorticks_off()
plt.margins(0.2) # Tweak spacing to prevent clipping of tick-labels. The margin must be a float in the range [0, 1].
plt.grid(True)
plt.setp(ax,yticks=[0,5])
plt.text(60, .025, 'text here') #print text at positin of x vs y variable posn.
plt.legend('best') #check other posn. https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend

plt.plot(df2['A'],df2['B']) #or plt.plot('A', 'B', data=df2)  with blue squares 
#or
plt.plot_date(x, y, xdate=True) # plot date 
plt.plot( df1['0_date'], df1['2_MWSEL_VAL0'], 'r--', df1['0_date'], df1['3_N25SEL_VAL0'], 'b--') #plot 2 lines on 1 axes
plt.scatter('T48SPRD_VAL0', val, data=df1, alpha=0.5)
plt.hist(val, data=df3, ec='white')

#or
a=sns.relplot(x='A', y='B', hue="C", legend="brief", data=df2, kind='scatter')
a.set_xticklabels(rotation=45)
a.set_axis_labels("hede",'hodo')
a.set(xlim=(0,5), ylim=(0,5), xticks=[0,2.5,5], yticks=[0,2.5,5])

plt.show()
#or
plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)


#2.3 subplots
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(1, figsize=(9, 3))
plt.subplot(131) #first is columns and 2nd is row then index(posn.)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')

plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None) # Automatically adjust subplot parameters to give specified padding.
#or
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()

#or
plt.subplot2grid((3, 2), (0, 0)) #more definitive positioning

#2.4 line properties
# https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
plt.plot(names, values, linewidth=2.0)
#or
lines = plt.plot(names, values)
# then
plt.setp(lines, color='red', linewidth=2.0, )

#2.5 logarithmic
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

plt.figure(1)

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear') #set the scaling of y axis
plt.title('linear')
plt.grid(True)


# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)


# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

#2.6 close
plt.cla() #Clear an axis
plt.clf() #Clear an entire figure
plt.close() #Close a window


#2.7 loop example from RM&D case
fig = plt.figure(figsize=(8, 12))

mule=1 #edit # in this line
for i in df1.columns[2:5]: #edit # in this line
    plt.subplot(3,1,mule) #edit # in this line
    plt.title( str(i) + ' vs time\n max = ' + str(report_of_max.loc[i,'peak value'].round(2)) + ' at ' + str(report_of_max.loc[i,'date']) ) 
    plt.ylabel(i)
    plt.xlabel('date')
    plt.grid(True)
    plt.minorticks_on()
    #plt.text(df1.loc[df1_shape[0]//2,'0_date'], 10, 'text here')
    #plt.legend('best')
    plt.plot_date('0_date', i, data=df1, xdate=True, linestyle ='solid', linewidth=1, markersize=2.5)
    mule=mule+1

plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.savefig('time plots.pdf', papertype='letter', orientation='portrait',  facecolor="0.9" )


#2.8 multiple page pdf
from matplotlib.backends.backend_pgf import PdfPages
import matplotlib.pyplot as plt

with PdfPages('multipage.pdf') as pdf:
    # page 1
    plt.plot([2, 1, 3])
    pdf.savefig()

    # page 2
    plt.cla()
    plt.plot([3, 1, 2])
    pdf.savefig()

#3 seaborn
#https://seaborn.pydatarelplot.org/generated/seaborn.relplot.html#seaborn.relplot

#relplot, line or scatter
sns.relplot(x='A', y='B', hue="C", legend="brief", data=df2, kind='scatter')

#categorical plots 
#https://seaborn.pydata.org/generated/seaborn.catplot.html#seaborn.catplot
#https://seaborn.pydata.org/tutorial/categorical.html#categorical-tutorial
df2['E']=pd.cut(df2['D'],4)   #create 4 categories
sns.catplot(x='E', y='B', data=df2, kind='bar')
              
#create regression line
sns.lmplot(x="A", y="B", data=df2, order=2)

#create 4 different plots by col and col_wrap
sns.relplot(x='A', y='B', col="E", col_wrap=2, kind='line', data=df2)

#saving
a=sns.relplot(x='A', y='B', hue="C", legend="brief", data=df2, kind='scatter')
a.savefig('svm_conf.png')
