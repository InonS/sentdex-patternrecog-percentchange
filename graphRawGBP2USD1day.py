'''
@file graphRawGBP2USD1day.py
@author: Inon Sharony
@date Oct 28, 2014
'''

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def graphRawGBP2USD1day():
    
    print 'start of graphRawGBP2USD1day()'
    
    print 'loading csv text data from file into multidimensional array (stripping date to raw number format)'
    
    date, bid, ask = np.loadtxt(
                                '/home/Slava/workspace/SentdexTutorial/src/AlgoTradingTutorial/GBPUSD1d.txt',    # file name                                
                                delimiter=',',  # text data file delimiter 
                                converters={0:mdates.strpdate2num('%Y%m%d%H%M%S')}, # convert column 0 using matplotlib.dates "strip date to number" with the given format
                                unpack=True
                                )
    
    print 'len(date) = ',len(date),' (number of data entries)'
    
    #fig = plt.figure(figsize=(10,7)) # unused variable 'fig' ?
    
    # this was the original command in the video. subplot2grid isn't recognized...
    ''' @bug here ''''''ax1 = plt.subplot2grid(40,40
                           (40,40),
                           (0,0),
                           rowspan=40, 
                           colspan=40)'''
      
    #ax1 = plt.subplot(2,1,1)    
    #ax1.plot(date,bid)
    
    #ax1 = plt.subplot(2,1,2) 
    #ax1.plot(date,ask)

    ax1 = plt.subplot(1,1,1)
    ax1.plot(date,bid,'b',date,ask,'r')
    
    plt.title('GBP to USD for May 1, 2013')
    plt.xlabel('time')
    plt.ylabel('price')

    print 'setting y-axis offset (intercept)'
    ''' @bug here ''' # plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    #plt.gca().set_ybound(lower=0, upper=ax1.plot.get_ylim()[2])
    plt.subplots_adjust(bottom=.23)
    
    print 'setting x-axis labels format\n(since data is for a single day, the date part is somewhat of an overkill...)'
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    
    print 'setting x-axis labels rotation'
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
                    
    print 'twinning x-axis values with the previous plot' 
    ax1_2 = ax1.twinx()
    
    print 'creating plot for bid-ask spread and formatting it'
    ax1_2.fill_between(
                       date,            # x
                       0,               # y1 (from): x-axis
                       (ask-bid),       # y2 (to): bid-ask spread
                       facecolor='g',   # format: green
                       alpha=.3)        # format: alpha
    
    print 'setting grid'
    plt.grid(True)

    print 'showing plot'
    plt.show()
    
    print 'end of graphRawGBP2USD1day()'
# end of graphRawGBP2USD1day()
        
print 'calling graphRawGBP2USD1day():' 
graphRawGBP2USD1day()
print '# program termination'
