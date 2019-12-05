"""
@file PercentChange.py
@author: Inon Sharony
@date Oct 29, 2014

Goal: Pattern matching (Machine Classification) to find Technical Analysis
graph patterns.

Assumption of Technical Analysis: Graph patterns supposedly reflect trends in
the traders' beliefs, and therefore yield

Hypothesis: The re-appearance of a certain series of signals or variables
(a pattern) is a predictor of the following terms in the series,
or the outcome.

Objectives:
1. Run a batch of algorithms on past data, testing each against past outcomes,
and projecting this insight into the future.
In the context of Machine Learning, making inferences is
termed "data snooping".

2. Test the underlying assumption: Past pattern recognition will yield success
with future data.


Explicit variables:

1. by what way is a pattern recognized?
or, how is similarity calculated? percentChange()

Percent change:
* Starting at some point in time, measure the changes in price
relative to that at the initial point.
* If changes are consistently in the same direction (increasing/decreasing)
a pattern is identified.
* Consistent changes are considered reinforcing of the pattern.
* This is a-symmetric, and places greatest weight on starting points,
so could be argued to have less predictive power into the future.

Reverse percent change:
* Relate the price at the latest price to prices which preceded it.

Other alternatives:
* Start-point to end-point
* Sequential point-to-point

2. What is the required similarity to be even considered "a match"?
    lowerBoundSimilarity

3. What is the searched pattern length? patternMatchLength (e.g. 10, 1000, ...)

4. How far back in history do we search for patterns?
are all historical data given equal importance (data freshness)?

5. How far forward into the future are we trying to predict?
From 20 ticks after the last tick of the pattern to 30 ticks.
The predictive power need not project too far into the future, but just long
enough for an action to be taken and completed in the time between pattern
recognition and the time when the expected outcome is supposed to be realized
(usually a few milliseconds).
This, amongst others, can be machine learned...

6. How exact must similarity be between predicted and actual results be,
in order for the prediction to be even considered "successful"? (~70%-80%)

7. THE BIG ONE: how precise do we want our forecasts to be?
is a 74% match so much worse than a 75% that we drop any and all such matches?
Machine Learn it!


Implicit variables:

8. Opportunity vs. accuracy:
Optimize area-under-curve (AUC) of chances of success vs. # of attempts
(the receiver operating characteristic, or ROC, which is the curve of
true positive rate (accuracy) vs. false positive rate (opportunity).
The integral (AUC) yields total success!


Optimization:

1. Change a parameter.

2. As long as performance increases in an exponential manner,

[f(x+dx) / f(x)] > [f(x) / f(x-dx)]

continue changing this parameter in the same manner.


Improvements:

1. Normalized data to make patterns learned applicable to all securities
(for now, not logarithmically)

2. Write patternRecognition() in C and Cythonize it.

3. Numpy arrays instead of python lists / arrays
"""
import time
from logging import INFO, basicConfig, debug
from sys import stdout

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from tqdm import trange

from graphRawGBP2USD1day import GBPUSD1d_PATH


def loadDatetimeBidAskFromCSVdataFile(ifilename, skipfraction=0):
    # type: (str, float) -> tuple
    print 'loading csv text data from file (', ifilename, ') into \
multidimensional array (stripping date to raw number format)'
    date, bid, ask = np.loadtxt(
        ifilename,
        delimiter=',',  # text data file delimiter
        converters={0: mdates.strpdate2num('%Y%m%d%H%M%S')},
        # convert\column 0 using matplotlib.dates "strip date to number" with the given format
        skiprows=int(skipfraction * countLinesInFile(ifilename)),
        unpack=True
    )
    return date, bid, ask


def countLinesInFile(fileName):
    with open(fileName) as myfile:
        for j, line in enumerate(myfile):
            pass
    return j + 1


def percentChange(startPoint, currentPoint):
    if 0 == startPoint:
        return 0

    retval = 100. * (float(currentPoint) - float(startPoint)) / abs(float(startPoint))

    if 0 == retval:
        return pow(10, -10)

    return retval


def average(r):
    assert len(r) > 0
    return reduce(lambda x, y: x + y, r) / len(r)  # lambda expression


def signbitToStr(b):
    """
    Take numerical expression and return whether it evaluates to
    being Negative or not in a string
    :param b:
    """
    if np.signbit(b):
        return "Negative"
    else:
        return "Non-negative"


def patternStorage(i_line, i_futureSkip, i_outcomeDepth, i_patternMatchLength,
                   o_patternArray, o_patternOutcomesArr):
    # processing time
    pattStartTime = time.time()

    timeStepsToSkip = i_futureSkip + i_outcomeDepth

    '''last line to be processed (recall that we need
    (i_futureSkip + i_outcomeDepth) many points after stopLine
    in order for even the most future predictions to have something
    to compare against'''
    stopLine = len(i_line) - timeStepsToSkip

    '''we will calculate i_patternMatchLength many percentChange indexes
     between each of what will be our currentPoint,
      and i_patternMatchLength many points prior to it
      In effect, this is TickReversal<i_patternMatchLength>'''
    percentChangeArray = []
    for i in range(0, i_patternMatchLength):
        percentChangeArray.append(0)

    ''' first line to be processed (recall that we need
    i_patternMatchLength many points prior to this
    for a pattern to be searched)'''
    startLine = i_patternMatchLength + 1
    while stopLine > startLine:  # scan over i_line

        ''' calculate percent changed between the currentPoint
        and pattern matchLength points before it'''
        for i in range(0, len(percentChangeArray)):
            try:
                percentChangeArray[i] = \
                    percentChange(i_line[startLine - i_patternMatchLength],
                                  i_line[startLine - i_patternMatchLength + i])
            except Exception, e:
                print str(e)
                percentChangeArray.append(0)

        # print average of values in (futuristic) outcomeRange
        outcomeRange = \
            i_line[startLine + i_futureSkip:
                   startLine + i_futureSkip + i_outcomeDepth]

        currentPoint = i_line[startLine]
        '''print "average(outcomeRange) = ", average(outcomeRange)

        # print value at current point
        print "currentPoint = ", currentPoint

        print "average(outcomeRange) - currentPoint = ",\
            average(outcomeRange) - currentPoint'''

        futureOutcome = percentChange(currentPoint, average(outcomeRange))
        '''print "futureOutcome = percentChange(currentPoint, \
            average(outcomeRange)) = ", futureOutcome'''

        # print "percentChangeArray = ", percentChangeArray

        o_patternArray.append([])

        for i in range(0, len(percentChangeArray)):
            o_patternArray[len(o_patternArray)
                           - 1].append(percentChangeArray[i])

        '''for eachPattern in o_patternArray:
            print eachPattern'''
        # print len(o_patternArray)

        o_patternOutcomesArr.append(futureOutcome)

        ''' print percentChanged between currentPoint
        and i_patternMatchLength many previous points'''
        '''signbits = []
        overall = bool(True)
        for i, percentChangeElement in enumerate(percentChangeArray):
            signbits.append(np.signbit(percentChangeElement))
            if((0 < i) and (signbits[1] != signbits[i])):
                overall = False
            print "percentChangeElement[", i, "] = ", percentChangeElement,\
                " (", signbitToStr(signbits[i]), ")"'''

        '''if(overall):
            if(signbits[1]):
                print "$$$ Sell! $$$"
            else:
                print "*** Buy! ***"
        else:
            print'''

        startLine += 1

        # time.sleep(1)

    pattEndTime = time.time()
    return pattEndTime - pattStartTime


def currentPattern(i_line, i_patternMatchLength):
    _currentPattern = []
    for i in range(0, i_patternMatchLength):
        '''print "len(i_line) = ", len(i_line), " beginIndex = ",\
            -1 - i_patternMatchLength, " endIndex = ",\
            i - i_patternMatchLength, " startPoint = ",\
            i_line[-1 - i_patternMatchLength], " endPoint = ",\
            i_line[-1 - i_patternMatchLength + i]'''

        _currentPattern.append(percentChange(i_line[-1 - i_patternMatchLength], i_line[-1 - i_patternMatchLength + i]))

    return _currentPattern


def exponentialWeight(i, n):
    """ sum over n (from 0 to N - 1) of exp(inx) is (1 - exp(iNx)) / (1 - exp(ix))
therefore, the sum over n (from 1 to N) of exp(-n) is exp(inx) is
(1 - exp(iNx)) / (1 - exp(ix)) + exp(iNx) - exp(0) with x = i
which is (1 - exp(-N)) / (1 - exp(-1)) + exp(-N) - 1
which simplifies to (1 - exp(-N)) * (1 / (1 - exp(-1)) - 1)
or (1 - exp(-N)) * ((1 - 1 - exp(-1)) / (1 - exp(-1)))
(1 - exp(-N)) * (exp(-1) / (exp(-1) - 1))
(1 - exp(-N)) * (1 / (1 - exp(1)))
(1 - exp(-N)) / (1 - exp(1))

e(0) + e(-1) = 1 + e(-1)
e(0) + e(-1) + e(-2) = 1 + e(-1)*(1 + e(-1)) = (1 + e(-1))^2
e(0) + e(-1) + e(-2) + e(-3) = 1 + e(-1)*((1 + e(-1))^2)  = (1 + e(-1))^3
"""
    return np.exp(1 - i) / pow((1 + np.exp(-1)), n)


def showPlotPatternRecognition(i_patternMatchLength, i_patternsMatched,
                               i_patternOutcomes, i_pcolor,
                               i_indexesOfPatternsToBePloted,
                               i_patternForRecognition,
                               currentMovement, i_outcomesAverage):
    if 0 == len(i_indexesOfPatternsToBePloted):
        return

    xlabels = []
    for i in range(1, i_patternMatchLength + 1):
        xlabels.append(i)

    fig = Figure(figsize=(10, 6))
    for i in i_indexesOfPatternsToBePloted:
        plt.plot(xlabels, i_patternsMatched[i], ':')
        # @todo histogram of predicted outcomes
        plt.scatter(i_patternMatchLength * 1.1,
                    i_patternOutcomes[i],
                    c=i_pcolor[i], alpha=0.3)
    plt.scatter(i_patternMatchLength * 1.2,
                currentMovement,
                c='k', s=25)
    plt.scatter(i_patternMatchLength * 1.2,
                i_outcomesAverage,
                c='b', s=35, alpha=0.3)
    plt.plot(xlabels, i_patternForRecognition, 'k', linewidth=4)
    plt.grid(True)
    plt.title("Pattern Recognition")
    plt.show()


def patternRecognition(patternArray,
                       patternOutcomesArr,
                       predictionArray,
                       patternMatchLength,
                       patternForRecognition,
                       doGraph, lowerBoundSimilarity=80):
    indexesOfPatternsMatched = []
    pcolor = [None] * len(patternArray)

    similarityArray = []
    for k in range(0, patternMatchLength):
        similarityArray.append(0)

    lowerBoundSimilarityElementwise = 0
    maxSim = 0
    for eachPattern in patternArray:

        i = 0
        skipThisPattern = False
        while patternMatchLength > i:

            similarityArray[i] = (100. - abs(percentChange(
                eachPattern[i], patternForRecognition[i])))
            ''' * exponentialWeight(patternMatchLength -
            i, patternMatchLength)'''

            debug("status %f >? %f" % (lowerBoundSimilarityElementwise, similarityArray[i]))  # https://github.com/InonS/sentdex-patternrecog-percentchange/issues/1

            # optimization
            if lowerBoundSimilarityElementwise > similarityArray[i]:
                skipThisPattern = True
                break

            i += 1

        if skipThisPattern:
            continue

        howSim = average(similarityArray)
        patternIndex = patternArray.index(eachPattern)
        # print "eachPattern = ", eachPattern
        # print patternIndex, " : ", howSim, "% ",

        maxSim = max(maxSim, howSim)
        if max(maxSim, lowerBoundSimilarity) > howSim:
            # print
            continue

        indexesOfPatternsMatched.append(patternIndex)

        if (patternForRecognition[patternMatchLength - 1] <
                patternOutcomesArr[patternIndex]):
            pcolor[patternIndex] = 'g'  # '#24bc00'
            predictionArray.append(1.0)
        else:
            pcolor[patternIndex] = 'r'  # '#d40000'
            predictionArray.append(-1.0)

        # print " = ", similarityArray
        #
        # print '###########################'
        # print '###########################'
        # print "patternForRecognition = ", patternForRecognition
        # print '==========================='
        # print '==========================='
        # print "howSim = ", howSim, "% "
        # print "eachPattern = ", eachPattern
        # print '---------------------------'
        # print "patternIndex = ", patternIndex
        # print "patternOutcomesArr[patternIndex] = ", \
        #     patternOutcomesArr[patternIndex]
        #
        # if doGraph:
        #     showPlotPatternRecognition(patternMatchLength, patternArray,
        #                                patternOutcomesArr, pcolor,
        #                                patternIndex, patternForRecognition, None, None)
        #
        # print '@@@@@@@@@@@@@@@@@@@@@@@@@@@'
        # print '@@@@@@@@@@@@@@@@@@@@@@@@@@@'

    # print "maxSim = ", maxSim, "%"

    # print "predictionArray = ", predictionArray
    avgPrediction = average(predictionArray) if len(predictionArray) > 0 else None
    # print "avgPrediction = ", avgPrediction

    return indexesOfPatternsMatched, pcolor, (0 < avgPrediction if avgPrediction else None)


def searchAllSampleLengths(entireAvgLine, patternMatchLength):
    """
    back-tester:
    1. accuracyArray = How accurate are our predictions?
    2. samps = How many tests have we run?
    3. accuracy / num of tests = %accuracy
    :param entireAvgLine:
    :param patternMatchLength:
    """

    accuracyArray = []
    samps = 0

    '''the pattern finder will attempt to predict into the future starting
    after futureSkip many points, and for outcomeDepth many points'''
    futureSkip = 20
    outcomeDepth = 10
    outcomesArray = []

    # put in multidim array:
    patternsArray = []
    patternOutcomesArr = []
    predictionArray = []

    dataLength = len(entireAvgLine)
    startingSampleLength = dataLength / 2  # 2 * patternMatchLength

    postfix = dict()
    sample_iterator = trange(startingSampleLength, dataLength, unit="sample", postfix=postfix)
    for currentSampleLength in sample_iterator:

        avgLine = entireAvgLine[:currentSampleLength]

        currentOutcomeRange = entireAvgLine[
                              currentSampleLength + futureSkip:currentSampleLength + futureSkip + outcomeDepth]
        currentAverageOutcome = average(currentOutcomeRange) if len(currentOutcomeRange) > 0 else 0
        currentMovement = percentChange(entireAvgLine[currentSampleLength],
                                        currentAverageOutcome)
        outcomesArray.append(currentMovement)

        # print
        # print "***************************************************************"
        # print "***************************************************************"

        # run patternStorage
        duration = patternStorage(avgLine, futureSkip, outcomeDepth,
                                  patternMatchLength,
                                  patternsArray,
                                  patternOutcomesArr)

        # print "---------------------------------------------------------------"

        debug("currentSampleLength = ", currentSampleLength, ", len(patternsArray) = ", len(patternsArray),
              ",len(patternOutcomesArr) = ", len(patternOutcomesArr), " (should be equal). duration = ", duration,
              " seconds")

        # print "***************************************************************" print
        # "***************************************************************" print

        patternForRecognition = currentPattern(avgLine, patternMatchLength)

        indexesOfPatternsMatched, pcolor, isPredictionGood = \
            patternRecognition(patternsArray, patternOutcomesArr,
                               predictionArray, patternMatchLength,
                               patternForRecognition, False)

        '''showPlotPatternRecognition(patternMatchLength, patternsArray,
                                   patternOutcomesArr, pcolor,
                                   indexesOfPatternsMatched,
                                   patternForRecognition,
                                   currentMovement, average(outcomesArray))'''

        if not isPredictionGood:
            # print "predicted drop"

            debug("last point in patternForRecognition = ", patternForRecognition[len(patternForRecognition) - 1],
                  ", currentMovement = ", currentMovement)

            if (patternForRecognition[len(patternForRecognition) - 1]
                    > currentMovement):
                accuracyArray.append(100.0)
            else:
                accuracyArray.append(0.0)
        else:
            print "predicted rise"
            print "last point in patternForRecognition = ", \
                patternForRecognition[len(patternForRecognition) - 1], \
                ", currentMovement = ", currentMovement
            if (patternForRecognition[len(patternForRecognition) - 1]
                    > currentMovement):
                accuracyArray.append(0.0)
            else:
                accuracyArray.append(100.0)

        samps += 1
        # print "Back-tested accuracy is: ", str(average(accuracyArray))[:5], "% after ", samps, " samples"
        postfix['Back-tested accuracy'] = str(average(accuracyArray))[:5] + "%"
        sample_iterator.set_postfix(postfix)
        currentSampleLength += 1


def main():
    basicConfig(stream=stdout, level=INFO)

    totalStart = time.time()

    # define input file
    f = GBPUSD1d_PATH  # file name
    print "file=", f

    print "lineNum=", countLinesInFile(f)

    # load input file to RAM
    date, bid, ask = loadDatetimeBidAskFromCSVdataFile(f, 0.95)
    dataLength = len(date)
    print 'len(date) = ', dataLength, ' (number of data entries loaded)'

    # basic input will be the i_bids-i_asks average
    searchAllSampleLengths((bid + ask) / 2, 30)

    totalEnd = time.time()

    totalTime = totalEnd - totalStart

    print "# Normal termination. Entire processing time took: ", \
        totalTime, " s"


main()
