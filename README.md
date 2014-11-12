Sentdex-PatternRecog_percentChange
==================================

Sentdex ML for Algo-trading tutorial

Assumption of Technical Analysis: 
---------------------------------
Graph patterns supposedly reflect trends in
the traders' beliefs, and therefore yield

Hypothesis:
-----------
The re-appearance of a certain series of signals or variables
(a pattern) is a predictor of the following terms in the series,
or the outcome.

Goal: 
-----
Pattern matching (Machine Classification) to find Technical Analysis
graph patterns.

Prerequisite:
-------------
http://sentdex.com/GBPUSD.zip
or any other tick data csv file  in format of date time, bid,ask

Objectives:
-----------
1. Run a batch of algorithms on past data, testing each against past outcomes,
and projecting this insight into the future.
In the context of Machine Learning, making inferences is
termed "data snooping".
2. Test the underlying assumption: Past pattern recognition will yield success
with future data.

Explicit variables:
-------------------
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

Implicit variable:
------------------
8. Opportunity vs. accuracy:
Optimize area-under-curve (AUC) of chances of success vs. # of attempts
(the receiver operating characteristic, or ROC, which is the curve of
true positive rate (accuracy) vs. false positive rate (opportunity).
The integral (AUC) yields total success!

Optimization:
-------------
1. Change a parameter.
2. As long as performance increases in an exponential manner,
[f(x+dx) / f(x)] > [f(x) / f(x-dx)]
continue changing this parameter in the same manner.

Improvements:
-------------
1. Normalized data to make patterns learned applicable to all securities
(for now, not logarithmically)
2. Write patternRecognition() in C and Cythonize it.
3. Numpy arrays instead of python lists / arrays
