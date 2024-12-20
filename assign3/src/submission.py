# 20200429 Doyoung Kim

import random
import collections
import math
import sys
from collections import Counter
from util import *

SEED = 4312

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    result = {}
    for word in x.split():
        if word in result:
            result[word] += 1
        else:
            result[word] = 1
    return result
    # END_YOUR_ANSWER


############################################################
# Problem 3b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    """
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    """
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    def gradient(phi, y, w):
        label = 1 if y == 1 else 0
        value = lambda k: (sigmoid(dotProduct(w, phi)) - label) * phi.get(k, 0)
        return {k: value(k) for k in phi.keys()}

    for _ in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            increment(weights, -eta, gradient(phi, y, weights))
    # END_YOUR_ANSWER
    return weights


############################################################
# Problem 3c: bigram features


def extractBigramFeatures(x):
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
    phi = extractWordFeatures(x)
    splitted = ["<s>"] + x.split() + ["</s>"]

    for i in range(len(splitted) - 1):
        pair = (splitted[i], splitted[i + 1])
        if pair in phi:
            phi[pair] += 1
        else:
            phi[pair] = 1
    # END_YOUR_ANSWER
    return phi
