#!/usr/bin/python

import graderUtil
import util
import time
from util import *

grader = graderUtil.Grader()
submission = grader.load("submission")

try:
    import solution

    grader.addHiddenPart = grader.addBasicPart
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False


def test_correct(func_name, assertion=lambda pred: True, equal=lambda x, y: x == y):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(equal(pred, answer))

    return test


def test_wrong(func_name, assertion=lambda pred: True):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(pred != answer and pred is not None)

    return test


def veceq(vec1, vec2):
    def veclen(vec):
        return sum(1 for k, v in vec.items() if v != 0)

    if veclen(vec1) != veclen(vec2):
        return False
    else:
        return all(v == vec2.get(k, 0) for k, v in vec1.items())


############################################################
# Problem 3: sentiment classification
############################################################


### 3a


# Basic sanity check for feature extraction
def test3a0():
    ans = {"a": 2, "b": 1}
    grader.requireIsEqual(ans, submission.extractWordFeatures("a b a"))


grader.addBasicPart("3a-0-basic", test3a0, maxSeconds=1, description="basic test")


def test3a1():
    random.seed(SEED)

    def get_gen():
        for i in range(10):
            sentence = " ".join(
                [random.choice(["a", "aa", "ab", "b", "c"]) for _ in range(100)]
            )
            pred = submission.extractWordFeatures(sentence)
            if solution_exist:
                answer = solution.extractWordFeatures(sentence)
                yield grader.requireIsTrue(veceq(pred, answer))
            else:
                yield True

    all(get_gen())


grader.addHiddenPart(
    "3a-1-hidden",
    test3a1,
    maxSeconds=1,
    description="test multiple instances of the same word in a sentence",
)


### 3b


def test3b0():
    trainExamples = (("hello world", 1), ("goodnight moon", -1))
    testExamples = (("hello", 1), ("moon", -1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(
        trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01
    )
    grader.requireIsGreaterThan(0, weights["hello"])
    grader.requireIsLessThan(0, weights["moon"])


grader.addBasicPart(
    "3b-0-basic",
    test3b0,
    maxSeconds=1,
    description="basic sanity check for learning correct weights on two training and testing examples each",
)


def test3b1():
    trainExamples = (("hi bye", 1), ("hi hi", -1))
    testExamples = (("hi", -1), ("bye", 1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(
        trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01
    )
    grader.requireIsLessThan(0, weights["hi"])
    grader.requireIsGreaterThan(0, weights["bye"])


grader.addBasicPart(
    "3b-1-basic",
    test3b1,
    maxSeconds=1,
    description="test correct overriding of positive weight due to one negative instance with repeated words",
)


def test3b2():
    trainExamples = readExamples("polarity.train")
    devExamples = readExamples("polarity.dev")
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(
        trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        devExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    devError = evaluatePredictor(
        devExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print("Official: train error = %s, dev error = %s" % (trainError, devError))
    grader.requireIsEqual(0.0737198, trainError, 0.015)
    grader.requireIsEqual(0.2771525, devError, 0.02)


grader.addBasicPart(
    "3b-2-basic",
    test3b2,
    maxPoints=6,
    maxSeconds=8,
    description="test classifier on real polarity dev dataset",
)

### 3c


def test3c0():
    sentence = "I am what I am"
    ans = {
        ("am", "what"): 1,
        "what": 1,
        ("I", "am"): 2,
        "I": 2,
        ("what", "I"): 1,
        "am": 2,
        ("<s>", "I"): 1,
        ("am", "</s>"): 1,
    }
    grader.requireIsEqual(ans, submission.extractBigramFeatures(sentence))


grader.addBasicPart(
    "3c-0-basic", test3c0, maxSeconds=1, description="test basic bigram features"
)


def test3c1():
    random.seed(SEED)

    def get_gen():
        for i in range(10):
            sentence = " ".join(
                [random.choice(["a", "aa", "ab", "b", "c"]) for _ in range(100)]
            )
            pred = submission.extractBigramFeatures(sentence)
            if solution_exist:
                answer = solution.extractBigramFeatures(sentence)
                yield grader.requireIsTrue(veceq(pred, answer))
            else:
                yield True

    all(get_gen())


grader.addHiddenPart(
    "3c-1-hidden",
    test3c1,
    2,
    maxSeconds=1,
    description="test feature extraction on repeated bigrams",
)

grader.grade()
