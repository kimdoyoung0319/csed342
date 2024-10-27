# 20200429 Doyoung Kim

import util, math, random
from collections import defaultdict
from util import ValueIteration


# Returns a tuple whose ith element is decreased by 1 from t. Other elements are
# same.
def decreased(i: int, t: tuple):
    li = list(t)
    li[i] -= 1
    return tuple(li)


############################################################
# Problem 1a: BlackjackMDP


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the
    # player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        # total, next card (if any), multiplicity for each card
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ["Take", "Peek", "Quit"]

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None.
    # When the probability is 0 for a particular transition, don't include that
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER
        match action:
            case "Take":
                return self.takeResults(state)
            case "Peek":
                return self.peekResults(state)
            case "Quit":
                return self.quitResults(state)

        return None

    # Returns a list of (newState, prob, reward) if action 'Take' is held in
    # argument state.
    def takeResults(self, state):
        total, top, deck = state

        # What should be returned when the game is in end state?
        # i.e. deck is None.
        if deck is None:
            return []

        if all(count == 0 for count in deck):
            return [((total, top, None), 1.0, total)]

        if top is not None and self.cardValues[top] + total <= self.threshold:
            nextState = (self.cardValues[top] + total, None, decreased(top, deck))
            return [(nextState, 1.0, 0)]
        elif top is not None:
            nextState = (self.cardValues[top] + total, None, None)
            return [(nextState, 1.0, 0)]

        # What should happen when the player took a card from deck, which leads
        # the deck to be empty and the sum of card values in player's hand
        # to exceed the threshold?
        result = []
        for card in range(len(deck)):
            if deck[card] <= 0:
                continue
            elif all(count == 0 for count in decreased(card, deck)):
                score = self.cardValues[card] + total
                nextState = (score, None, None)
                result.append((nextState, deck[card] / sum(deck), score))
            elif self.cardValues[card] + total > self.threshold:
                nextState = (self.cardValues[card] + total, None, None)
                result.append((nextState, deck[card] / sum(deck), 0))
            else:
                nextState = (self.cardValues[card] + total, None, decreased(card, deck))
                result.append((nextState, deck[card] / sum(deck), 0))

        return result

    # Returns a list of (newState, prob, reward) if action 'Peek' is held in
    # argument state.
    def peekResults(self, state):
        total, top, deck = state

        if deck is None or top is not None:
            return []

        return [
            ((total, card, deck), deck[card] / sum(deck), -self.peekCost)
            for card in range(len(deck))
            if deck[card] > 0
        ]

    # Returns a list of (newState, prob, reward) if action 'Quit' is held in
    # argument state.
    def quitResults(self, state):
        total, top, deck = state

        if deck is None:
            return []

        # What should be 'top' of state when quitting? None? Original top?
        return [((total, top, None), 1.0, total)]
        # END_YOUR_ANSWER

    def discount(self):
        return 1


############################################################
# Problem 1b: ValueIterationDP


class ValueIterationDP(ValueIteration):
    """
    Solve the MDP using value iteration with dynamic programming.
    """

    def solve(self, mdp):
        difference = 1
        threshold = 0.001
        values = defaultdict(float)

        # BEGIN_YOUR_ANSWER
        while difference >= threshold:
            difference = 0
            for state in mdp.states:
                prevValue = values[state]
                values[state] = max(
                    self.computeQ(mdp, values, state, action)
                    for action in mdp.actions(state)
                )
                difference = max(difference, abs(prevValue - values[state]))
        # END_YOUR_ANSWER

        # Compute the optimal policy now
        policy = self.computeOptimalPolicy(mdp, values)
        self.pi = policy
        self.V = values


############################################################
