# 20200429 Doyoung Kim

import util, math, random
from collections import defaultdict
from util import ValueIteration

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
        def decreased(deck, card):
            li = list(deck)
            li[card] -= 1
            return tuple(li)

        def isDeckEmptyAfterTake(deck, card):
            cards = range(len(deck))
            if deck[card] > 1:
                return False
            else:
                return all(deck[other] == 0 for other in cards if other != card)

        value = self.cardValues
        threshold = self.threshold
        total, top, deck = state

        if deck is None:
            return []

        cards = range(len(deck))
        cardsCount = sum(deck)

        if action == "Take" and top is None:
            result = []
            for card in cards:
                if deck[card] <= 0:
                    continue
                prob, reward, updated = (deck[card] / cardsCount, 0, None)
                if isDeckEmptyAfterTake(deck, card):
                    reward = value[card] + total
                elif value[card] + total <= threshold:
                    updated = decreased(deck, card)
                result.append(((value[card] + total, None, updated), prob, reward))
            return result

        if action == "Take" and top is not None:
            updated, reward = (None, 0)
            if isDeckEmptyAfterTake(deck, top):
                reward = value[top] + total
            elif value[top] + total <= threshold:
                updated = decreased(deck, top)
            return [((value[top] + total, None, updated), 1.0, reward)]

        if action == "Peek" and top is None:
            result = []
            for card in cards:
                if deck[card] <= 0:
                    continue
                prob = deck[card] / cardsCount
                result.append(((total, card, deck), prob, -self.peekCost))
            return result

        if action == "Peek" and top is not None:
            return []

        if action == "Quit":
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
        values = defaultdict(float)
        changed = defaultdict(lambda: False)

        # BEGIN_YOUR_ANSWER
        difference = 1
        error = 0.001
        while difference >= error:
            difference = 0
            for state in mdp.states:
                prev = values[state]
                qValues = [
                    self.computeQ(mdp, values, state, action)
                    for action in mdp.actions(state)
                ]
                values[state] = max(qValues)
                difference = max(difference, abs(prev - values[state]))
        # END_YOUR_ANSWER

        # Compute the optimal policy now
        policy = self.computeOptimalPolicy(mdp, values)
        self.pi = policy
        self.V = values


############################################################
