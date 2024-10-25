# 20240000 John Doe

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
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError  # remove this line before writing code
        # END_YOUR_ANSWER

    def discount(self):
        return 1

############################################################
# Problem 1b: ValueIterationDP

class ValueIterationDP(ValueIteration):
    '''
    Solve the MDP using value iteration with dynamic programming.
    '''
    def solve(self, mdp):
        V = {}  # state -> value of state

        # BEGIN_YOUR_ANSWER (our solution is 13 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError  # remove this line before writing code
        # END_YOUR_ANSWER

        # Compute the optimal policy now
        pi = self.computeOptimalPolicy(mdp, V)
        self.pi = pi
        self.V = V

############################################################