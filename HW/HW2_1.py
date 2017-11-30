# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 23:11:21 2016

@author: quang
"""

def flip_virtual_coin(numberOfCoins, numberOfTime):
    """
    Flipping virtual fair coins

    Inputs
    ------
    - numberOfCoins: number of fair coins being flipped
    - numberOfTime: number of independently tosses

    Outputs
    ------
    - outcomeOfFlipping: a numberOfCoins times numberOfTime matrix stored as 
        2D array; each row is numberOfTime outcomes of a coin, which represents
        by 0:'tail' or 1:'head'
    """
    import numpy as np
    outcomeOfFlipping = np.zeros([numberOfCoins, numberOfTime])
    for i in range(numberOfCoins):
        for j in range(numberOfTime):
            outcomeOfFlipping[i,j]=np.random.choice(2)
    return outcomeOfFlipping
    
def choose_three_coins(outcomeOfFlipping):
    """
    Return 3 coins as follow; c_1 is the first coin fliped, c_rand is the coin 
    chosen random, and c_min is the coin which had the minimum frequency of 
    heads
    
    Inputs
    ------
    - outcomeOfFlipping
    
    Outputs
    ------
    - outcomeOfThreeCoins: array of 3 rows correspond to outcome of 3 chosen coins with the order
        c_1, c_rand, c_min; array[0,:] is the outcome of coin c_1
    """
    import numpy as np
    c_rand_index = np.random.choice(outcomeOfFlipping.shape[0])
    c_min_index = outcomeOfFlipping.sum(axis = 1).argmin()
    outcomeOfThreeCoins = outcomeOfFlipping[[0, c_rand_index, c_min_index]]
    return outcomeOfThreeCoins

def fraction_of_heads(outcome):
    """
    Return the fraction of heads obtained for the coins specified in outcome
    
    Inputs
    ------
    - outcome: outcome of the interest coins
    
    Outputs
    ------
    - V: array of the fraction of heads of the interest coins
    """
    V = outcome.sum(axis = 1) / outcome.shape[1]
    return V

def expreriment(numberOfTime):
    import numpy as np
    headFractionDistribution = np.zeros([numberOfTime,3])
    for i in range(numberOfTime):
        outcomeOfAllCoins = flip_virtual_coin(1000, 10)
        outcomeOfThreeCoins = choose_three_coins(outcomeOfAllCoins)
        headFractionDistribution[i,:] = fraction_of_heads(outcomeOfThreeCoins)
    return headFractionDistribution

headFractionDistribution = expreriment(100)
    
    
    
    