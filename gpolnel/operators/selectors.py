""" Selection operators select candidate solutions for breeding
The module `gpol.selectors` contains some popular selection
operators (selectors) used to select one or several (for DE) candidate
solutions based on their fitness (except when it comes to random
selection, also implemented here). Given the fact solutions' selection
is traditionally fitness-based, as such invariant to solutions'
representation, almost all the selectors can be applied with any kind
of OPs.
"""

import timeit

import math
import random
import torch
from gpolnel.utils.tree import Tree

from copy import deepcopy

def prm_tournament(pressure):
    """ Implements tournament selection algorithm

    This function is used to provide the tournament (inner function)
    with the necessary environment (the outer scope) - the selection's
    pressure.

    Note that tournament selection returns the index of the selected
    solution, not the representation. This function can be used for the
    GeneticAlgorithm or the GSGP instances.

    Parameters
    ----------
    pressure : float
        Selection pressure.

    Returns
    -------
    tournament : function
        A function which implements tournament selection algorithm
        with a pool calculated as 'int(len(population) * pressure)'.
    """
    def tournament(pop, min_):
        """ Implements tournament selection algorithm

        The tournament selection algorithm returns the most-fit
        solution from a pool of randomly selected solutions. The pool
        is calculated as 'int(len(population) * pressure)'. This
        function can be used for the GeneticAlgorithm or the GSGP
        instances.

        Parameters
        ----------
        pop : Population
            The pointer to the population to select individuals from.
        min_ : bool
            The purpose of optimization.

        Returns
        -------
        int
            The index of the most-fit solution from the random pool.
        """
        # Computes tournament pool size with respect to the population
        pool_size = math.ceil(len(pop) * pressure)
        # Gets random indices of the individuals
        indices = torch.randint(low=0, high=len(pop), size=(pool_size, ))
        # Returns the best individual in the pool
        return indices[torch.argmin(pop.fit[indices])] if min_ else indices[torch.argmax(pop.fit[indices])]

    return tournament


def prm_double_tournament(pressure, size_criteria=True):
    """
    Implements a double tournament selection based on fitness and tree size.

    Parameters:
    - pressure: float, the proportion of the population to sample for each tournament.
    - size_criteria: bool, True if the final selection is based on tree size, False if based on fitness.

    Returns:
    - Function that performs the double tournament on a given population.
    """

    def double_tournament(pop, min_=True):
        """
        Conducts two sequential tournaments to select an individual from the population.

        Parameters:
        - pop: Population from which to select individuals.
        - min_: Boolean, True if minimizing fitness, False if maximizing.

        Returns:
        - Index of the winning individual based on the criteria.
        """

        def get_size(individual):
            """
            Returns the size of the individual.
            Uses get_size method if available, otherwise uses the length of the representation.
            """
            if isinstance(individual, Tree):
                return individual.get_size()
            return len(individual.repr_)

        def tournament(pop, fitness_min=True):
            """
            Conducts a single tournament based on fitness.

            Parameters:
            - pop: Population from which to select individuals.
            - fitness_min: Boolean, True if minimizing fitness, False if maximizing.

            Returns:
            - Index of the winner of the tournament.
            """
            pool_size = int(len(pop.individuals) * pressure)
            contestants = random.sample(pop.individuals, pool_size)
            if fitness_min:
                winner = min(contestants, key=lambda ind: ind.fit)
            else:
                winner = max(contestants, key=lambda ind: ind.fit)
            return pop.individuals.index(winner)

        # First stage: Perform two independent tournaments based on fitness
        winner1_idx = tournament(pop, fitness_min=min_)
        winner2_idx = tournament(pop, fitness_min=min_)

        # Second stage: Compete based on tree size or fitness
        winner1 = pop.individuals[winner1_idx]
        winner2 = pop.individuals[winner2_idx]
        
        if size_criteria:
            final_winner = min([winner1, winner2], key=get_size)
        else:
            final_winner = min([winner1, winner2], key=lambda ind: ind.fit) if min_ else max([winner1, winner2], key=lambda ind: ind.fit)

        return pop.individuals.index(final_winner)

    return double_tournament


def roulette_wheel(pop, min_):
    """ Implements roulette wheel selection algorithm

    Generates and returns the index in [0, len(pop)[ range
    after fitness proportionate (a.k.a. roulette wheel)
    selection algorithm.

    Parameters
    ----------
    pop : Population
        The pointer to the population to select individuals from.
    min_ : bool
        The purpose of optimization. In this procedure, as selection
        is performed randomly, it exists only for to obey library's
        standards.

    Returns
    -------
    int
        The index of the solution after fitness proportionate selection.
    """
    prop_fit = pop.fit/pop.fit.sum()
    _, indices = torch.sort(prop_fit, descending=min_)
    cum_fit = torch.cumsum(prop_fit, dim=0)
    return indices[cum_fit > random.uniform(0, 1)][0]


def rank_selection(pop, min_):
    """ Implements rank selection algorithm

    Generates and returns the index in [0, len(pop)[ range. Parents'
    selection depends on the relative rank of the fitness and not the
    fitness itself. The higher the rank of a given parent, the higher
    its probability of being selected. It is recommended to use when
    the individuals in the population have very close fitness values.

    Parameters
    ----------
    pop : Population
        The pointer to the population to select individuals from.
    min_ : bool
        The purpose of optimization. In this procedure, as selection
        is performed randomly, it exists only for to obey library's
        standards.

    Returns
    -------
    int
        The index of the solution after fitness rank selection.
    """
    _, indices = torch.sort(pop.fit, descending=min_)
    indices_ = indices + 1
    indices_prop = indices_/indices_.sum()
    cum_indices = torch.cumsum(indices_prop, dim=0)
    sel = random.uniform(0, 1)
    return torch.flip(indices, (0, ))[cum_indices > sel][0] if min_ else indices[cum_indices > sel][0]


def rnd_selection(pop, min_):
    """ Implements random selection algorithm

    Generates and returns random index in [0, len(pop)[ range.

    Parameters
    ----------
    pop : Population
        The pointer to the population to select individuals from.
    min_ : bool
        The purpose of optimization. In this procedure, as selection
        is performed randomly, it exists only for to obey library's
        standards.

    Returns
    -------
    int
        A random index in [0, len(pop)[ range.
    """
    return random.randint(0, len(pop)-1)
