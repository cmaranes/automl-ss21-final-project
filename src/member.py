from enum import IntEnum

# https://stackoverflow.com/questions/48338847/how-to-copy-a-class-instance-in-python
from copy import deepcopy
import numpy as np
import logging


# The two following classes just make it convenient to select which mutation/recombination/selectoin to use with EA
class Recombination(IntEnum):
    NONE = -1  # can be used when only mutation is required
    UNIFORM = 0  # uniform crossover (only really makes sense for function dimension > 1)
    INTERMEDIATE = 1  # intermediate recombination


class Mutation(IntEnum):
    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    GAUSSIAN = 1  # Gaussian mutation


class ParentSelection(IntEnum):
    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2


class Member:
    """
    Class to simplify member handling.
    """

    member_count = 0  # Used for id of members

    def __init__(self, search_space, initial_x, target_function,
                 mutation: Mutation = None, recombination: Recombination = None,
                 recom_prob=None, max_num_params: int = None, seed: int = 0) -> None:
        """
        Init
        :param search_space: Search space in which this member is defined
        :param initial_x: Initial coordinate of the member
        :param target_function: The target function that determines the fitness value
        :param mutation: hyperparameter that determines which mutation type use
        :param recombination: hyperparameter that determines which recombination type to use
        :param recom_prob: Optional hyperparameter that is only active if recombination is uniform
        :param max_num_params: Maximum number of the parameters of the CNN
        :param seed: Random seed for reproducibility purposes
        """
        self.seed = seed
        self.search_space = search_space
        self._f = target_function
        self._max_num_params = max_num_params

        if initial_x is None:
            while True:  # Ensure that the configuration satisfy the number of parameters
                sampled_configuration = search_space.sample_configuration()
                satisfied_params = self._f.satisfy_num_param(sampled_configuration, self._max_num_params)
                if satisfied_params or max_num_params is None:
                    break
            self.x = sampled_configuration
        else:
            self.x = initial_x

        self._age = 0  # basically indicates how many offspring were generated from this member
        self._mutation = mutation
        self._recombination = recombination
        self.x_changed = True
        self._fit = None
        self._recom_prob = recom_prob
        self._first_epoch = True
        self.n_trained_epochs = 0
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self.member_id = Member.member_count
        Member.member_count += 1

    def get_fitness(self):
        if self.x_changed:  # Only if the x_coordinate has changed we need to evaluate the fitness.
            self.x_changed = False
            self._fit = self._f.train(self.x)
        return self._fit  # otherwise we can return the cached value

    def get_info(self):
        info = {'configuration': self.x.get_dictionary(),
                'top3': self._fit['acc'],
                'precision': self._fit['precision'],
                'num_params': self._fit['num_params']}
        return info

    def train_n_epoch(self, n_epochs=1):
        self._first_epoch = False
        self.x_changed = False
        self._fit = self._f.train(self.x, self.member_id, budget=n_epochs)
        self.n_trained_epochs += n_epochs
        return self._fit

    def mutate(self):
        """
        Mutation which creates a new offspring. It mutates until the max_num_params constraint is satisfied
        :return: new member who is based on this member
        """
        self.logger.info("Mutation has been chosen")
        new_x = deepcopy(self.x)
        if self._mutation == Mutation.UNIFORM:
            while True:
                gene = np.random.choice(list(new_x.get_dictionary().keys()),
                                        replace=False)
                gene = str(gene)
                hyperparameter = self.search_space.get_hyperparameter(str(gene))
                # This is because if it has previously mutated and the number of layers is less, the key
                # is mantained and it should not be like that (but in ConfigSpace)
                if new_x[gene] == None:
                    continue
                # https://github.com/automl/ConfigSpace/blob/5e8acfddc8a1f5dec566e2a59f53d4c628fe0f1c/ConfigSpace/hyperparameters.pyx#L75
                new_value = hyperparameter.sample(np.random)
                self.logger.debug(f"Mutating {gene}: {new_x[gene]} -> {new_value}")
                new_x[gene] = new_value
                satisfied_params = self._f.satisfy_num_param(new_x, self._max_num_params)
                if satisfied_params:  # If params not satisfied other param can be changed until finding equilibrium
                    child = Member(self.search_space, new_x, self._f, self._mutation, self._recombination,
                                   self._recom_prob, self._max_num_params, seed=self.seed)
                    self.logger.debug(f"Mutation sucessful!")
                    break
                self.logger.debug(f"Hyperparameter constraint is not satisfied. Mutating again...")

        elif self._mutation == Mutation.GAUSSIAN:
            # Gaussian has not been implemented since there are just a few real (float) parameters
            raise NotImplementedError

        elif self._mutation != Mutation.NONE:
            # We won't consider any other mutation types
            raise NotImplementedError

        self._age += 1
        return child

    def recombine(self, partner):
        """
        Recombination of this member with a partner. It recombines until the max_num_params constraint is satisfied
        :param partner: Member
        :return: new offspring based on this member and partner
        """
        if self._recombination == Recombination.INTERMEDIATE:
            raise NotImplementedError
        elif self._recombination == Recombination.UNIFORM:
            assert self._recom_prob is not None, \
                'for this recombination type you have to specify the recombination probability'

            self.logger.info("Recombination has been chosen")
            self.logger.debug("-- Parent 1 -- ")
            self.logger.debug(str(self.x))
            self.logger.debug("-- Parent 2 -- ")
            self.logger.debug(str(partner.x))
            while True:
                new_x = deepcopy(self.x)
                for hyperparam in list(new_x.get_dictionary().keys()):
                    # These are not recombined since there can be problem when then reading the number of parameters
                    # of an unexisting layer
                    self.logger.debug(f"Deciding hyperparam {hyperparam}")
                    if hyperparam in ["n_conv_layers", "n_fc_layers"]:
                        continue
                    take_partner = np.random.random()
                    if take_partner < self._recom_prob:
                        if hyperparam in list(partner.x.get_dictionary().keys()):
                            # This is because if it has previously mutated and the number of layers is less, the key
                            # is mantained and it should not be like that (but in ConfigSpace)
                            if new_x.get(hyperparam) is None:
                                continue
                            if partner.x.get(hyperparam) is None:
                                continue
                            self.logger.debug(f"Taking {hyperparam} from parent 2")
                            # self.logger.debug(f"Parent 1 {hyperparam} value: {new_x[hyperparam]}")
                            new_x[hyperparam] = partner.x[hyperparam]

                satisfied_params = self._f.satisfy_num_param(new_x, self._max_num_params)
                if satisfied_params:  # If params not satisfied other param can be changed until finding equilibrium
                    self.logger.debug(f"Recombination sucessful!")
                    break
                self.logger.debug(f"Hyperparameter constraint is not satisfied. Recombining again...")

        elif self._recombination == Recombination.NONE:
            new_x = deepcopy(self.x)  # copy is important here to not only get a reference
        else:
            raise NotImplementedError
        child = Member(self.search_space, new_x, self._f, self._mutation, self._recombination,
                       self._recom_prob, self._max_num_params, seed=self.seed)
        self._age += 1
        return child

    def __str__(self):
        """Makes the class easily printable"""
        str = "Population member: Age={}, x={}, f(x)={}".format(self._age, self.x, self.fitness)
        return str

    def __repr__(self):
        """Will also make it printable if it is an entry in a list"""
        return self.__str__() + '\n'
