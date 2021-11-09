import os

from member import Member, Mutation, Recombination, ParentSelection
import logging
from utils import sort_population, save_members, plot_grey_box_optimization, table_precision_accuracy, \
    table_numParams_precision
import numpy as np
import time
import wandb
import csv
import shutil


class EA:
    def __init__(self, search_space, target_func, run_name, population_size: int = 10,
                 mutation_type: Mutation = Mutation.UNIFORM,
                 recombination_type: Recombination = Recombination.UNIFORM,
                 recom_proba: float = 0.5, selection_type: ParentSelection = ParentSelection.NEUTRAL,
                 max_runtime: int = 3600, children_per_step: int = 5,
                 fraction_mutation: float = .5,
                 max_num_params: int = 2e7,
                 min_precision: int = 0.39,
                 wandb_log=False,
                 hyperband=False,
                 max_budget: int = 50,
                 random_seed: int = 0
                 ):
        """
        Evolutionary algorithm wich allows hyperband
        :param search_space: Search space of the type ConfigSpace
        :param target_func: callable target function we optimize, in this case a CNN
        :param run_name: Name of the run
        :param population_size: int
        :param mutation_type: hyperparameter to set mutation strategy
        :param recombination_type: hyperparameter to set recombination strategy
        :param recom_proba: conditional hyperparameter dependent on recombination_type UNIFORM
        :param selection_type: hyperparameter to set selection strategy
        :param max_runtime: maximum runtime in seconds
        :param children_per_step: how many children to produce per step. With Hyperband this is ignored
        :param fraction_mutation: balance between sexual and asexual reproduction
        :param max_num_params: Maximum number of the parameters of the CNN
        :param min_precision: Minimum precision to achieve
        :param wandb_log: True if we log in wandb. False and it will no log in wandb
        :param hyperband: True if hyperband is applied
        :param max_budget: Maximum epochs that a model can be trained
        :param random_seed: Random seed for reproducibility purposes
        """
        assert 0 <= fraction_mutation <= 1
        assert 0 <= recom_proba <= 1
        assert 0 < children_per_step
        assert 0 < max_runtime
        assert 0 < population_size
        assert 0 < max_num_params
        assert 0 < min_precision <= 1
        assert 0 < max_budget

        # Constraints
        self.max_num_param = max_num_params
        self.min_precision = min_precision
        self.max_budget = max_budget

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        self.start_time = time.time()

        # Step 1: initialize Population
        population = []
        for i in range(population_size):
            self.logger.info(f"Creating member {i + 1} of {population_size}")
            member = Member(search_space, None, target_func, mutation_type, recombination_type, recom_proba,
                            max_num_params=self.max_num_param)
            # We train the initial population
            member.train_n_epoch(self.max_budget)
            population.append(member)

        save_members(population, run_name)
        self.population = population
        self.population = sort_population(self.population, self.min_precision)
        self.pop_size = population_size
        self.selection = selection_type
        self.max_runtime = max_runtime
        self.num_children = children_per_step
        self.frac_mutants = fraction_mutation
        self.run_name = run_name
        self.random_seed = random_seed

        mean_precision, mean_accuracy = self.get_average_fitness()
        self.logger.info(f"Initial average precision {mean_precision} and accuracy {mean_accuracy} of the population")
        self._architectures = []

        # online logging
        self.wandb_log = wandb_log
        # Logging csv
        self.csv_name = f"{self.run_name}.csv"
        headers = ["avg_acc", "best_acc", "avg_prec", "best_prec", "step", "time"]
        with open(self.csv_name, 'a') as f:
            w = csv.writer(f)
            w.writerow(headers)

        # Hyperband
        self.hyperband = hyperband

    def get_average_fitness(self) -> (float, float):
        """Helper to quickly access average population fitness"""
        mean_precision = np.mean(list(map(lambda x: x.get_fitness()['precision'], self.population)))
        mean_accuracy = np.mean(list(map(lambda x: x.get_fitness()['acc'], self.population)))
        return mean_precision, mean_accuracy

    def select_parents(self, n_parents=None):
        """
        Method that implements all selection mechanism.
        For ease of computation we assume that the population members are sorted according to their fitness
        :return: list of ids of selected parents.
        """
        if n_parents is None:
            n_parents = self.num_children
        parent_ids = []
        if self.selection == ParentSelection.NEUTRAL:
            parent_ids = np.random.choice(self.pop_size, n_parents, replace=False)
        elif self.selection == ParentSelection.FITNESS:
            raise NotImplementedError
        elif self.selection == ParentSelection.TOURNAMENT:
            for i in range(n_parents):  # we want N many children
                # draw random tournament
                # sorted -> id gives us fitness -> min gets fittest member
                parent_id = np.min(np.random.choice(self.pop_size, max(2, n_parents), replace=False))
                parent_ids.append(parent_id)
        else:
            raise NotImplementedError
        self.logger.debug('Selected parents:')
        self.logger.debug(parent_ids)
        return parent_ids

    def step(self) -> (float, float):
        """
        Performs one step of an Evolutionary Algorithm
        parent selection -> offspring creation -> survival selection
        :return: average population fitness
        """
        # Step 2: Parent selection
        parent_ids = self.select_parents()

        # Step 3: Variation / create offspring
        children = []
        for parent_id in parent_ids:
            # determine if more asexual or sexual reproduction determines offspring
            random_number = np.random.random()
            if random_number < self.frac_mutants:
                child = self.population[parent_id].mutate()
            else:
                child = self.population[parent_id].recombine(self.population[np.random.choice(parent_ids)])
            child.train_n_epoch(self.max_budget)
            children.append(child)
            self._architectures.append(child.get_fitness())

        # Step 4: Survival selection
        # (\mu + \lambda)-selection i.e. combine offspring and parents in one sorted list, keep the #pop_size best

        # Save children as evaluations
        save_members(children, self.run_name)

        self.population.extend(children)
        self.population = sort_population(self.population, self.min_precision)
        self.population = self.population[:self.pop_size]
        return self.get_average_fitness()

    def step_hyperband(self) -> (float, float):
        """
        Performs one step of an Evolutionary Algorithm with Hyperband
        parent selection -> offspring creation -> survival selection
        :return: average population fitness
        """
        max_budget_per_model = self.max_budget
        min_budget_per_model = 2
        eta = 2
        s_max = np.floor(np.log(max_budget_per_model / min_budget_per_model) / np.log(eta)).astype(np.int)
        configs_dicts = []
        children = []
        for s in reversed(range(s_max + 1)):
            n_models = np.ceil((s_max + 1) / (s + 1) * eta ** s).astype(np.int)
            min_budget_per_model_iter = eta ** (-s) * max_budget_per_model

            # ----- Successive Halving -----

            # 2. Sample n_models from the parent population
            parent_ids = self.select_parents(n_models)

            # 3. Step 3: Variation / create offspring
            children_iter = []
            for parent_id in parent_ids:
                # determine if more asexual or sexual reproduction determines offspring
                if np.random.random() < self.frac_mutants:
                    child = self.population[parent_id].mutate()
                else:
                    child = self.population[parent_id].recombine(self.population[np.random.choice(parent_ids)])
                children_iter.append(child)

            # 4. Successive Halving
            configs_dict = {i: {'config': children_iter[i],
                                'f_evals': {}} for i in range(n_models)}

            configs_to_eval = list(range(n_models))
            b_previous = 0
            b = np.int(min_budget_per_model_iter)
            while b <= max_budget_per_model:
                for config_id in configs_to_eval:
                    fitness = children_iter[config_id].train_n_epoch(n_epochs=b - b_previous)
                    configs_dict[config_id]['f_evals'][b] = {}
                    configs_dict[config_id]['f_evals'][b]['precision'] = fitness['precision']
                    configs_dict[config_id]['f_evals'][b]['acc'] = fitness['acc']

                # Number of configs to proceed
                num_configs_to_proceed = np.floor(len(configs_to_eval) / eta).astype(np.int)

                eval_configs_curr_budget = dict(
                    filter(lambda config_id_dict: config_id_dict[0] in configs_to_eval, configs_dict.items()))

                # Select the config to evaluate on the higher budget.
                # As we are maximizing, we take by the tail
                precision_satisfied = {}
                precision_no_satisfied = {}
                for config_id_dict in eval_configs_curr_budget.keys():
                    precision = eval_configs_curr_budget[config_id_dict]['f_evals'][b]['precision']
                    if precision >= self.min_precision:
                        precision_satisfied[config_id_dict] = eval_configs_curr_budget[config_id_dict]
                    else:
                        precision_no_satisfied[config_id_dict] = eval_configs_curr_budget[config_id_dict]

                precision_satisfied_sorted = list(
                    dict(sorted(precision_satisfied.items(),
                                key=lambda config_id_dict: config_id_dict[1]['f_evals'][b]['acc'])).keys())
                precision_no_satisfied_sorted = list(
                    dict(sorted(precision_no_satisfied.items(),
                                key=lambda config_id_dict: config_id_dict[1]['f_evals'][b]['precision'])).keys())
                configs_to_eval = precision_no_satisfied_sorted + precision_satisfied_sorted
                b_previous = b
                b = np.round(b * eta).astype(np.int)
                if b <= max_budget_per_model:
                    configs_to_eval = configs_to_eval[-num_configs_to_proceed:]

            children.extend([children_iter[i] for i in configs_to_eval])
            configs_dicts.append(configs_dict)
            # Clean model weights since they are not going to be needed
            for f in os.listdir(self.run_name):
                os.remove(os.path.join(self.run_name, f))

        # plot_grey_box_optimization(configs_dicts, min_budget_per_model=2)

        for i in range(0, len(children)):
            self.logger.debug(f"Children {i} trained {children[i].n_trained_epochs} epochs")

        # Step 4: Survival selection
        # (\mu + \lambda)-selection i.e. combine offspring and parents in one sorted list, keep the #pop_size best
        # Save children as evaluations
        save_members(children, self.run_name)
        self.population.extend(children)
        self.population = sort_population(self.population, self.min_precision)
        self.population = self.population[:self.pop_size]

        return self.get_average_fitness()

    def log_population(self, avg_precision, avg_accuracy, step, time_elapsed):
        """
        Log the avg_precison and avg_accuracy of the population per step and timestamp in a csv file and in
        WanDB if configured
        """
        best_precision = self.population[0].get_fitness()['precision']
        best_accuracy = self.population[0].get_fitness()['acc']
        if self.wandb_log:
            precision_accuracy = table_precision_accuracy('automl_run_%s.json' % self.run_name)
            table = wandb.Table(data=precision_accuracy, columns=["precision", "accuracy"])
            precisions_accuracies = wandb.plot.scatter(table, x="precision", y="accuracy",
                                                       title="Precisions - Accuracies")

            precision_numParams = table_numParams_precision('automl_run_%s.json' % self.run_name)
            table = wandb.Table(data=precision_numParams, columns=["numParams", "precision"])
            precisions_numParams = wandb.plot.scatter(table, x="numParams", y="precision",
                                                      title="NumParams - Precision")
            wandb.log(data={"best_precision": best_precision,
                            "best_accuracy": best_accuracy,
                            "avg_precision": avg_precision,
                            "avg_accuracy": avg_accuracy,
                            "time_elapsed": time_elapsed,
                            "Precision - Accuracies": precisions_accuracies,
                            "Num Params - Precision": precisions_numParams},
                      step=int(time_elapsed))  # As time_elapsed in seconds should not be a problem

        # Log for visualization
        row = [avg_accuracy, best_accuracy, avg_precision, best_precision, step, int(time_elapsed)]
        with open(self.csv_name, 'a') as f:
            w = csv.writer(f)
            w.writerow(row)

    def optimize(self):
        """
        Optimization loop that stops after a runtime threshold
        :return:
        """
        step = 0
        time_elapsed = time.time() - self.start_time
        # Log initial population
        avg_precision, avg_accuracy = self.get_average_fitness()
        self.logger.info("Logging initial population...")
        self.log_population(avg_precision, avg_accuracy, step, time_elapsed)

        step = 1
        while time_elapsed < self.max_runtime:
            if not self.hyperband:
                avg_precision, avg_accuracy = self.step()
            else:
                avg_precision, avg_accuracy = self.step_hyperband()

            time_elapsed = time.time() - self.start_time
            self.log_population(avg_precision, avg_accuracy, step, time_elapsed)
            self.logger.info(f"Elapsed time: {time_elapsed}, Maximum time: {self.max_runtime}")
            step += 1

            # Clean model weights since they are not going to be needed
            for f in os.listdir(self.run_name):
                os.remove(os.path.join(self.run_name, f))

        self.logger.info("Search finished!")
        best_accuracy = self.population[0].get_fitness()['acc']
        best_precision = self.population[0].get_fitness()['precision']
        num_params = self.population[0].get_fitness()['num_params']
        self.logger.info(f"Accuracy {best_accuracy} | Precision {best_precision} | Num Params {num_params}")

        # Remove dir with weights for cleaning
        shutil.rmtree(self.run_name)

        return self.population[0].x.get_dictionary()
