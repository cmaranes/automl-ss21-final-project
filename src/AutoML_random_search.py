import argparse
import logging

from EA import EA
from utils import micro17flowerConfigSpace, setup_seed, table_precision_accuracy, \
    table_numParams_precision, save_members
import os
from target import CNN
from datetime import datetime
import wandb
import json
import time
import csv
from member import Member
import shutil

class RandomSearch:
    def __init__(self, search_space, target_func, run_name,
                 max_num_params: int = 2e7,
                 min_precision: int = 0.39,
                 max_runtime: int = 3600,
                 wandb_log=False,
                 max_budget: int = 50,
                 random_seed: int = 0
                 ):
        """
        Random Search
        :param search_space: Search space of the type ConfigSpace
        :param target_func: callable target function we optimize, in this case a CNN
        :param run_name: Name of the run
        :param max_num_params: Maximum number of the parameters of the CNN
        :param min_precision: Minimum precision to achieve
        :param wandb_log: True if we log in wandb. False and it will no log in wandb
        :param max_budget: Maximum epochs that a model can be trained
        :param random_seed: Random seed for reproducibility purposes
        """

        # Constraints
        self.max_num_param = max_num_params
        self.min_precision = min_precision
        self.max_budget = max_budget
        self.max_runtime = max_runtime

        self.search_space = search_space
        self.target_func = target_func
        self.best_member = None

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        self.start_time = time.time()

        self.population = []
        self.run_name = run_name
        self.random_seed = random_seed

        # online logging
        self.wandb_log = wandb_log
        # Logging csv
        self.csv_name = f"{self.run_name}.csv"
        headers = ["best_acc", "best_prec", "step", "time"]
        with open(self.csv_name, 'a') as f:
            w = csv.writer(f)
            w.writerow(headers)


    def log_best_member(self, step, time_elapsed):
        best_precision = self.best_member.get_fitness()['precision']
        best_accuracy = self.best_member.get_fitness()['acc']
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
                            "time_elapsed": time_elapsed,
                            "Precision - Accuracies": precisions_accuracies,
                            "Num Params - Precision": precisions_numParams},
                      step=int(time_elapsed))

        # Log for visualization
        row = [best_accuracy, best_precision, step, int(time_elapsed)]
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
        while time_elapsed < self.max_runtime:
            self.logger.info(f"Creating member {step}")
            member = Member(self.search_space, None, self.target_func, max_num_params=self.max_num_param)
            # We train the initial population
            member.train_n_epoch(self.max_budget)
            save_members([member], self.run_name)
            if self.best_member is None:
                self.best_member = member
            else:
                best_member_precision = self.best_member.get_fitness()['precision']
                best_member_accuracy = self.best_member.get_fitness()['acc']
                new_member_precision = member.get_fitness()['precision']
                new_member_accuracy = member.get_fitness()['acc']

                # Both satisfy precision
                if best_member_precision >= self.min_precision and new_member_precision >= self.min_precision:
                    if new_member_accuracy > best_member_accuracy:
                        self.best_member = member
                # Only the new member satisfy precision
                elif best_member_precision < self.min_precision <= new_member_precision:
                    self.best_member = member
                # The new member does not satisfy precision
                elif best_member_precision >= self.min_precision > new_member_precision:
                    pass
                # None meet the precision constraint
                else:
                    if new_member_precision > best_member_precision:
                        self.best_member = member

            time_elapsed = time.time() - self.start_time
            self.log_best_member(step, time_elapsed)
            self.logger.info(f"Elapsed time: {time_elapsed}, Maximum time: {self.max_runtime}")
            step += 1

            # Clean model weights since they are not going to be needed
            for f in os.listdir(self.run_name):
                os.remove(os.path.join(self.run_name, f))

        self.logger.info("Search finished!")
        best_accuracy = self.best_member.get_fitness()['acc']
        best_precision = self.best_member.get_fitness()['precision']
        num_params = self.best_member.get_fitness()['num_params']
        self.logger.info(f"Accuracy {best_accuracy} | Precision {best_precision} | Num Params {num_params}")

        # Remove dir with weights for cleaning
        shutil.rmtree(self.run_name)

        return self.best_member.x.get_dictionary()

if __name__ == '__main__':
    """
    Random Search baseline made by Carlos Maranes. The implementation do not evaluate configurations that do not
    meet the parameters constraint.
    
    This file has been done with the intention of being used as a baseline to compare the main optimizer with.
    """

    cmdline_parser = argparse.ArgumentParser('AutoML SS21 final project - Random Search Baseline')

    # Should be 50 according to the script of the project
    cmdline_parser.add_argument('-e', '--epochs',
                                default=50,
                                help='Maximum Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-s', "--constraint_max_model_size",
                                default=2e7,
                                help="maximal model size constraint",
                                type=int)
    cmdline_parser.add_argument('-p', "--constraint_min_precision",
                                default=0.39,
                                help='minimal constraint constraint',
                                type=float)
    cmdline_parser.add_argument('-t', "--max_time",
                                default=86400,
                                help='Maximum running time (seconds)',
                                type=int)
    cmdline_parser.add_argument("--seed",
                                default=0,
                                help='Random seed for reproducibility',
                                type=int)
    cmdline_parser.add_argument('-w', '--wandb_api_key',
                                default=None,
                                help='WanDB API Key for logging',
                                type=str)

    args, unknowns = cmdline_parser.parse_known_args()

    # Logging
    logging.basicConfig(level=args.verbose)

    # Constraints
    constraint_model_size = args.constraint_max_model_size
    constraint_precision = args.constraint_min_precision
    constraint_epochs = args.epochs
    constraint_running_time = args.max_time


    # Seed for reproducibility purposes
    random_seed = args.seed
    setup_seed(random_seed)

    # Define the CNN search space
    search_space = micro17flowerConfigSpace(random_seed)

    # Run name
    date_now = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    run_name = f"randomSearch_seed_{random_seed}_date_{date_now}"

    # Define the target function. In this case, the CNN
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower')
    target_function = CNN(data_dir=data_dir, run_name=run_name, seed=random_seed)  # The target is the CNN

    # Login WANDB
    wandb_log = False
    if args.wandb_api_key is not None:
        wandb_log = True
        wandb.login(key=args.wandb_api_key)
        # https://docs.wandb.ai/v/master/library/init
        wandb.init(project="AutoML", name=run_name)

    rs = RandomSearch(search_space, target_function, run_name, max_runtime=constraint_running_time,
                      max_budget=constraint_epochs, random_seed=random_seed, wandb_log=wandb_log)
    optimum = rs.optimize()

    with open(f'opt_cfg_{run_name}.json', 'w') as f:
        json.dump(optimum, f)

