import argparse
import logging

from EA import EA
from utils import micro17flowerConfigSpace, setup_seed
import os
from target import CNN
from datetime import datetime
import wandb
import json

if __name__ == '__main__':
    """
    AutoML Optimizer made by Carlos Marañes. The chosen implementation has been
    a variation of an Evolutionary Algorithm with Hyperband which supports constraints.
    By adding Hyperband, bad configurations are cut off early. If a configuration with the specified constraints 
    is not satisfied the algorithm returns the one with the best found precision.
    
    This script should be called as following to match the reported results on the slides
    $ python AutoML.py --pop_size 50 --num_children 10 --hyperband True
    
    When not using the hyperband parameter, it will just perform a regular evolutionary strategy
    
    After passing max_time, it will output 4 files:
        - automl_run_{moreInfo}.json -> includes all the fully evaluated configurations
        - opt_cfg_{moreInfo}.json -> includes the optimal found configuration that can be evaluated in main.py
        - {moreInfo}.csv -> information about how the algorithm evolves over time for visualization purposes
    During the run, it will also create a {moreInfo} folder which stores the weights of the partially trained algorithm
    to avoid high GPU consumption when doing hyperband.
    """

    cmdline_parser = argparse.ArgumentParser('AutoML SS21 final project - EA-Hyperband')

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
                                default=123,
                                help='Random seed for reproducibility',
                                type=int)
    cmdline_parser.add_argument('-w', '--wandb_api_key',
                                default=None,
                                help='WanDB API Key for logging',
                                type=str)
    # EA arguments
    cmdline_parser.add_argument('--pop_size',
                                default=10,
                                help='Population size',
                                type=int)
    cmdline_parser.add_argument('-c', '--num_children',
                                default=5,
                                help='Number of children per step',
                                type=int)
    cmdline_parser.add_argument('--hyperband',
                                default=False,
                                help='Use hyperband',
                                type=bool)

    args, unknowns = cmdline_parser.parse_known_args()

    # Logging
    logging.basicConfig(level=args.verbose)

    # Constraints
    constraint_model_size = args.constraint_max_model_size
    constraint_precision = args.constraint_min_precision
    constraint_epochs = args.epochs
    constraint_running_time = args.max_time

    # EA parameters
    pop_size = args.pop_size
    num_children = args.num_children
    if args.hyperband:
        logging.info("Hyperband enabled. Number of children is ignored")

    # Seed for reproducibility purposes
    random_seed = args.seed
    setup_seed(random_seed)

    # Define the CNN search space
    search_space = micro17flowerConfigSpace(random_seed)

    # Run name
    date_now = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    run_name = f"popSize_{pop_size}_numChildren_{num_children}_hyperband_{args.hyperband}_" \
               f"minPrecision_{constraint_precision}_maxModelSize_{constraint_model_size}_" \
               f"seed_{random_seed}_date_{date_now}"

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

    ea = EA(search_space, target_function, run_name, population_size=pop_size, max_runtime=constraint_running_time,
            min_precision=constraint_precision, max_num_params=constraint_model_size,
            children_per_step=num_children, hyperband=args.hyperband, max_budget=constraint_epochs,
            random_seed=random_seed, wandb_log=wandb_log)
    optimum = ea.optimize()

    with open(f'opt_cfg_{run_name}.json', 'w') as f:
        json.dump(optimum, f)

