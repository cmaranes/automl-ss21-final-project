import random
import torch
import numpy

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].float().sum()
    res.append(correct_k.mul_(1.0/batch_size))
  return res


# --------- My Utils ----------
# https://automl.github.io/ConfigSpace/master/
import ConfigSpace as CS
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
import json
import matplotlib.pyplot as plt


def micro17flowerConfigSpace(seed=0):
    cs = CS.ConfigurationSpace(seed=seed)

    learning_rate_init = UniformFloatHyperparameter('learning_rate_init',
                                                    0.00001, 0.1, default_value=0.01, log=True)
    batch_size = UniformIntegerHyperparameter('batch_size', 4, 300, default_value=16, log=True)
    kernel_size = OrdinalHyperparameter('kernel_size', sequence=[3, 5, 7])
    use_BN = CategoricalHyperparameter('use_BN', choices=[True, False])
    global_avg_pooling = CategoricalHyperparameter('global_avg_pooling', choices=[True, False])
    dropout_rate = UniformFloatHyperparameter('dropout_rate', 0.01, 0.5, default_value=0.2, log=True)


    # ---- Convolutional Layers ----
    n_conv_layers = UniformIntegerHyperparameter("n_conv_layers", 1, 3, default_value=3)
    # output channel from 16 since the default has 16
    n_channels_conv_0 = UniformIntegerHyperparameter("n_channels_conv_0", 16, 1024, default_value=512, log=True)
    n_channels_conv_1 = UniformIntegerHyperparameter("n_channels_conv_1", 16, 1024, default_value=512, log=True)
    n_channels_conv_2 = UniformIntegerHyperparameter("n_channels_conv_2", 16, 1024, default_value=512, log=True)

    # ---- Fully Connected Layers ----
    n_fc_layers = UniformIntegerHyperparameter("n_fc_layers", 1, 3, default_value=3)
    n_channels_fc_0 = UniformIntegerHyperparameter("n_channels_fc_0", 4, 512, default_value=256, log=True)
    n_channels_fc_1 = UniformIntegerHyperparameter("n_channels_fc_1", 4, 512, default_value=256, log=True)
    n_channels_fc_2 = UniformIntegerHyperparameter("n_channels_fc_2", 4, 512, default_value=256, log=True)

    cs.add_hyperparameters([learning_rate_init, n_conv_layers, n_channels_conv_0, n_channels_conv_1, n_channels_conv_2,
                           n_fc_layers, n_channels_fc_0, n_channels_fc_1, n_channels_fc_2, batch_size,
                            kernel_size, use_BN, global_avg_pooling, dropout_rate])

    # Add conditions to restrict the hyperparameter space
    use_conv_layer_2 = CS.conditions.InCondition(n_channels_conv_2, n_conv_layers, [3])
    use_conv_layer_1 = CS.conditions.InCondition(n_channels_conv_1, n_conv_layers, [2, 3])
    cs.add_conditions([use_conv_layer_2, use_conv_layer_1])

    use_fc_layer_2 = CS.conditions.InCondition(n_channels_fc_2, n_fc_layers, [3])
    use_fc_layer_1 = CS.conditions.InCondition(n_channels_fc_1, n_fc_layers, [2, 3])
    cs.add_conditions([use_fc_layer_2, use_fc_layer_1])

    return cs

# https://stackoverflow.com/questions/52730405/does-pytorch-seed-affect-dropout-layers
def setup_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# A priori sorting
# We are assuming num_params is always satisfied
def sort_population(population, min_precision: int):
    """
    A priori sorting. First members with best accuracy that satisfy the min precision
    and then the best accuracy that do not satisfy the min precision
    :param population: List of members of the population
    :param min_precision: Minimum threshold precision
    """
    precision_satisfied = []
    precision_no_satisfied = []
    for i in range(0, len(population)):
        member = population[i]
        if member.get_fitness()['precision'] >= min_precision:
            precision_satisfied.append(member)
        else:
            precision_no_satisfied.append(member)
    precision_satisfied.sort(key=lambda x: x.get_fitness()['acc'], reverse=True)
    # If precision not satisfied, we search for best precision to meet the constraint
    precision_no_satisfied.sort(key=lambda x: x.get_fitness()['precision'], reverse=True)
    return precision_satisfied + precision_no_satisfied


def save_members(members, run_name):
    data = []
    try:
        with open('automl_run_%s.json' % run_name) as f:
            data = json.load(f)
    except FileNotFoundError:
        print("JSON storing members not found, creating it...")

    with open('automl_run_%s.json' % run_name, 'w') as f:
        for member in members:
            new_data = member.get_info()
            data.append(new_data)
        json.dump(data, f)
        f.write("\n")


def table_precision_accuracy(json_file):
    with open(json_file, "r") as read_file:
        data = json.load(read_file)

    accuracies = []
    precisions = []

    for i in range(0, len(data)):
        element = data[i]
        acc = element['top3']
        prec = element['precision']
        accuracies.append(acc)
        precisions.append(prec)

    data = [[x, y] for (x, y) in zip(precisions, accuracies)]
    return data

def table_numParams_precision(json_file):
    with open(json_file, "r") as read_file:
        data = json.load(read_file)

    num_params = []
    precisions = []

    for i in range(0, len(data)):
        element = data[i]
        precision = element['precision']
        num_param = element['num_params']
        num_params.append(num_param)
        precisions.append(precision)

    data = [[x, y] for (x, y) in zip(num_params, precisions)]
    return data

def visualize_evaluations(json_file):
    with open(json_file, "r") as read_file:
        data = json.load(read_file)

    accuracies = []
    precisions = []
    parameters = []
    colors = []

    for i in range(0, len(data)):
        element = data[i]
        acc = element['top3']
        prec = element['precision']
        params = element['num_params']
        accuracies.append(acc)
        parameters.append(params)
        precisions.append(prec)
        if params < 2 * 1e7 and prec >= 0.39:
            colors.append('green')
        elif params < 2 * 1e7:
            colors.append('orange')
        else:
            colors.append('blue')

    plt.scatter(precisions, accuracies, c=colors)
    plt.axvline(x=0.39, c='r')
    plt.xlabel("Precision")
    plt.ylabel("Accuracy")
    plt.show()

    colors = []

    for i in range(0, len(data)):
        element = data[i]
        prec = element['precision']
        params = element['num_params']
        if prec >= 0.39 and params <= 2 * 1e7:
            colors.append('green')
        elif prec >= 0.39:
            colors.append('orange')
        else:
            colors.append('blue')

    plt.scatter(parameters, accuracies, c=colors)
    plt.axvline(x=2*1e7, c='r')
    plt.xlabel("Number of parameters")
    plt.ylabel("Accuracy")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-whitegrid')


def plot_grey_box_optimization(configs_list, min_budget_per_model):
    if len(configs_list) == 1:
        n_rows, n_cols = 1, 1
        filename = 'successive_halving_results.pdf'
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5, 5), sharex='col', sharey='row')
        axs = [axs]
    else:
        n_hyperband_iter = len(configs_list)
        n_cols = ((n_hyperband_iter - np.mod(n_hyperband_iter, 3)) / 3 + np.mod(n_hyperband_iter, 3)).astype(np.int)
        n_rows = 3
        filename = 'hyperband_results.pdf'
        fig, axs = plt.subplots(3, n_cols, figsize=(n_cols * n_rows, n_rows * 2), sharex='col', sharey='row')
        axs = axs.reshape(-1)

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for idx, (configs_dict, ax) in enumerate(zip(configs_list, axs)):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.3)
        all_budgets = []
        for config_id, config_dict in configs_dict.items():
            budgets = np.array(list(config_dict['f_evals'].keys()))
            val_errors = np.array(list(config_dict['f_evals'].values())).T
            val_errors = [d['acc'] for d in val_errors]
            ax.scatter(budgets, val_errors, s=6)
            ax.plot(budgets, val_errors)
            all_budgets.extend(budgets)

        # Use the same x-axis limits for all subplots for easier comparison.
        ax.set_xlim(min_budget_per_model, 60)

        for budget in np.unique(all_budgets):
            ax.axvline(budget, c='black', lw=0.5)
        if idx == 0:
            ax.set_xlabel('Budget (Epochs)')
            ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(filename)
    return fig