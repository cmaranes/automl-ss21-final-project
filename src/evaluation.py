from utils import visualize_evaluations
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import json

import logging
from collections import OrderedDict
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold   # We use 3-fold stratified cross-validation
import time

from cnn import torchModel

plot_type = 2

if plot_type == 0:
    # Runs to compare and parameters
    ea_hyperband = ['popSize_50_numChildren_10_hyperband_True_minPrecision_0.39_seed_42_date_2021_08_31-05:08:57_PM',
                    'popSize_50_numChildren_10_hyperband_True_minPrecision_0.39_seed_0_date_2021_09_01-05:01:24_PM',
                    'popSize_50_numChildren_10_hyperband_True_minPrecision_0.39_seed_123_date_2021_09_02-04:36:25_PM'] # Best
    ea = ['popSize_50_numChildren_10_hyperband_False_minPrecision_0.39_seed_42_date_2021_08_31-05:09:44_PM', # Best
          'popSize_50_numChildren_10_hyperband_False_minPrecision_0.39_seed_0_date_2021_09_01-05:01:21_PM',
          'popSize_50_numChildren_10_hyperband_False_minPrecision_0.39_seed_123_date_2021_09_02-04:36:25_PM']
    random_search = ['randomSearch_seed_42_date_2021_08_31-05:10:47_PM',
                     'randomSearch_seed_0_date_2021_09_01-05:01:32_PM', # Best
                     'randomSearch_seed_123_date_2021_09_02-04:36:25_PM']
    runs_folder = 'runs'

    # Constraints
    constraint_min_precision = 0.39
    constraint_max_model_size = 2e7

    max_time = 86400
    resolution = 10000
    plot_types = [('best_acc', 'Best Accuracy'), ('best_prec', 'Best Precision')]
    #plot_types = [('best_acc', 'Best Accuracy'), ('avg_acc', 'Population average accuracy'),
    #              ('best_prec', 'Best Precision'), ('avg_prec', 'Population average precision')]
    #algorithms = [(ea_hyperband, 'EA-Hyperband'), (ea, 'EA')]
    #algorithms = [ea, ea_hyperband, random_search]
    algorithms = [(ea_hyperband, 'EA-Hyperband'), (ea, 'EA'), (random_search, 'Random Search')]
    #plot_types = ['avg_acc', 'best_acc', 'avg_prec', 'best_prec']
    #algorithms = [(ea_hyperband, 'EA-Hyperband'), (ea, 'EA')]

elif plot_type == 1:  # Comparison different precisions
    # Runs to compare and parameters
    prec_39 = ['popSize_50_numChildren_10_hyperband_True_minPrecision_0.39_seed_42_date_2021_08_31-05:08:57_PM',
                    'popSize_50_numChildren_10_hyperband_True_minPrecision_0.39_seed_0_date_2021_09_01-05:01:24_PM',
                    'popSize_50_numChildren_10_hyperband_True_minPrecision_0.39_seed_123_date_2021_09_02-04:36:25_PM']
    prec_40 = ['popSize_50_numChildren_10_hyperband_True_minPrecision_0.4_maxModelSize_20000000.0_seed_0_date_2021_09_08-09:59:05_AM',
               'popSize_50_numChildren_10_hyperband_True_minPrecision_0.4_maxModelSize_20000000.0_seed_42_date_2021_09_09-10:15:30_AM',
               'popSize_50_numChildren_10_hyperband_True_minPrecision_0.4_maxModelSize_20000000.0_seed_123_date_2021_09_07-09:20:39_AM']
    prec_42 = ['popSize_50_numChildren_10_hyperband_True_minPrecision_0.42_maxModelSize_20000000.0_seed_0_date_2021_09_08-09:58:33_AM',
               'popSize_50_numChildren_10_hyperband_True_minPrecision_0.42_maxModelSize_20000000.0_seed_42_date_2021_09_08-09:58:40_AM',
               'popSize_50_numChildren_10_hyperband_True_minPrecision_0.42_seed_123_date_2021_09_06-09:44:23_AM']
    runs_folder = 'runs_precision'

    # Constraints
    constraint_min_precision = 0.39
    constraint_max_model_size = 2e7

    max_time = 86400
    plot_types = [('best_acc', 'Best Accuracy'), ('avg_acc', 'Population average accuracy'),
                  ('best_prec', 'Best Precision'), ('avg_prec', 'Population average precision')]
    algorithms = [(prec_39, '0.39'), (prec_40, '0.40'), (prec_42, '0.42')]


elif plot_type == 2:  # Others for debugging
    # Runs to compare and parameters
    debug = ['popSize_50_numChildren_10_hyperband_True_minPrecision_0.39_seed_123_date_2021_09_02-04:36:25_PM']
    runs_folder = 'runs_precision'

    # Constraints
    constraint_min_precision = 0.4
    constraint_max_model_size = 2e7

    max_time = 86400
    plot_types = [('best_acc', 'Best Accuracy'), ('avg_acc', 'Population average accuracy'),
                  ('best_prec', 'Best Precision'), ('avg_prec', 'Population average precision')]
    algorithms = [(debug, 'debug')]


# ----- Plots improvement over time -----
for plot_type, plot_name in plot_types:
    functions = []
    times_functions = []
    for algorithm_runs, algorithm_name in algorithms:
        for run in algorithm_runs:
            times, y = [], []
            csv_filename = os.path.join(runs_folder, f'{run}.csv')
            with open(csv_filename, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        # print(f'Column names are {", ".join(row)}')
                        line_count += 1
                        continue
                    times.append(float(row['time']))
                    y.append(float(row[plot_type]))
                    line_count += 1
            f = interpolate.interp1d(np.array(times), np.array(y), kind='previous', fill_value=(y[0], y[-1]), bounds_error=False)
            functions.append(f)
            times.sort()
            times_functions.extend(times)

        means = []
        stds = []
        times_functions.sort()
        for t in times_functions:
            point_ys = []
            for function_interpolated in functions:
                point_ys.append(function_interpolated(t))
            means.append(np.mean(point_ys))
            stds.append(np.std(point_ys))

        # https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
        plt.step(times_functions, means, label=algorithm_name)
        #plt.xscale('log')
        plt.fill_between(times_functions, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.3, step='pre')

    plt.xlabel('Time(s)')
    plt.ylabel(plot_name)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(runs_folder, f'{plot_type}_time.png'))
    #plt.show()
    plt.clf()


# ------ Plots configurations evaluated -------
for algorithm_runs, algorithm_name in algorithms:
    percentage_ok_evaluations = []  # Percentage of evaluated configurations that meet the constraints
    for run in algorithm_runs:
        ok_evaluations = 0
        non_ok_evaluations = 0
        json_file = os.path.join(runs_folder, f"automl_run_{run}.json")
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
            if params < constraint_max_model_size and prec >= constraint_min_precision:
                colors.append('green')
                ok_evaluations += 1
            elif params < constraint_max_model_size:
                non_ok_evaluations += 1
                colors.append('orange')
            else:
                non_ok_evaluations += 1
                colors.append('blue')

        percentage_ok = ok_evaluations / (ok_evaluations + non_ok_evaluations)
        percentage_ok_evaluations.append(percentage_ok)
        print(run)
        print(f"OK EVALUATIONS {ok_evaluations}, NON-OK-EVALUATIONS {non_ok_evaluations}")
        print(f"Percentage OK {percentage_ok}")


        plt.scatter(precisions, accuracies, c=colors, alpha=0.3)
        plt.axvline(x=constraint_min_precision, c='r')
        plt.xlabel("Precision")
        plt.xlim(left=0, right=1)
        plt.ylabel("Accuracy")
        plt.ylim(bottom=0, top=1)
        plt.savefig(os.path.join(runs_folder, f'precision-accuracy-{run}.png'))
        plt.clf()

        colors = []
        for i in range(0, len(data)):
            element = data[i]
            prec = element['precision']
            params = element['num_params']
            if prec >= constraint_min_precision and params <= constraint_max_model_size:
                colors.append('green')
            elif prec >= constraint_min_precision:
                colors.append('orange')
            else:
                colors.append('blue')

        plt.scatter(parameters, accuracies, c=colors, alpha=0.3)
        plt.axvline(x=constraint_max_model_size, c='r')
        plt.xlabel("Number of parameters")
        plt.xlim(left=0, right=constraint_max_model_size)
        plt.ylabel("Accuracy")
        plt.ylim(bottom=0, top=1)
        plt.savefig(os.path.join(runs_folder, f'numParams-accuracy-{run}.png'))
        plt.legend()
        plt.clf()

    print(f"{algorithm_name} percentage OK, mean: {np.mean(percentage_ok_evaluations)}, std: {np.std(percentage_ok_evaluations)}")

# ----------------- TRAINING AND EVALUATION OF CONFIGURATION FOUND ------------
def evaluate_cfg(model_config,
         data_dir,
         use_teset_data=False,
         num_epochs=10,
         batch_size=50,
         learning_rate=0.001,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.Adam,
         constraints=None,
         data_augmentations=None,
         save_model_str=None):
    """
    Training loop for configurableNet.
    :param model_config: network config (dict)
    :param data_dir: dataset path (str)
    :param use_teset_data: if we use a separate test dataset or crossvalidation
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during trainnig (torch.optim.Optimizer)
    :param constraints: Constraints that needs to be fulfilled, the order determines the degree of difficulty
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :return:
    """

    # Device configuration
    if constraints is None:
        constraints = OrderedDict([('model_size', 5e7), ('precision', 0.61)])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_width = 16
    img_height = 16
    #data_augmentations = [transforms.Resize([img_width, img_height]),
    #                      transforms.ToTensor()]
    if data_augmentations is None:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    elif isinstance(data_augmentations, list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    # Load the dataset
    tv_data = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentations)
    test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=data_augmentations)

    cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # to make CV splits consistent

    train_sets = []
    val_sets = []
    if use_teset_data:
        train_sets.append(tv_data)
        val_sets.append(test_data)
    else:
        for train_idx, valid_idx in cv.split(tv_data, tv_data.targets):
            train_sets.append(Subset(tv_data, train_idx))
            val_sets.append(Subset(tv_data, valid_idx))

    # instantiate training criterion
    train_criterion = train_criterion().to(device)
    scores_accuracy = []
    scores_precision = []

    num_classes = len(tv_data.classes)
    #image size
    input_shape = (3, img_width, img_height)
    for train_set, val_set in zip(train_sets, val_sets):
        # Make data batch iterable
        # Could modify the sampler to not uniformly random sample
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                shuffle=False)

        model = torchModel(model_config,
                           input_shape=input_shape,
                           num_classes=num_classes).to(device)

        total_model_params = np.sum(p.numel() for p in model.parameters())

        # instantiate optimizer
        optimizer = model_optimizer(model.parameters(),
                                    lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, device)
            score, _, score_precision, _ = model.eval_fn(val_loader, device)

        score_accuracy_top3, _, score_precision, predictions = model.eval_fn(val_loader, device)

        scores_accuracy.append(score_accuracy_top3)
        scores_precision.append(np.mean(score_precision))

        if save_model_str:
            # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
            if os.path.exists(save_model_str):
                save_model_str += '_'.join(time.ctime())
            torch.save(model.state_dict(), save_model_str)

    # RESULTING METRIC
    # RESULTING METRIC
    # RESULTING METRIC
    optimized_metrics = {"model_size": total_model_params,
                         "precision": np.mean(scores_precision),
                         "top3_accuracy": np.mean(scores_accuracy)}
    print(optimized_metrics)
    for constraint_name in constraints:
        if constraint_name == 'model_size':
            # HERE IS THE CONSTRAINT THAT MUST BE SATISFIED
            assert optimized_metrics[constraint_name] <= constraints[constraint_name], \
                "Number of parameters exceeds model size constraints!"
        else:
            if use_teset_data:
                logging.info("Constraints are checked on a separate test set")
            else:
                logging.info("Constraints are checked on a cross validation sets ")
            logging.info(f"The constraint {constraint_name}: "
                         f"{optimized_metrics[constraint_name]} >= {constraints[constraint_name]} is satisfied? "
                         f"{optimized_metrics[constraint_name] >= constraints[constraint_name]}")

    print('Resulting Model Score:')
    print(' acc [%]')
    print(optimized_metrics['top3_accuracy'])

    if use_teset_data:
        with open('result_test.json', 'w') as f:
            json.dump(optimized_metrics, f)
    else:
        with open('result_cv.json', 'w') as f:
            json.dump(optimized_metrics, f)
    return predictions

logging.basicConfig(level='INFO')

# Default network
print("EVALUATING DEFAULT NETWORK")
architecture_default = {
        'n_conv_layers': 3,
        'n_channels_conv_0': 457,
        'n_channels_conv_1': 511,
        'n_channels_conv_2': 38,
        'kernel_size': 5,
        'global_avg_pooling': True,
        'use_BN': False,
        'n_fc_layers': 2,
        'n_channels_fc_0': 27,
        'n_channels_fc_1': 17,
        'n_channels_fc_2': 273,
        'dropout_rate': 0.2}
predictions_default = evaluate_cfg(
    architecture_default,
    data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower'),
    use_teset_data=True,
    num_epochs=50,
    batch_size=290,
    learning_rate=0.0016651181400389517,
    train_criterion=torch.nn.CrossEntropyLoss,
    model_optimizer=torch.optim.Adam,
    data_augmentations=None,  # Not set in this example
    constraints=OrderedDict([('model_size', constraint_max_model_size), ('precision', constraint_min_precision)]),
    )

predictions_default = ('default', predictions_default)

def mcNemarTest(model1, model2):
    print("Comparing:")
    print(model1[0])
    print(model2[0])
    #A = sum(np.array(model1[1]) == np.array(model2[1]))
    B = sum((np.logical_not(np.array(model2[1]))) & np.array(model1[1]))
    C = sum(np.array(model2[1]) & np.logical_not(np.array(model1[1])))
    #D = sum((not np.array(model1[1])) == (not np.array(model2[1])))
    print(f"McNemar test possible {B + C} > 20 -> {(B + C > 20)}")
    chi_value = ((np.abs(B - C) -1)**2) / (B + C)
    print(f"{chi_value} > 3.841")
    print("If higher, we reject the hypothesis that the tree and the random forest have the same performance")

predictions = []
for algorithm_runs, algorithm_name in algorithms:
    for run in algorithm_runs:
        print(f"Evaluating {algorithm_name}: {run}")
        with open(os.path.join(runs_folder, f"opt_cfg_{run}.json"), 'r') as f:
            opt_cfg = json.load(f)
        predictions_algorithm = evaluate_cfg(
            opt_cfg,
            data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower'),
            use_teset_data=True,
            num_epochs=50,
            batch_size=opt_cfg['batch_size'],
            learning_rate=opt_cfg['learning_rate_init'],
            train_criterion=torch.nn.CrossEntropyLoss,
            model_optimizer=torch.optim.Adam,
            data_augmentations=None,  # Not set in this example
            constraints=OrderedDict(
                [('model_size', constraint_max_model_size), ('precision', constraint_min_precision)]),
        )
        predictions.append((run, predictions_algorithm))
        #mcNemarTest(predictions[0], predictions_default)

# -------- McNemar Test ----------
# This is manually selected

print('#' * 20)
print('McNemar EA-HB, default')
mcNemarTest(predictions[2], predictions_default)
print('#' * 20)
print('McNemar EA-HB, Random')
mcNemarTest(predictions[2], predictions[8])
print('#' * 20)
print('McNemar EA-HB, EA')
mcNemarTest(predictions[2], predictions[5])