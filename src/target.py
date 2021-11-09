from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as transforms
import os
from cnn import torchModel
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, Subset
import time


def get_optimizer_and_crit(cfg):
    if cfg['optimizer'] == 'AdamW':
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg['train_criterion'] == 'mse':
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss
    return model_optimizer, train_criterion


class CNN:
    """
    Target function to optimize.
    It includes the CNN and the dataset
    """

    def __init__(self, data_dir, run_name, seed: int, use_teset_data=False):
        """
        Init
        :param data_dir: Directory containing the dataset
        :param seed: Seed for reproducibility purposes
        """
        self.cv = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)  # to make CV splits consistent
        data_augmentations = transforms.ToTensor()
        self.tv_data = ImageFolder(os.path.join(data_dir, "train"), transform=data_augmentations)
        self.test_data = ImageFolder(os.path.join(data_dir, "test"), transform=data_augmentations)

        # Loading datasets
        self.train_sets = []
        self.val_sets = []
        if use_teset_data:
            self.train_sets.append(self.tv_data)
            self.val_sets.append(self.test_data)
        else:
            for train_idx, valid_idx in self.cv.split(self.tv_data, self.tv_data.targets):
                self.train_sets.append(Subset(self.tv_data, train_idx))
                self.val_sets.append(Subset(self.tv_data, valid_idx))

        img_width, img_height = 16, 16
        self._input_shape = (3, img_width, img_height)
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Using {self._device} device for training")
        self.seed=seed
        self.use_teset_data = use_teset_data
        self.run_name = run_name
        # Create dir with models of the models of this run
        try:
            os.mkdir(self.run_name)
        except OSError:
            self.logger.info("Creation of the directory %s failed" % self.run_name)
        else:
            self.logger.info("Successfully created the directory %s " % self.run_name)

    def model_params(self, cfg) -> int:
        """
        Number of parameters of the configuration
        :param cfg: Configuration to compute the model parameters
        """
        # To check constraints better in CPU, since may there are very big models that could not fit on the GPU
        device = "cpu"
        model = torchModel(cfg,
                           input_shape=self._input_shape,
                           num_classes=len(self.tv_data.classes)).to(device)
        model_params = np.sum(p.numel() for p in model.parameters())
        # Free memory
        # https://stackoverflow.com/questions/52205412/how-to-free-up-all-memory-pytorch-is-taken-from-gpu-memory
        del model
        torch.cuda.empty_cache()
        return model_params

    def satisfy_num_param(self, cfg, max_num_params: int) -> bool:
        """
        Check if the configuration has less than the maximum number of parameters allowed
        :param cfg: Configuration to evaluate
        :param max_num_params: Maximum number of parameters allowed
        """
        return self.model_params(cfg) < max_num_params

    def train(self, cfg, cfg_id, budget: int = 50):
        """
        Train the CNN with config cfg for budget epochs
        :param cfg: Configuration to create the CNN
        :param cfg_id: id of the cfg (Member.id). It is used for storing the model weights in disk
        :param budget: Number of epochs to train the CNN
        """
        start_time = time.time()

        lr = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.001
        batch_size = cfg['batch_size'] if cfg['batch_size'] else 200
        num_epochs = int(np.ceil(budget))

        self.logger.debug(f"Training for {num_epochs} epochs...")
        self.logger.debug(str(cfg))

        scores_accuracy = []
        scores_precision = []
        num_classes = len(self.tv_data.classes)
        for count, (train_set, val_set) in enumerate(zip(self.train_sets, self.val_sets)):
            train_loader = DataLoader(dataset=train_set,
                                      batch_size=batch_size,
                                      shuffle=True)
            val_loader = DataLoader(dataset=val_set,
                                    batch_size=batch_size,
                                    shuffle=False)

            model = torchModel(cfg,
                               input_shape=self._input_shape,
                               num_classes=num_classes,
                               seed=self.seed).to(self._device)

            # Load model, if exists
            model_path = os.path.join(self.run_name, f"cfg_id_{cfg_id}_count_{count}")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))

            # instantiate training criterion
            model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
            optimizer = model_optimizer(model.parameters(),
                                        lr=lr)
            # instantiate training criterion
            train_criterion = train_criterion().to(self._device)

            # Train the model
            for epoch in range(num_epochs):
                train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, self._device)
                score, _, score_precision, _ = model.eval_fn(val_loader, self._device)
                #logging.info('Train accuracy %f', train_score)
                #logging.info('Test accuracy %f', score)

            score_accuracy_top3, _, score_precision, _ = model.eval_fn(val_loader, self._device)
            scores_accuracy.append(score_accuracy_top3)
            scores_precision.append(np.mean(score_precision))
            # Save model into disk
            torch.save(model.state_dict(), model_path)
            del model
            torch.cuda.empty_cache()
            # break

        total_model_params = self.model_params(cfg)
        optimized_metrics = {"num_params": total_model_params,
                             "precision": np.mean(scores_precision),
                             "acc": np.mean(scores_accuracy)}

        time_elapsed = time.time() - start_time
        # count+1 because it starts on 0
        self.logger.debug(f"Training time for {num_epochs} epoch(s) and {count+1} k-folds has been {time_elapsed}")

        return optimized_metrics
