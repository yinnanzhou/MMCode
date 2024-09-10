import time
import torch
import torch.nn as nn


class Trainer:
    def __init__(self,
                 num_inputs: int,
                 classifier: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.CrossEntropyLoss,
                 print_every: int,
                 device: str,
                 use_cuda: bool = True,
                 use_scheduler: bool = True):

        # Initialize model
        self.classifier = nn.DataParallel(classifier)
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_inputs = num_inputs

        # Training config
        self.print_every = print_every

        # zyn
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=10,
                                                         gamma=0.2)

        # Losses
        self.losses = {'train': [], 'test': []}
        self.acces = {'train': [], 'test': []}

        self.use_cuda = use_cuda
        self.device = device
        self.use_scheduler = use_scheduler

        if self.use_cuda:
            self.classifier.to(device)

        # To store predictions and ground truth
        self.ground_truth = []
        self.predictions = []

    def _train_iteration(self, data: list, label: torch.tensor) -> float:
        """ """

        batch_size = data[0].size(0)

        if self.use_cuda:
            data = [x.to(self.device) for x in data]
            label = label.to(self.device)

        self.optimizer.zero_grad()
        pred = self.classifier(*data)
        loss = self.criterion(pred, label)
        loss.backward()
        self.optimizer.step()

        _, preds = torch.max(pred.data, 1)
        batch_corrects = torch.sum(preds == label.data)

        return loss.item() * batch_size, batch_corrects.item()

    def _test_iteration(self, data: list, label: torch.tensor) -> float:
        """ """
        with torch.no_grad():
            batch_size = data[0].size(0)

            if self.use_cuda:
                data = [x.to(self.device) for x in data]
                label = label.to(self.device)

            pred = self.classifier(*data)
            loss = self.criterion(pred, label)

            _, preds = torch.max(pred.data, 1)
            batch_corrects = torch.sum(preds == label.data)

            # Collect predictions and ground truths for the confusion matrix
            self.ground_truth.extend(label.cpu().numpy())
            self.predictions.extend(preds.cpu().numpy())

        return loss.item() * batch_size, batch_corrects.item()

    def _train_epoch(self, trainloader: torch.utils.data.DataLoader,
                     testloader: torch.utils.data.DataLoader) -> None:
        """ """

        train_loss = 0
        num_s_train = 0
        running_corrects_train = 0.0

        for i, sample in enumerate(trainloader):
            data = [x.float() for x in sample[0:-1]]
            label = sample[-1]

            num_s_train += len(label)

            _train_loss, _batch_corrects = self._train_iteration(data, label)

            train_loss += _train_loss
            running_corrects_train += _batch_corrects

        self.losses['train'].append(train_loss / num_s_train)
        self.acces['train'].append(running_corrects_train / num_s_train)

        test_loss = 0
        num_s_test = 0
        running_corrects_test = 0.0

        # Clear previous epoch's predictions and ground truths
        self.ground_truth = []
        self.predictions = []

        for i, sample in enumerate(testloader):
            data = [x.float() for x in sample[0:-1]]
            label = sample[-1]

            num_s_test += len(label)

            _test_loss, _batch_corrects = self._test_iteration(data, label)

            test_loss += _test_loss
            running_corrects_test += _batch_corrects

        self.losses['test'].append(test_loss / num_s_test)
        self.acces['test'].append(running_corrects_test / num_s_test)

    def train(self, trainloader: torch.utils.data.DataLoader,
              testloader: torch.utils.data.DataLoader, epochs: int) -> None:

        for epoch in range(1, epochs + 1):

            start = time.time()

            self._train_epoch(trainloader, testloader)

            if epoch % self.print_every == 0:

                print(
                    "Epoch {} {:.2f}s: train loss {:.6f}, train acc {:.3f}, test loss {:.6f}, test acc {:.3f}"
                    .format(epoch,
                            time.time() - start, self.losses['train'][-1],
                            self.acces['train'][-1], self.losses['test'][-1],
                            self.acces['test'][-1]))

            if self.use_scheduler:
                self.scheduler.step()
