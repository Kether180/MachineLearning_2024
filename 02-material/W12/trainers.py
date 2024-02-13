import os
import pickle
from datetime import datetime
from abc import ABC, abstractmethod


import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from fashionmnist_utils import mnist_reader
from metrics import MetricLogger


class Trainer(ABC):
    def __init__(self, model):
        self.model = model
        self.name = (
            f'{type(model).__name__}-{datetime.now().strftime("%m-%d--%H-%M-%S")}'
        )

    @abstractmethod
    def train(self, *args):
        ...

    @abstractmethod
    def predict(self, input):
        ...

    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def save(self):
        ...

    @staticmethod
    @abstractmethod
    def load(path: str):
        ...



def get_data(transform, train=True):
    return FashionMNIST(os.getcwd(), train=train, transform=transform, download=True)


class PyTorchTrainer(Trainer):
    def __init__(self, nn_module, transform, optimizer, batch_size):
        super().__init__(nn_module)

        self.train_data, self.val_data, self.test_data = None, None, None

        self.transform = transform
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.init_data()

        self.logger = SummaryWriter()

    def init_data(self):
        data = get_data(self.transform, True)
        test_data = get_data(self.transform, False)
        val_len = int(len(data) * 0.2)

        torch.manual_seed(42)
        train_data, val_data = random_split(data, [len(data) - val_len, val_len])

        self.train_data = DataLoader(train_data, self.batch_size)
        self.val_data = DataLoader(val_data, self.batch_size)
        self.test_data = DataLoader(test_data, self.batch_size)

    def train(self, epochs):
        """Train the model using SGD. 

        Args:
            epochs (int): The total number of training epochs.
        """
        self.logger.add_graph(self.model, next(iter(self.train_data))[0])

        update_interval = len(self.train_data) // 5

        train_logger = MetricLogger(classes=10)
        val_logger = MetricLogger(classes=10)
        
        
        # Early stopping  
        # Uncomment lines below this if you want early stopping \
        #last_loss = 1000
        #patience = 5
        #triggertimes = 0
        # / Early stopping
        for e in range(epochs):
            print(f"[Epoch {e + 1}]")
            running_loss = 0.0

            self.model.train()
            for i, (x, y) in enumerate(tqdm(self.train_data, leave=None)):
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                train_logger.log(out, y)

                if (i + 1) % update_interval == 0:
                    self.logger.add_scalar(
                        "loss",
                        running_loss / update_interval,
                        global_step=i + e * len(self.train_data),
                    )
                    self.logger.add_scalar(
                        "train_accuracy",
                        train_logger.accuracy,
                        i + e * len(self.train_data),
                    )
                    train_logger.reset()
                    running_loss = 0.0

            val_logger.reset()
            self.model.eval()
            running_val_loss=0
            for x, y in tqdm(self.val_data, leave=None):
                out = self.model(x)
                loss_val = F.cross_entropy(out, y)
                val_logger.log(out, y)
                running_val_loss += loss_val.item()

            self.logger.add_scalar("accuracy", val_logger.accuracy, e)


            # Early stopping
            # Uncomment lines below this if you want early stopping \
            # current_acc = running_val_loss

            # if current_loss > last_loss:
            #     trigger_times += 1
            #     print('Trigger Times:', trigger_times)

            #     if trigger_times >= patience:
            #         print('Early stopping!\nStart to test process.')
            #         break

            # else:
            #     print('trigger times: 0')
            #     trigger_times = 0

            # last_loss = current_loss
            ## / Early stopping

            print(
                f"[Validation] acc: {val_logger.accuracy:.4f}, precision: {val_logger.precision.mean():.4f}, recall: {val_logger.recall.mean():.4f}"
            )
            
            
    def train_es(self, epochs, patience):
        """Train the model using SGD. 

        Args:
            epochs (int): The total number of training epochs.
        """
        self.logger.add_graph(self.model, next(iter(self.train_data))[0])

        update_interval = len(self.train_data) // 5

        train_logger = MetricLogger(classes=10)
        val_logger = MetricLogger(classes=10)
        
        
        # Early stopping  
        last_loss = 1000
        trigger_times = 0
        # / Early stopping
        for e in range(epochs):
            print(f"[Epoch {e + 1}]")
            running_loss = 0.0

            self.model.train()
            for i, (x, y) in enumerate(tqdm(self.train_data, leave=None)):
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                train_logger.log(out, y)

                if (i + 1) % update_interval == 0:
                    self.logger.add_scalar(
                        "loss",
                        running_loss / update_interval,
                        global_step=i + e * len(self.train_data),
                    )
                    self.logger.add_scalar(
                        "train_accuracy",
                        train_logger.accuracy,
                        i + e * len(self.train_data),
                    )
                    train_logger.reset()
                    running_loss = 0.0

            val_logger.reset()
            self.model.eval()
            running_val_loss=0
            for x, y in tqdm(self.val_data, leave=None):
                out = self.model(x)
                loss_val = F.cross_entropy(out, y)
                val_logger.log(out, y)
                running_val_loss += loss_val.item()

            self.logger.add_scalar("accuracy", val_logger.accuracy, e)


            # Early stopping

            current_loss = running_val_loss

            if current_loss > last_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                break

            else:
                print('trigger times: 0')
                trigger_times = 0

            last_loss = current_loss
            ## / Early stopping

            print(
                f"[Validation] acc: {val_logger.accuracy:.4f}, precision: {val_logger.precision.mean():.4f}, recall: {val_logger.recall.mean():.4f}"
            )

    def predict(self, input):
        """Generate predictions for the specified input.

        Args:
            input (tensor): A BxN (for MLP) or Bx1xHxW (for CNN) shaped tensor.

        Returns:
            _type_: _description_
        """
        return self.model(input)

    def evaluate(self):
        """Test the model on the test dataset and collect metric information and predictions.

        Returns:
            (MetricLogger, tensor): The logging results as well as the predictions as class labels.
        """
        test_logger = MetricLogger(classes=10)
        predictions = []
        self.model.eval()
        for x, y in tqdm(self.test_data, leave=None):
            out = self.model(x)
            predictions.append(torch.argmax(out, dim=1))
            test_logger.log(out, y)
        return test_logger, torch.cat(predictions)

    def save(self):
        """Save the trained model to disk.
        """
        self.train_data, self.val_data, self.test_data = None, None, None
        self.logger = None

        file_name = os.path.join("models", self.name)
        with open(file_name + ".pkl", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str):
        """Instantiate a model from a saved state.
        """
        with open(path, "rb") as file:
            new = pickle.load(file)
            new.init_data()
            return new

