import numpy as np
import torch


class MetricLogger:
    """Helper class for logging classification metrics for multiclass-classification problems.

    The class provides the ability to log accumulatively, i.e. each call to log updates the internal data structure with the new results. The reset method may be used to reset the logger, typically after an epoch.

    The metric properties are calculated and hence always up to date with the logged results.
    """

    def __init__(self, classes, one_hot=True):
        """Create new logger.
        Argument in the constructor is needed since Scikit-learn provides numerical predictions while PyTorch provides one-hot encoded predictions. 

        Args:
            classes (int, optional): The number of problem classes.
            one_hot (bool, optional): Whether the predictions and labels are one-hot encoded or not. Use the default if in doubt. Defaults to True.
        """
        self.mat = np.zeros((classes, classes))
        self.one_hot = one_hot

    def log(self, predicted, target):
        """Log a new prediction and label pair. Adds a log entry to the confusion matrix based on the predicted and target values.

        Remember that the parameters have to be one-hot encoded if this option was chosen during initialization.

        Args:
            predicted (array/tensor): The output prediction from the model.
            target (array/tensor): The ground-truth label.
        """
        if type(predicted) is torch.Tensor:
            predicted = predicted.detach().numpy()
        if type(target) is torch.Tensor:
            target = target.detach().numpy()

        if self.one_hot:
            predicted = np.argmax(predicted, axis=1)

        for pi, ti in zip(predicted, target):
            self.mat[pi, ti] += 1

    def reset(self):
        """Reset the logger - sets the entries of the matrix to zero. Useful for reusing the same object between epochs.
        """
        self.mat = np.zeros(self.mat.shape)

    @property
    def correct(self):
        """The number of correct predictions for each class.

        This is just the confusion matrix diagonal.

        Returns:
            np.ndarray: Vector of the number of correct predictions for each class.
        """
        return np.diag(self.mat)

    @property
    def predicted_positive(self):
        """The number of positive predictions for each class.

        This is the sum of all the columns in the confusion matrix.

        Returns:
            np.ndarray: Vector of positive predictions for each class.
        """
        return self.mat.sum(axis=1)

    @property
    def actual_positive(self):
        """The number of positive labels for each class.

        This is the true number of positives for each class. It is the sum of all the rows in the confusion matrix.

        Returns:
            np.ndarray: Vector of positive labels for each class.
        """
        return self.mat.sum(axis=0)

    @property
    def accuracy(self):
        """Problem accuracy.

        This is the total number of correct predictions divided by the number 
        of samples in the dataset.

        Returns:
            float: Accuracy value.
        """
        return self.correct.sum() / self.mat.sum()

    @property
    def precision(self):
        """Class-wise precision.

        This is the number of correct predictions divided by the number of positive predictions for each class seperately.

        Returns:
            np.ndarray: Vector of class-wise precision values.
        """
        return self.correct / self.predicted_positive

    @property
    def recall(self):
        """Class-wise recall.

        This is the number of correct predictions divided by the number of positive labels for each class seperately.

        Returns:
            np.ndarray: Vector of class-wise recall values.
        """
        return self.correct / self.actual_positive

