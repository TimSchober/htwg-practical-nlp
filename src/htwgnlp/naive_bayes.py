"""Naive Bayes classifier for NLP.

This module contains the NaiveBayes class for NLP tasks.

Implementing this module is the 3rd assignment of the course. You can find your tasks by searching for `TODO ASSIGNMENT-3` comments.

Hints:
- Find more information about the Python property decorator [here](https://www.programiz.com/python-programming/property)
- To build the word frequencies, you can use the [Counter](https://docs.python.org/3/library/collections.html#collections.Counter) class from Python's collections module
- you may also find the Python [zip](https://docs.python.org/3/library/functions.html#zip) function useful.
- for prediction, you may find the [intersection](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.intersection.html) method of the pandas Index class useful.

"""

from collections import Counter

import numpy as np
import pandas as pd


class NaiveBayes:
    """Naive Bayes classifier for NLP tasks.

    This class implements a Naive Bayes classifier for NLP tasks.
    It can be used for binary classification tasks.

    Attributes:
        word_probabilities (pd.DataFrame): the word probabilities per class, None before training
        df_freqs (pd.DataFrame): the word frequencies per class, None before training. The index of the DataFrame is the vocabulary.
        log_ratios (pd.Series): the log ratios of the word probabilities, None before training. The index of the Series is the vocabulary.
        logprior (float): the logprior of the model, 0 before training. The index of the Series is the vocabulary.
        alpha (float): the smoothing parameter of the model
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initializes the NaiveBayes class.

        The init method accepts one hyperparameter as an optional argument, the smoothing parameter alpha.

        Args:
            alpha (float, optional): the smoothing parameter. Defaults to 1.0.
        """
        self.word_probabilities: pd.DataFrame | None = None
        self.df_freqs: pd.DataFrame | None = None
        self.log_ratios: pd.Series | None = None
        self._logprior = 0
        self.alpha = alpha


    @property
    def logprior(self) -> float:
        """Returns the logprior.

        Returns:
            float: the logprior
        """
        return self._logprior

    @logprior.setter
    def logprior(self, y: np.ndarray) -> None:
        """Sets the logprior.

        Note that `y` must contain both classes.

        Args:
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        assert len(y[y == 1]) > 0 and len(y[y == 0]) > 0, "y must contain both classes"
        self._logprior = np.log(len(y[y == 1])) - np.log(len(y[y == 0]))

    def _get_word_frequencies(self, x: list[list[str]], y: np.ndarray) -> None:
        """Computes the word frequencies per class.

        For a given list of tokenized text and a numpy array of class labels, the method computes the word frequencies for each class and stores them as a pandas DataFrame in the `df_freqs` attribute.

        In pandas, if a word does not occur in a class, the frequency should be set to 0, and not to NaN. Also make sure that the frequencies are of type int.

        Note that the this implementation of Naive Bayes is designed for binary classification.

        Args:
            x (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples.
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples.
        """
        freqs: dict[int, Counter] = {
            label: Counter() for label in np.unique(y).astype(int)
        }
        y_transformed = y.squeeze().astype(int).tolist()
        for x_i, y_i in zip(x, y_transformed):
            freqs[y_i].update(x_i)
        self.df_freqs = pd.DataFrame(freqs).fillna(0).astype("int")

    def _get_word_probabilities(self) -> None:
        """Computes the conditional probabilities of a word given a class using Laplacian Smoothing.

        Based on the word frequencies, the method computes the conditional probabilities for a word given its class and stores them in the `word_probabilities` attribute.
        """
        if self.df_freqs is not None:
            self.word_probabilities = (self.df_freqs + self.alpha) / (self.df_freqs.sum() + len(self.df_freqs))
        else:
            raise ValueError("df_freqs is None.")

    def _get_log_ratios(self) -> None:
        """Computes the log ratio of the conditional probabilities.

        Based on the word probabilities, the method computes the log ratios and stores them in the `log_ratios` attribute.
        """
        if self.word_probabilities is not None:
            self.log_ratios = np.log(self.word_probabilities[1] / self.word_probabilities[0])
        else:
            raise ValueError("word_probabilities is None.")

    def fit(self, x: list[list[str]], y: np.ndarray) -> None:
        """Fits a Naive Bayes model for the given text samples and labels.

        Before training naive bayes, a couple of assertions are performed to check the validity of the input data:
            - The number of text samples and labels must be equal.
            - y must be a 2-dimensional array.
            - y must be a column vector.

        if all assertions pass, the method calls the Naive Bayes training method is executed.

        Args:
            x (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        assert len(x) == y.shape[0], "The number of text samples and labels must be equal."
        assert y.ndim == 2, "y must be 2-dimensional."
        assert y.shape[1] == 1, "y must be a col vector."
        self._train_naive_bayes(x, y)

    def _train_naive_bayes(self, x: list[list[str]], y: np.ndarray) -> None:
        """Trains a Naive Bayes model for the given text samples and labels.

        Training is done in four steps:
            - Compute the log prior ratio
            - Compute the word frequencies
            - Compute the word probabilities of a word given a class using Laplacian Smoothing
            - Compute the log ratios

        Args:
            x (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        self.logprior = y
        self._get_word_frequencies(x, y)
        self._get_word_probabilities()
        self._get_log_ratios()

    def predict(self, x: list[list[str]]) -> np.ndarray:
        """Predicts the class labels for the given text samples.

        The class labels are returned as a column vector, where each entry represents the class label of the corresponding sample.

        Args:
            x (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples

        Returns:
            np.ndarray: a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        return np.where(self.predict_prob(x) > 0, 1, 0)

    def predict_prob(self, x: list[list[str]]) -> np.ndarray:
        """Calculates the log likelihoods for the given text samples.

        The class probabilities are returned as a column vector, where each entry represents the probability of the corresponding sample.

        Args:
            x (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples

        Returns:
            np.ndarray: a numpy array of class probabilities of shape (m, 1), where m is the number of samples
        """
        return np.array([self.predict_single(x_i) for x_i in x]).reshape(-1, 1)

    def predict_single(self, x: list[str]) -> float:
        """Calculates the log likelihood for a single text sample.

        Words that are not in the vocabulary are ignored.

        Args:
            x (list[str]): a tokenized text sample

        Returns:
            float: the log likelihood of the text sample
        """
        if self.log_ratios is None or self.word_probabilities is None:
            raise ValueError("Model must be trained before prediction.")
        return self.logprior + self.log_ratios.loc[self.word_probabilities.index.intersection(x)].sum()
