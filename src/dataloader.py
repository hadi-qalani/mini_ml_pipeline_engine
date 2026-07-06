from .dataset import Dataset


class Counter1:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)
        ...

    def __next__(self):
        return self


# -----------------------------------------------------------------------


class Counter2:
    def __init__(self, limit):
        self.current = 0
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration

        value = self.current
        self.current += 1
        return value


class DataLoader:
    """Iterate over a Dataset and return samples in batches.

    This class implements the iterator protocol and yields batches of
    samples from a Dataset instance.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
    ):
        """Initialize a DataLoader instance.

        Args:
            dataset: The Dataset to iterate over.
            batch_size: Number of samples to return in each batch.

        Raises:
            TypeError: If ``dataset`` is not an instance of Dataset.
            ValueError: If the dataset is empty.
            ValueError: If ``batch_size`` is less than or equal to zero.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("input must be a Dataset")
        if not dataset:
            raise ValueError("input dataset is empty")
        if batch_size <= 0:
            raise ValueError("batch size should be greater than 0")

        self._dataset = dataset
        self._start = 0
        self._end = 0
        self._batch_size = batch_size

    def __iter__(self):
        """Return the iterator.

        Resets the internal cursor to the beginning of the dataset.

        Returns:
            DataLoader: The iterator itself.
        """
        self._start = 0
        return self

    def __next__(self):
        """Return the next batch of samples.

        Returns:
            list: A list containing up to ``batch_size`` samples.

        Raises:
            StopIteration: If there are no more samples to iterate over.
        """
        batch = []

        if self._start >= len(self._dataset):
            raise StopIteration

        batch.clear()

        self._end = self._start + self._batch_size
        if self._end > len(self._dataset):
            self._end = len(self._dataset)

        for i in range(self._start, self._end):
            batch.append(self._dataset[i])

        self._start += self._batch_size

        return batch
