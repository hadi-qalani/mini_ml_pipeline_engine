class Dataset:
    """A simple dataset wrapper around a list of data.

    Attributes:
        _data: The underlying list containing the dataset items.
    """

    def __init__(self, data: list) -> None:
        """Initialize the dataset.

        Args:
            data: A non-empty list containing the dataset items.

        Raises:
            TypeError: If ``data`` is not a list.
            ValueError: If ``data`` is empty.
        """
        if not isinstance(data, list):
            raise TypeError("input is not a list")

        if not data:
            raise ValueError("list should not be empty")
        self._data = data

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            The number of items in the dataset.
        """
        return len(self._data)

    def __getitem__(self, key: int) -> list:
        """Return the item at the given index.

        Args:
            key: The index of the requested item.

        Returns:
            The item stored at the given index.
        """
        return self._data[key]
