from src.dataloader import DataLoader
from src.dataset import Dataset
from src.preprocessing import min_max_normalize
from src.split import train_test_split

data = [10.0, 20.0, 30.0, 40.0, 50.0]

norm_data = min_max_normalize(data)

train, test = train_test_split(norm_data)

train_dataset = Dataset(train)
test_dataset = Dataset(test)


train_dataset = Dataset(train)

train_loader = DataLoader(train_dataset, batch_size=2)

for batch in train_loader:
    print(batch)


test_dataset = Dataset(test)

test_loader = DataLoader(test_dataset, batch_size=2)

for batch in test_loader:
    print(batch)
