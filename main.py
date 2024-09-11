# Description:
# This is the main file to run the project.
# It will call the necessary functions to prepare the data and train the models.
# It will also call the test functions to evaluate the models.

from src.utils.data_preparation import prepare_data
from test import start_test
from train import start_train


def start(train):
    prepare_data()

    if train:
        start_train()
    else:
        start_test('models/best')


if __name__ == '__main__':
    start(train=False)