import argparse

from utils.dataset import (run_preprocessing_BraTS2018_training,
                     run_preprocessing_BraTS2018_validationOrTesting)
from utils.paths import DataPath


def run(mode=None):
    print('start')
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="train",
                    help="train for training set, val for validation set, "
                         "and test for testing set", type=str)
    args = parser.parse_args()
    print(args.mode)

    mode = args.mode if mode is None else mode
    if mode == "train":
        run_preprocessing_BraTS2018_training(
                    DataPath.data_folder, DataPath.training_data, 
                    DataPath.preprocessed_training_data_folder)
    elif mode == "val":
       run_preprocessing_BraTS2018_validationOrTesting(
                    DataPath.data_folder, DataPath.validation_data, 
                    DataPath.preprocessed_validation_data_folder)
    elif mode == "test":
    	run_preprocessing_BraTS2018_validationOrTesting(
                    DataPath.data_folder, DataPath.testing_data, 
                    DataPath.preprocessed_testing_data_folder)
    else:
        raise ValueError("Unknown value for --mode. Use \"train\", \"test\" or \"val\"")

if __name__ == "__main__":
    run()
