import sys
from Net import LunaModel
import torch as t
import torch.nn as nn
import numpy as np
import logging
import torch.optim as optim
from DataLoader import LunaDataset
from torch.utils.data import DataLoader
import argparse

logging.basicConfig(filename='luna.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    filemode='a',  # a means write after the former text, w means overwrite on the former text
                    level=logging.DEBUG,  # logging all the information higher than debug (info, warning etc.)
                    datefmt='%Y/%m/%d %H:%M:%S')  # the format of time stamp for logging


class PreLunaTrainingApp:
    def __init__(self, sys_argv=None):
        # for more information about argparse
        # see sys_argv_test.py
        if sys_argv is None:
            # sys.argv[1:] is all the variables we input through command line
            sys_argv = sys.argv[1:]  # if we want to use command line to input parameters

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8, type=int)
        parser.add_argument('--batch-size',
                            help='Number of samples of each batch',
                            default=1024, type=int)

        # we use argument variables from command line
        self.args = parser.parse_args(sys_argv)

    def main(self):
        # start
        logging.info('Starting {}, {}'.format(type(self).__name__, self.args))

        # initiate data loader
        self.prepare = DataLoader(LunaDataset(stride=1, isVal_bool=True),
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers)

        for _ in enumerate(self.prepare):
            pass


if __name__ == '__main__':
    PreLunaTrainingApp().main()