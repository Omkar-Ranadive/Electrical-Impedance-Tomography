import torch 
import shutil
import matplotlib.pyplot as plt 
import seaborn as sns 
import utils_math
import ast
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from constants import data_list
import pandas as pd 
from constants import EXP_PATH, DATA_PATH
import scipy.io as sio
import os 
import numpy as np
from collections import defaultdict


sns.set_theme(style='whitegrid')


def save_checkpoint(state, is_best, EXP_DIR, filename):
    torch.save(state, str(EXP_DIR / filename))
    if is_best:
        shutil.copyfile(str(EXP_DIR / filename), str(EXP_DIR / 'model_best.tar'))


class DataPreparer(): 
    def __init__(self, filename, max_rows=None, max_cols=None) -> None:
        self.max_rows = max_rows
        self.df = pd.read_csv(EXP_PATH / 'OptimizationMatrices' / filename)    
        self.matrices = []
        self.good_indices = []
        self.measurements = []
        self.contacts = []
        self.polys = []
        self.max_cols = max_cols
        self.padded_matrices, self.max_m = self._load_matrices()
    

    def _load_matrices(self):
        for index, row in self.df.iterrows():         
            C, M, D = row['C: Contacts'], row['M: polys'], row['D: Measurements']
            EXP_DIR = EXP_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys_D{D}'
            os.makedirs(EXP_DIR, exist_ok=True)
            #print(f"Running for: {C}contacts_{M}polys_D{D}")
            data = sio.loadmat(DATA_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys.mat')
            arr = data['JGQ']
            if not self.max_rows or (arr.shape[0] < self.max_rows): 
                arr = utils_math.scale_data(arr, technique='simple')
                indices = np.array(ast.literal_eval(row['Indices #2']), dtype=int)  # Convert string to numpy array 
                self.good_indices.append(indices)        
                self.matrices.append(arr)
                self.measurements.append(D)
                self.contacts.append(C)
                self.polys.append(M)

        # Find the largest number of columns
        max_m = max(matrix.shape[1] for matrix in self.matrices)
        if self.max_cols: 
            max_m = max(max_m, self.max_cols)
        
        # Pad each matrix along the columns and add it to the list
        padded_matrices = []
        for matrix in self.matrices:
            n, m = matrix.shape
            pad_width = ((0, 0), (0, max_m - m))  # Pad only along the columns
            padded_matrix = np.pad(matrix, pad_width, mode='constant')
            padded_matrices.append(padded_matrix)
        
        return padded_matrices, max_m


    def get_data_with_split(self, test_fraction=0.1):
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        grouped_indices_train = defaultdict(list)
        grouped_indices_test = defaultdict(list)

        total_matrices = len(self.padded_matrices) 
        num_test_samples = int(total_matrices * test_fraction)
        test_matrix_indices = set(np.random.choice(np.arange(0, total_matrices), size=num_test_samples))

        counter_train, counter_test = 0, 0 

        for i, matrix_i in enumerate(self.padded_matrices):
            indices_i = self.good_indices[i]
            measurement_i = self.measurements[i]
            num_rows = matrix_i.shape[0]

            test_flag = i in test_matrix_indices
            
            # Create training samples for each row index
            for row_index in range(num_rows):
                # Check if the row index is present in indices_i
                label = 1 if row_index in indices_i else 0
                sample = [i, measurement_i, row_index]
                if test_flag: 
                    X_test.append(sample)
                    Y_test.append(label)
                    grouped_indices_test[i].append(counter_test)
                    counter_test += 1
                else: 
                    X_train.append(sample)
                    Y_train.append(label)
                    grouped_indices_train[i].append(counter_train)
                    counter_train += 1
        
        return (self.padded_matrices, X_train, Y_train, grouped_indices_train, 
                X_test, Y_test, grouped_indices_test, self.max_m)


    def get_data_without_split(self): 
        # Create training samples for each matrix
        X = []
        Y = []
        grouped_indices = defaultdict(list)
        counter = 0 
        for i, matrix_i in enumerate(self.padded_matrices):
            indices_i = self.good_indices[i]
            measurement_i = self.measurements[i]
            num_rows = matrix_i.shape[0]
            # Create training samples for each row index
            for row_index in range(num_rows):
                # Check if the row index is present in indices_i
                label = 1 if row_index in indices_i else 0
                sample = [i, measurement_i, row_index]
                X.append(sample)
                Y.append(label)
                grouped_indices[i].append(counter)
                counter += 1

        return self.padded_matrices, X, Y, grouped_indices, self.max_m


class MatrixDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        matrix_i, measurement_i, row_index = self.data[idx]
        label = self.labels[idx]
        matrix_tensor = torch.tensor(matrix_i).int()
        measurement_tensor = torch.tensor(measurement_i).float()
        row_index_tensor = torch.tensor(row_index).int()
        label_tensor = torch.tensor(label).float()
        
        return matrix_tensor, measurement_tensor, row_index_tensor, label_tensor


class MatrixBatchSampler(Sampler):
    def __init__(self, grouped_indices, batch_size):
        self.grouped_indices = grouped_indices
        self.batch_size = batch_size

    def __iter__(self):
        for indices in self.grouped_indices.values():
            num_samples = len(indices)
            for i in range(0, num_samples, self.batch_size):
                yield indices[i:i+self.batch_size]

    def __len__(self):
        total_samples = sum(len(indices) for indices in self.grouped_indices.values())
        return total_samples // self.batch_size


class MatrixInferenceSampler(Sampler):
    def __init__(self, grouped_indices):
        self.grouped_indices = grouped_indices

    def __iter__(self):
        for indices in self.grouped_indices.values():
                yield indices

    def __len__(self):
        total_samples = sum(len(indices) for indices in self.grouped_indices.values())
        return total_samples