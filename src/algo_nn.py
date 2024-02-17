import torch
import torch.nn as nn
from constants import data_list
import pandas as pd 
from constants import EXP_PATH, DATA_PATH
import scipy.io as sio
import os 
import numpy as np
import utils_math
import ast
from torch.utils.data import Dataset, DataLoader


class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2 + 1, output_size)  

    def forward(self, input_data):
        matrix_i, measurement_i, row_index = input_data

        # Extract the row at the specified index from the matrix
        batch_size = matrix_i.size(0)
        
        # Select the row at row_index for all samples in the batch from matrix i ([batch_size, num_rows, num_features])
        # Shape: [batch_size, hidden_size]
        embedded_row = self.embedding(matrix_i[torch.arange(batch_size, dtype=torch.int32), row_index, :]) 

        # Process the matrix through the RNN
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(matrix_i) # Shape: [batch_size, seq_len, hidden_size]

        # Calculate attention weights
        # For torch.bmm shapes need to be (b, n, m) x (b, m, p) 
        # That is in bmm, (last dim of first mat should match with second last dim of second mat)
        attention_weights = torch.bmm(rnn_output, embedded_row.unsqueeze(2))  # Add a dim in embedded row to make it 3d
        attention_weights = nn.functional.softmax(attention_weights, dim=1) # Shape: [batch_size, seq_len, 1]

        # Apply attention weights to RNN output
        context_vector = torch.bmm(attention_weights.transpose(1, 2), rnn_output) # [batch_size, 1, hidden_size]

        # Concatenate attended representation with embedded row and additional features
        # Context Vector and Embedded Row will be = [batch_size, hidden_size], measurement will be [batch_size, 1]
        combined_vector = torch.cat((context_vector.squeeze(1), embedded_row.squeeze(1), 
                                     measurement_i.unsqueeze(1)), dim=1)

        # Final classification
        output = torch.sigmoid(self.fc(combined_vector))
        return output


class MatrixDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        matrix_i, measurement_i, row_index = self.data[idx]
        label = self.labels[idx]
        matrix_tensor = torch.from_numpy(matrix_i).float()
        measurement_tensor = torch.tensor(measurement_i).float()
        row_index_tensor = torch.tensor(row_index).int()
        label_tensor = torch.tensor(label).float()
        
        return matrix_tensor, measurement_tensor, row_index_tensor, label_tensor

def prepare_training_data(max_rows=None):
    df = pd.read_csv(EXP_PATH / 'OptimizationMatrices' / 'volumes.csv')    

    matrices = []
    good_indices = []
    measurements = []
    for index, row in df.iterrows():         
        C, M, D = row['C: Contacts'], row['M: polys'], row['D: Measurements']
        EXP_DIR = EXP_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys_D{D}'
        os.makedirs(EXP_DIR, exist_ok=True)
        #print(f"Running for: {C}contacts_{M}polys_D{D}")
        data = sio.loadmat(DATA_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys.mat')
        arr = data['JGQ']
        if not max_rows or (arr.shape[0] < max_rows): 
            arr = utils_math.scale_data(arr, technique='simple')
            indices = np.array(ast.literal_eval(row['Indices #2']), dtype=int)  # Convert string to numpy array 
            good_indices.append(indices)        
            matrices.append(arr)
            measurements.append(D)
    
    # Find the largest number of columns
    max_m = max(matrix.shape[1] for matrix in matrices)
    
    # Pad each matrix along the columns and add it to the list
    padded_matrices = []
    for matrix in matrices:
        n, m = matrix.shape
        pad_width = ((0, 0), (0, max_m - m))  # Pad only along the columns
        padded_matrix = np.pad(matrix, pad_width, mode='constant')
        padded_matrices.append(padded_matrix)

    # Create training samples for each matrix
    X = []
    Y = []
    for i, matrix_i in enumerate(padded_matrices):
        indices_i = good_indices[i]
        measurement_i = measurements[i]

        # Create training samples for each row index
        for row_index in range(matrix_i.shape[0]):
            # Check if the row index is present in indices_i
            label = 1 if row_index in indices_i else 0
            sample = (matrix_i, measurement_i, row_index)
            X.append(sample)
            Y.append(label)
    
    return X, Y, max_m


