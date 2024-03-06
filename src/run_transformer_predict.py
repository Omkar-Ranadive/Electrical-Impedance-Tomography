import ast
from constants import EXP_PATH, DATA_PATH
import argparse
import torch 
import algo_nn
from torch.utils.data import DataLoader
import numpy as np 
import utils_torch
from collections import defaultdict
import pandas as pd 
import scipy.io as sio
import utils_math
import algos
import time 
from utils import save_friendly


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Directory to load the saved model from")
    parser.add_argument("--name", type=str, default='model_best.tar', help="Checkpoint of the model to load")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit training to matrices < max_rows")
    parser.add_argument("--max_cols", type=int, default=None, 
                        help="Matrices will be padded with max(max_cols, max_col_of_matrices)")
    parser.add_argument("--bs", type=int, default=32, help="Batch size to use during inference")
    parser.add_argument("--cf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    EXP_DIR = EXP_PATH / 'nn' / args.exp_name

    checkpoint = torch.load(EXP_DIR / args.name)
    print(f"Epoch: {checkpoint['epoch']}, Best: {checkpoint['is_best']}, " 
          f"Val Acc: {checkpoint['val_acc']}, Train Acc: {checkpoint['train_acc']}")
    
    dataCreator = utils_torch.DataPreparer(filename='volumes.csv', max_rows=args.max_rows, max_cols=args.max_cols)

    padded_mat, X_test, Y_test, grouped_indices, max_cols = dataCreator.get_data_without_split()
    test_dataset = utils_torch.MatrixDataset(X_test, Y_test)

    input_size = max_cols # Number of features in each row
    hidden_size = max_cols
    output_size = 1  # Binary classification (good or bad row)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = algo_nn.AttentionModel(input_size, hidden_size, output_size, max_cols).to(device)
    model = algo_nn.TransformerModel(input_size, hidden_size, output_size, num_layers=3, nhead=3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    matrix_sampler = utils_torch.MatrixInferenceSampler(grouped_indices)
    test_loader = DataLoader(test_dataset, batch_sampler=matrix_sampler)
    start = 0
    all_candidates = defaultdict(list)
    batch_size = args.bs
    endtimes = []
    for i, (batch_matid_i, batch_measurement_i, batch_row_index, batch_Y) in enumerate(test_loader):
        starttime1 = time.time()
        correct_predictions = 0
        total_samples = 0
        matrix_rows = len(batch_Y)
        batch_matrix_i = torch.tensor(padded_mat[batch_matid_i[0]]).float()
        C, M, D = dataCreator.contacts[i], dataCreator.polys[i], dataCreator.measurements[i]
        print((f"Running inference on: ID: {i}, C : {C}, M: {M}, D: {D}, arr shape: {batch_matrix_i.size()}"))
        
        while start < matrix_rows:
            end = min(start + batch_size, matrix_rows)  # Adjust end to avoid out-of-bounds indexing
            matid_subset = batch_matid_i[start: end]
            measurement_subset = batch_measurement_i[start: end]
            row_index_subset = batch_row_index[start: end]
            batch_Y_subset = batch_Y[start: end] 

            outputs = model((batch_matrix_i.to(device), matid_subset.to(device), 
                             measurement_subset.to(device), row_index_subset.to(device)))
            
            # Calculate number of correct predictions
            batch_Y_subset = batch_Y_subset.unsqueeze(1) if batch_Y_subset.dim() == 1 else batch_Y_subset
            predicted_labels = (outputs > args.cf).float() 
            correct_predictions += (predicted_labels == batch_Y_subset.to(device)).sum().item()
            total_samples += batch_Y_subset.size(0)

            # Store indices for predicted labels equal to 1
            for idx, pred_label in enumerate(predicted_labels.squeeze()):
                if pred_label == 1:
                    all_candidates[i].append(row_index_subset[idx].item())
            
            start = end 
        
        endtimes.append(time.time() - starttime1)
    
    # Use selected indices to get final output using greedy
    df = []
    columns = ['C: Contacts', 'M: polys', 'D: Measurements', 'V^(1/D)', 'Indices', 'Elasped Time']

    for i in range(len(padded_mat)): 
        starttime2 = time.time()
        C, M, D = dataCreator.contacts[i], dataCreator.polys[i], dataCreator.measurements[i]
        arr = padded_mat[i]
        candidates = all_candidates[i]
        candidates = np.unique(np.array(candidates))
        print("*"*20)
        print(f"Matrix ID: {i}, C (Contact): {C}, M (Poly): {M}, D (Measurement): {D}, arr shape: {arr.shape}")
        print(f"Number of selected candidates by Neural Network: {len(candidates)}")
        if len(candidates) >= D:
            final_volume, selected_indices = algos.greedy_approach(arr[candidates], D)
            print("Final volume: ", final_volume)
        else: 
            final_volume = 0 
            selected_indices = []
        
        endtime = time.time() - starttime2
        elapsed_time = endtimes[i] + endtime

        df.append([C, M, D, final_volume, selected_indices, elapsed_time])

    df = pd.DataFrame(df, columns=columns)
    df.to_csv(EXP_PATH / 'OptimizationMatrices' / f'nn_{args.exp_name}_cf{save_friendly(args.cf)}_bs{args.bs}.csv', 
              index=False) 

        
        

