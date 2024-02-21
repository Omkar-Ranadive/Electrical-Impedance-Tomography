from constants import EXP_PATH, DATA_PATH
import argparse
import torch 
import algo_nn
from torch.utils.data import DataLoader
import numpy as np 
import utils_torch
from collections import defaultdict


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Directory to load the saved model from")
    parser.add_argument("--name", type=str, default='model_best.tar', help="Checkpoint of the model to load")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit training to matrices < max_rows")
    parser.add_argument("--max_cols", type=int, default=None, 
                        help="Matrices will be padded with max(max_cols, max_col_of_matrices)")
    parser.add_argument("--bs", type=int, default=32, help="Batch size to use during inference")
    args = parser.parse_args()
    EXP_DIR = EXP_PATH / 'nn' / args.exp_name

    checkpoint = torch.load(EXP_DIR / args.name)
    print(f"Epoch: {checkpoint['epoch']}, Best: {checkpoint['is_best']}, " 
          f"Val Acc: {checkpoint['val_acc']}, Train Acc: {checkpoint['train_acc']}")
    
    dataCreator = utils_torch.DataPreparer(filename='volumes.csv', max_rows=args.max_rows, max_cols=args.max_cols)

    X_test, Y_test, grouped_indices, max_cols = dataCreator.get_data_without_split()
    test_dataset = utils_torch.MatrixDataset(X_test, Y_test)

    input_size = max_cols # Number of features in each row
    hidden_size = max_cols  # Hidden size of LSTM and embedding
    output_size = 1  # Binary classification (good or bad row)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = algo_nn.AttentionModel(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    matrix_sampler = utils_torch.MatrixInferenceSampler(grouped_indices)
    test_loader = DataLoader(test_dataset, batch_sampler=matrix_sampler)
    start = 0
    selected_indices = defaultdict(list)
    batch_size = args.bs

    for i, (batch_matrix_i, batch_measurement_i, batch_row_index, batch_Y) in enumerate(test_loader):
        correct_predictions = 0
        total_samples = 0
        matrix_rows = len(batch_Y)
        while start < matrix_rows:
            end = min(start + batch_size, matrix_rows)  # Adjust end to avoid out-of-bounds indexing
            matrix_subset = batch_matrix_i[start: end]
            measurement_subset = batch_measurement_i[start: end]
            row_index_subset = batch_row_index[start: end]
            batch_Y_subset = batch_Y[start: end] 
            print(matrix_subset.size(), measurement_subset.size(), batch_Y_subset.size(), row_index_subset.size())
            outputs = model((matrix_subset.to(device), measurement_subset.to(device), row_index_subset.to(device)))
            # Calculate number of correct predictions
            batch_Y_subset = batch_Y_subset.unsqueeze(1) if batch_Y_subset.dim() == 1 else batch_Y_subset
            predicted_labels = (outputs > 0.5).float() 
            correct_predictions += (predicted_labels == batch_Y_subset.to(device)).sum().item()
            total_samples += batch_Y_subset.size(0)

            # Store indices for predicted labels equal to 1
            for idx, pred_label in enumerate(predicted_labels.squeeze()):
                if pred_label == 1:
                    selected_indices[i].append(row_index_subset[idx].item())
            
            start = end 


    print(selected_indices)


