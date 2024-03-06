from constants import EXP_PATH, DATA_PATH
import algo_nn
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os 
import logging 
import utils_torch
from torch.utils.tensorboard import SummaryWriter
import time 
import numpy as np 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--bs", type=int, default=32, help="Batch size of training samples")
    # parser.add_argument("--features", type=int, required=True, help="Number of hidden features which the model has")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit training to matrices < max_rows")
    parser.add_argument("--save_it", type=int, default=1, 
                        help="""
                        Models get saved every save_it iterations depending on this param. 
                        If -1, only final model gets saved.
                        """)
    parser.add_argument("--print_it", type=int, default=100, 
                        help="Print progress to terminal every print_it iterations")
    parser.add_argument("--max_cols", default=None, type=int, 
                        help="Matrices will be padded with max(max_cols, max_col_of_matrices)")

    args = parser.parse_args()
    EXP_DIR = EXP_PATH / 'nn' / args.exp_name
    os.makedirs(EXP_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=(EXP_DIR / 'info.log'), format='%(message)s')
    logger = logging.getLogger() 

    writer = SummaryWriter(log_dir=EXP_DIR)

    dataCreator = utils_torch.DataPreparer(filename='volumes.csv', max_rows=args.max_rows, max_cols=args.max_cols)
    (padded_mat, X_train, Y_train, grouped_indices_train, 
    X_val, Y_val, grouped_indices_val, max_cols) = dataCreator.get_data_with_split()
       
    # Convert training and validation data to custom dataset
    train_dataset = utils_torch.MatrixDataset(X_train, Y_train)
    train_sampler = utils_torch.MatrixBatchSampler(grouped_indices_train, args.bs)

    val_dataset = utils_torch.MatrixDataset(X_val, Y_val)
    val_sampler = utils_torch.MatrixBatchSampler(grouped_indices_val, args.bs)

    # Create DataLoader for training and validation data
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    logger.info("*"*20) 
    logger.info(f"Number of training samples: {len(X_train)}")
    logger.info(f"Number of validation samples: {len(X_val)}")
    logger.info(f"Total number of training batches: {len(train_loader)}")
    logger.info(f"Total number of validation batches: {len(val_loader)}")

    # Initialize the model
    input_size = max_cols # Number of features in each row
    hidden_size = max_cols # Hidden size for transformer (= max_cols)
    output_size = 1  # Binary classification (good or bad row)
    num_epochs = args.epochs
    best_acc = 0

    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Padded size of cols: {max_cols}")
    logger.info(f"Hidden Size (lstm): {hidden_size}")
    logger.info(f"Max row size of matrix: {args.max_rows}")
    logger.info(f"Batch size: {args.bs}")

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = algo_nn.TransformerModel(input_size, hidden_size, output_size, num_layers=3, nhead=3).to(device)
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Calculate weights for classes. Class 1 (selected index) is more important and is sparser in the dataset 
    num_positive_samples = (np.array(Y_train) == 1).sum()
    num_negative_samples = (np.array(Y_train) == 0).sum()
    total_samples = len(Y_train)
    # weight_positive = total_samples / (2 * num_positive_samples)
    # weight_negative = total_samples / (2 * num_negative_samples)
    pos_weight = num_negative_samples / num_positive_samples

    logger.info(f"Number of positive samples in training: {num_positive_samples}")
    logger.info(f"Number of negative samples in training: {num_negative_samples}")
    # logger.info(f"Weights: Positive class: {weight_positive}, Negative Class: {weight_negative}")
    logger.info(f"Positive weights: {pos_weight}")
    # Define class weights
    class_weights = torch.tensor([pos_weight], dtype=torch.float).to(device)
    # Define BCE loss with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    train_accs = []
    val_accs = []
    losses = []
    start_time = time.time()
    
    # Training loop with DataLoader
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0 
        total_samples = 0

        for i, (batch_matid_i, batch_measurement_i, batch_row_index, batch_Y) in enumerate(train_loader):        
            # Zero the gradients
            optimizer.zero_grad()
            
            matrix_i = torch.tensor(padded_mat[batch_matid_i[0]]).float()
            # Forward pass
            outputs = model((matrix_i.to(device), batch_matid_i.to(device), 
                             batch_measurement_i.to(device), batch_row_index.to(device)))

            # Compute loss
            batch_Y = batch_Y.unsqueeze(1) if batch_Y.dim() == 1 else batch_Y
            batch_loss = criterion(outputs, batch_Y.to(device))
            
            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()
            
            # Accumulate total loss
            total_loss += batch_loss.item()

            # Calculate number of correct predictions
            predicted_labels = (outputs > 0.5).float() 
            correct_predictions += (predicted_labels == batch_Y.to(device)).sum().item()
            total_samples += batch_Y.size(0)

            if i % args.print_it == 0: 
                print((f'Epoch [{epoch+1}/{num_epochs}], Batch: {i}, Avg batch loss: {(total_loss/(i+1)):.4f}, '
                      f'Avg Train Accuracy: {correct_predictions/total_samples}'))
                
        # Calculate average loss and train accuracy 
        avg_loss = total_loss / len(train_loader)
        avg_train_accuracy = correct_predictions / total_samples
        train_accs.append(avg_train_accuracy)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_accuracy, epoch)

        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}')

        # Evaluation with DataLoader
        model.eval()
        total_samples_val = 0
        correct_predictions_val = 0 
        for batch_matid_val, batch_measurement_i_val, batch_row_index_val, batch_Y_val in val_loader:
            matrix_i_val = torch.tensor(padded_mat[batch_matid_val[0]]).float()

            # Forward pass for validation
            outputs_val = model((matrix_i_val.to(device), batch_matid_val.to(device), 
                                 batch_measurement_i_val.to(device), batch_row_index_val.to(device)))
            
            batch_Y_val = batch_Y_val.unsqueeze(1) if batch_Y_val.dim() == 1 else batch_Y_val

            # Calculate number of correct predictions
            predicted_labels_val = (outputs_val > 0.5).float() 
            correct_predictions_val += (predicted_labels_val == batch_Y_val.to(device)).sum().item()
            total_samples_val += batch_Y_val.size(0)

        # Calculate average accuracy
        avg_accuracy = correct_predictions_val / total_samples_val
        val_accs.append(avg_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {avg_accuracy:.4f}')
        writer.add_scalar('Accuracy/val', avg_accuracy, epoch)

        # Update best accuracy 
        is_best = avg_accuracy > best_acc
        best_acc = max(best_acc, avg_accuracy)
        
        # Save the model 
        if (args.save_it != -1 and epoch % args.save_it == 0) or (epoch == num_epochs-1): 
            save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': avg_accuracy, 
            'train_acc': avg_train_accuracy,
            'loss': avg_loss,   
            'is_best': is_best
            }
            utils_torch.save_checkpoint(save_dict, is_best, EXP_DIR, filename=f'checkpoint_e{epoch}.tar')
    
    elapsed_time = (time.time() - start_time)/60  # Time in minutes 
    logger.info(f"Total Training time: {elapsed_time} minutes")
    writer.close()