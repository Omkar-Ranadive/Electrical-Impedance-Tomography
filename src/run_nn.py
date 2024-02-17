from constants import EXP_PATH, DATA_PATH
import algo_nn
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os 
import logging 
from utils_torch import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import time 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit training to matrices < max_rows")
    parser.add_argument("--save_every", type=int, default=-1, 
                        help="""
                        Models get saved every k checkpoints depending on this param. 
                        If -1, only final model gets saved.
                        """)

    args = parser.parse_args()
    EXP_DIR = EXP_PATH / 'nn' / args.exp_name
    os.makedirs(EXP_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=(EXP_DIR / 'info.log'), format='%(message)s')
    logger = logging.getLogger() 

    writer = SummaryWriter(log_dir=EXP_DIR)

    X, Y, max_cols = algo_nn.prepare_training_data(max_rows=args.max_rows)

    # Split the data into training and validation sets (90/10 split)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)
    
    # Convert training and validation data to custom dataset
    train_dataset = algo_nn.MatrixDataset(X_train, Y_train)
    val_dataset = algo_nn.MatrixDataset(X_val, Y_val)

    # Create DataLoader for training and validation data
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    logger.info("*"*20) 
    logger.info(f"Number of training samples: {len(X_train)}")
    logger.info(f"Number of validation samples: {len(X_val)}")
    logger.info(f"Total number of training batches: {len(train_loader)}")
    logger.info(f"Total number of validation batches: {len(val_loader)}")

    # Initialize the model
    input_size = max_cols # Number of features in each row
    hidden_size = max_cols  # Hidden size of LSTM and embedding
    output_size = 1  # Binary classification (good or bad row)
    num_epochs = args.epochs
    print_every = 500
    save_every = args.save_every
    best_acc = 0

    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Hidden Size (lstm): {hidden_size}")
    logger.info(f"Max row size of matrix: {args.max_rows}")
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = algo_nn.AttentionModel(input_size, hidden_size, output_size).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

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

        for i, (batch_matrix_i, batch_measurement_i, batch_row_index, batch_Y) in enumerate(train_loader):        
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model((batch_matrix_i.to(device), batch_measurement_i.to(device), batch_row_index.to(device)))

            # Compute loss
            batch_Y = batch_Y.unsqueeze(0) if batch_Y.dim() == 1 else batch_Y
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

            if i % print_every == 0: 
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
        total_accuracy = 0

        for batch_matrix_i_val, batch_measurement_i_val, batch_row_index_val, batch_Y_val in val_loader:
            # Forward pass for validation
            outputs_val = model((batch_matrix_i_val.to(device), batch_measurement_i_val.to(device), 
                                batch_row_index_val.to(device)))
            
            # Calculate accuracy
            batch_accuracy = ((outputs_val > 0.5).float().eq(batch_Y_val.to(device)).sum().item()) / len(batch_Y_val)
            total_accuracy += batch_accuracy

        # Calculate average accuracy
        avg_accuracy = total_accuracy / len(val_loader)
        val_accs.append(avg_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {avg_accuracy:.4f}')
        writer.add_scalar('Accuracy/val', avg_accuracy, epoch)

        # Update best accuracy 
        is_best = avg_accuracy > best_acc
        best_acc = max(best_acc, avg_accuracy)
        
        # Save the model 
        if save_every != -1 and (epoch % save_every == 0 or epoch == num_epochs-1): 
            save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': avg_accuracy, 
            'train_acc': avg_train_accuracy,
            'loss': avg_loss,   
            'is_best': is_best
            }
            save_checkpoint(save_dict, is_best, EXP_DIR, filename=f'checkpoint_e{epoch}.tar')
    
    elapsed_time = (time.time() - start_time)/60  # Time in minutes 
    logger.info(f"Total Training time: {elapsed_time} minutes")
    writer.close()