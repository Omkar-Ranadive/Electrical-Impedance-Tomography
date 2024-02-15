from constants import EXP_PATH, DATA_PATH
import algo_nn
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os 
import logging 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    EXP_DIR = EXP_PATH / 'nn' / args.exp_name
    os.makedirs(EXP_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=(EXP_DIR / 'info.log'), format='%(message)s')
    logger = logging.getLogger() 

    X, Y = algo_nn.prepare_training_data()

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
    input_size = X_train[0][0][1].shape[0]  # Number of features in each row
    hidden_size = 120  # Hidden size of LSTM and embedding
    output_size = 1  # Binary classification (good or bad row)
    num_epochs = args.epochs
    print_every = 100

    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Hidden Size (lstm): {hidden_size}")
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = algo_nn.AttentionModel(input_size, hidden_size, output_size).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()


   # Training loop with DataLoader
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

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

            if i % print_every == 0: 
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch: {i}, Avg batch loss: {(batch_loss/(i+1)):.4f}')

        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        # Evaluation with DataLoader
        model.eval()
        total_accuracy = 0

        for batch_matrix_i_val, batch_measurement_i_val, batch_row_index_val, batch_Y_val in val_loader:
            # Forward pass for validation
            outputs_val = model((batch_matrix_i_val.to(device), batch_measurement_i_val.to(device), 
                                batch_row_index_val.to(device)))
            
            # Calculate accuracy
            batch_accuracy = ((outputs_val > 0.5).float().eq(batch_Y_val).sum().item()) / len(batch_Y_val)
            total_accuracy += batch_accuracy

        # Calculate average accuracy
        avg_accuracy = total_accuracy / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {avg_accuracy:.4f}')