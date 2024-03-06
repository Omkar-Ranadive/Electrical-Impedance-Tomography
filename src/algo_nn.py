import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_cols):
        """
        RNN-based attention model 
        Args:
            input_size (int): Maximum number of cols of matrix 
            hidden_size (int): Number of features 
            output_size (int): Binary classification (output_size=1) (1=row is selected, 0=row not selected)
        """
        super(AttentionModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(max_cols, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2 + 1, output_size)  

    def forward(self, input_data):
        matrix_i, matrix_id, measurement_i, row_indices = input_data
        batch_size = matrix_id.size(0)

        # Select the row at row_index for all samples in the batch from matrix i ([batch_size, num_rows, num_features])
        embedded_row = torch.stack([matrix_i[idx, :] for idx in row_indices])
        # Shape: [batch_size, hidden_size]
        embedded_row = self.embedding(embedded_row) 

        # Process the matrix through the RNN
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(matrix_i.unsqueeze(0)) # Shape: [1, seq_len, hidden_size]
        rnn_output = rnn_output.expand(batch_size, -1, -1) # Shape: [batch_size, seq_len, hidden_size]

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


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        # self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = TransformerEncoderLayer(hidden_size, nhead, hidden_size, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size * 2 + 1, output_size)  


    def forward(self, input_data):
        matrix_i, matrix_id, measurement_i, row_indices = input_data
        batch_size = matrix_id.size(0)

        # Select the row at row_index for all samples in the batch from matrix i ([batch_size, num_rows, num_features])
        embedded_row = torch.stack([matrix_i[idx, :] for idx in row_indices])
        # Shape: [batch_size, hidden_size]
        embedded_row = self.embedding(embedded_row) 

        # # Positional encoding
        # embedded_row = self.pos_encoder(embedded_row)

        # Transformer encoder
        # Matrix_i inputted to transformer: [1, seq_len, num_features]
        transformer_output = self.transformer_encoder(matrix_i.unsqueeze(0))
        # transformer_output = transformer_output.transpose(0, 1)

        # Calculate context vector
        context_vector = torch.mean(transformer_output, dim=1)  # Use mean pooling as context vector [1, num_features]
        context_vector = context_vector.expand(batch_size, -1, -1)  # Expand to [batch_size, num_features]
        # Concatenate attended representation with embedded row and additional features
        # Context Vector and Embedded Row will be = [batch_size, hidden_size], measurement will be [batch_size, 1]
        combined_vector = torch.cat((context_vector.squeeze(1), embedded_row.squeeze(1), 
                                     measurement_i.unsqueeze(1)), dim=1)

        # Final classification
        output = torch.sigmoid(self.fc(combined_vector))
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Add to persistent state (Positional encodings won't be updated during backprop)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)