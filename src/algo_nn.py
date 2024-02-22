import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
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




