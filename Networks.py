import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.data.dataloader as dataloader
from torch.nn.utils.rnn import pad_sequence

class EmbeddingNetwork(nn.Module):
    
    def __init__(self, vocab_size, hidden_dim, embedding_dim, dropout=0.3):
        
        super(EmbeddingNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.GRU = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=dropout, num_layers=2)
        
    def forward(self, input_sequence, max_len):
#         print(input_sequence)
#         print(max_len)
        embedded = self.embedding(input_sequence)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, max_len, batch_first=True)
        outputs, self.h_n = self.GRU(packed)
        # Unpack padding
        """
            Honestly, I do not know if at this point, I need the output. I would rather prefer to work with the
            self.h_n cell and so will not `pad_padded_sequence`
        """
        sequence_embedded, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # on which output should the prediction be done?
        # self.h_n =  num_layers, batch_size, hidden_dim
        batch_size = self.h_n.shape[1]
        final_hidden_state = self.h_n[-1].view(batch_size, -1)  # Taking the output from the last stacked layer
        return final_hidden_state, sequence_embedded
    
class ClassificationNetwork(nn.Module):
    
    
    def __init__(self, embedding_model, hidden_dim, num_classes):
        super(ClassificationNetwork, self).__init__()
        self.embedding_model = embedding_model
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(num_classes * 2)
        self.relu = nn.ReLU()
        self.linear_first = nn.Linear(hidden_dim, num_classes * 2)
        self.linear = nn.Linear(num_classes * 2, num_classes)

    def forward(self, input_sequence, max_len):
        
        final_hidden_state, _ = self.embedding_model(input_sequence, max_len)
        output_predicted = self.linear(self.dropout(self.bn(self.relu(self.linear_first(final_hidden_state)))))
        return output_predicted

    def get_embedder(self):
        return self.embedding_model    
