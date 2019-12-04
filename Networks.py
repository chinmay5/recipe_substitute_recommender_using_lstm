import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.data.dataloader as dataloader
from torch.nn.utils.rnn import pad_sequence

class EmbeddingNetwork(nn.Module):
    
    def __init__(self, vocab_size, hidden_dim, embedding_dim, dropout=0.3, bidirectional=False, glove=None):
        
        super(EmbeddingNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self._load_glove_weights(glove)
        # self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.GRU = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=dropout, num_layers=2, bidirectional=bidirectional)
        
    def _load_glove_weights(self, glove):
        glove_pretrained_weights = torch.FloatTensor(glove.vectors) # was earlier get_vectors()
        self.embedding_dim = glove_pretrained_weights.size(1)
        self.embedding = nn.Embedding(
            glove_pretrained_weights.size(0), glove_pretrained_weights.size(1)
        )
        self.embedding.weight = nn.Parameter(glove_pretrained_weights)
        self.embedding.weight.requires_grad = False
        
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
        # self.h_n =  num_layers * num_directions, batch, hidden_size
        batch_size = self.h_n.shape[1]
        final_hidden_state = self.h_n[-1].view(batch_size, -1)  # Taking the output from the last stacked layer
        #output of shape (batch_size, seq_len, num_directions * hidden_size)
        last_embedded = sequence_embedded[:,-1,:]
        last_embedding = last_embedded.view(batch_size, -1)
        #print("last embedding is {}".format(last_embedding.shape))
        #print("sequence output is {}".format(sequence_embedded.shape))
        return final_hidden_state, sequence_embedded # Literature suggests to use last output term for Classification and not h_n
                                                     # However, for us, the final_hidden_state works better
    
class ClassificationNetwork(nn.Module):
    
    
    def __init__(self, embedding_model, hidden_dim, num_classes, p=0.5, num_directions=1):
        super(ClassificationNetwork, self).__init__()
        self.embedding_model = embedding_model
        self.dropout = nn.Dropout(p)
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
