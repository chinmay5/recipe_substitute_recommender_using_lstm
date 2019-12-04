class Vocab(object):
    
    def __init__(self):
        super(Vocab,self).__init__()
        self.word_2_idx = {}
        self.idx_2_word = {}
        self.max_idx = 0
        self.word_2_idx['<UNK>'] = 0
        self.idx_2_word[0] = '<UNK>'

    def add_word(self, word):
        if word in self.word_2_idx.keys():
            return
        self.max_idx += 1
        self.word_2_idx[word] = self.max_idx
        self.idx_2_word[self.max_idx] = word
        
    def get_idx(self, word):
        if word not in self.word_2_idx.keys():
            raise ValueError("Not acceptable as -- {} does not exist in vocabulary!".format(word))
        return self.word_2_idx[word]
    
    def get_word(self, idx):
        if idx not in self.idx_2_word.keys():
            raise ValueError("Not acceptable as -- {} does not exist in vocabulary!".format(idx))
        return self.idx_2_word[idx]
    
    def __repr__(self):
        return str(self.word_2_idx)
    
    def check_any_word_in_vocab(self, sentence):
        # Check if sentence has any word in vocabulary
        # highly inefficient but okay :(
        for any_ingredient in sentence.split(" "):
            if any_ingredient in self.word_2_idx.keys():
                return True, any_ingredient
        return False # for ease in construct later
    
