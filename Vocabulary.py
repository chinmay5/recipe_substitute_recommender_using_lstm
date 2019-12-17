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
        
    def get_all_words_in_vocab(self, sentence):
        # Check if sentence has any word in vocabulary
        # return all the matching words then
        # highly inefficient but okay :(
        matching_list = []
        for any_ingredient in sentence.split(" "):
            if any_ingredient in self.word_2_idx.keys():
            	matching_list.append(self.word_2_idx[any_ingredient])              
        return len(matching_list)>0, matching_list  # for ease in construct later
        
    def get_all_words_in_vocab_mit_map(self, sentence, input_map):
        # We first check for the entries in the map
        # If the entry is in the map values, we take the key and then return its corresponding vocabulary element
        matching_list = []
        for any_ingredient in sentence.split(" "):
            for key, value in input_map.items():
                if any_ingredient in value:
                   # The corresponding key should be used now as an entry in the vocabulary
                   matching_list.append(self.word_2_idx[key])
        return len(matching_list)>0, matching_list  # for ease in construct later
            
    
