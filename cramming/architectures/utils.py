
def get_size(idx, hidden_size, interm_size, embedding_decoder):

    def _idx_to_size(idx):
        if idx == -1: # Embedding layer
            if embedding_decoder == 'decoupled':
                return hidden_size//4, interm_size//4
            elif embedding_decoder == 'coupled' or embedding_decoder == 'scaled':
                return hidden_size, interm_size
            else:
                raise ValueError(f'Invalid embedding_decoder mode {embedding_decoder}')
        if idx == 0:
            if embedding_decoder == 'decoupled' or embedding_decoder == 'scaled':
                return hidden_size//4, interm_size//4
            elif embedding_decoder == 'coupled':
                return hidden_size, interm_size
            else:
                raise ValueError(f'Invalid embedding_decoder mode {embedding_decoder}')
        if idx <= 5:
            return hidden_size//4, interm_size//4
        elif idx <= 10:
            return hidden_size//2, interm_size//2
        else:
            return (hidden_size, interm_size)
    
    return _idx_to_size(idx)[0], _idx_to_size(idx)[1], _idx_to_size(idx+1)[0]
        