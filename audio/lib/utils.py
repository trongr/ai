def ixes_to_string(IX_TO_CHAR, ixes):   
    """
    Convert a list of indices to a string
    """
    return "".join([IX_TO_CHAR[i] for i in ixes])

def batchX_to_str(IX_TO_CHAR, batchX):
    """
    Convert a batch of character indices of shape H x W (say 100 x 20), where 
    each row contains the characters in a sentence, and the rows are sentences
    in a batch. Each row is the previous row shifted in time by one, so the unique
    characters are the ones along the top row, and the right most column. This is
    the sequence we want to return.
    """
    width = batchX.shape[1]
    top = batchX[0]
    right = batchX[:, width - 1]
    joined = top.tolist() + right[1:].tolist()
    return ixes_to_string(IX_TO_CHAR, joined)