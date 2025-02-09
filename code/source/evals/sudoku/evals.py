import torch

def compute_board_accuracy(pred, Y, is_input):
    #print(pred.shape)
    B = pred.shape[0]
    pred = pred.reshape((B, -1, 9)).argmax(-1)  # Reshape `pred` to [B, 81, 9] and take the index of the maximum value (argmax) to get predicted numbers (0~8)
    Y = Y.argmax(dim=-1).reshape(B, -1)  # Convert target output `Y` from one-hot encoding to numerical representation and reshape to [B, 81]
    mask = 1 - is_input.reshape(B, -1)  # Reshape `is_input` to [B, 81] and compute its inverse mask (1 indicates blank spaces to be predicted, 0 indicates pre-filled numbers)
    
    num_blanks = mask.sum(1)  # Compute the total number of blank spaces to be filled for each sample
    num_correct = (mask * (pred == Y)).sum(1)  # Compute the number of correctly predicted blank spaces
    board_correct =  (num_correct == num_blanks).int()  # Determine whether each board is completely correct
    return num_blanks, num_correct, board_correct
