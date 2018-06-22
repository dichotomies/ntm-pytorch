import torch
import random

def data_loader(nb_sequences, sz_sequence_max, sz_sequence_min,
        sz_sequence_column, use_cuda = False):
    device = torch.device("cuda" if use_cuda else "cpu")
    for i in range(nb_sequences):
        sz_sequence = random.choice(
            range(sz_sequence_min, sz_sequence_max + 1)
        )
        sequence = torch.zeros(
            sz_sequence + 2,
            sz_sequence_column + 2
        )
        sequence[1:sz_sequence +1, :sz_sequence_column] = torch.randint(
            0,
            2,
            [sz_sequence, sz_sequence_column]
        )
        
        sequence[0, -2] = 1 # set delimiter token at start
        sequence[-1, -1] = 1 # set delimiter token at end
        
        enlarged_input_sequence = torch.autograd.Variable(
            torch.stack(
                [
                    *sequence, 
                    *torch.zeros(sz_sequence, sz_sequence_column + 2)
                ], 
                dim = 0
            )
        ).to(device)
                                                            
        enlarged_output_sequence = torch.autograd.Variable(
            torch.stack(
                [
                    *torch.zeros(sz_sequence + 2, sz_sequence_column + 2), 
                    *sequence[1:sz_sequence+1]
                ], 
                dim = 0
            )
        ).to(device)

        yield enlarged_input_sequence, enlarged_output_sequence

