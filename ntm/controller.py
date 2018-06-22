import torch

class Controller(torch.nn.Module):
    def __init__(self, sz_sequence_vector, sz_read_vector, sz_output):
        torch.nn.Module.__init__(self)
        self.layer1 = torch.nn.Linear(
            sz_sequence_vector + sz_read_vector, 
            sz_output
        )
    
    def forward(self, x, r_t):
        return torch.nn.functional.relu(
            self.layer1(
                torch.cat([x, r_t]))
            )
