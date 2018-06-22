import torch
from .controller import Controller
from .head import ReadHead, WriteHead
import numpy as np

class NeuralTM(torch.nn.Module):
    def __init__(self, N = 128, M = 20, sz_controller = 100, 
            sz_sequence_vector = 8):
        torch.nn.Module.__init__(self)
        self.memory_bias = torch.zeros(N, M).fill_(1e-6)
        self.read_bias = torch.zeros(M).fill_(1e-6)
        self.weight_bias = torch.zeros(N)
        # set start position of heads to middle of memory
        self.weight_bias[N // 2] = 1 
        # extend size of sequence vector by 2 delimiter tokens
        sz_sequence_vector += 2
        self.controller = Controller(sz_sequence_vector, M, sz_controller)
        self.r_to_s = torch.nn.Linear(M, sz_sequence_vector)
        self.readhead = ReadHead(sz_controller, N, M)
        self.writehead = WriteHead(sz_controller, N, M)
    
    @property
    def device(self):
        # use weight of r_to_s as reference for selecting CUDA/CPU
        return self.r_to_s.weight.device
    
    def forward(self, sequence):
        
        dev = self.device
        # load dynamic network states from bias variables
        M_t = torch.autograd.Variable(self.memory_bias.data.clone()).to(dev)
        ww_t = torch.autograd.Variable(self.weight_bias.data.clone()).to(dev)
        rw_t = torch.autograd.Variable(self.weight_bias.data.clone()).to(dev)
        r_t = torch.autograd.Variable(self.read_bias.data.clone()).to(dev)
        # we init the following lists only for visualization
        ww_t_list = []
        rw_t_list = []
        a_t_list = []
        r_t_list = []
        
        sequence_out = torch.autograd.Variable(
            torch.zeros_like(sequence)
        ).to(dev)

        for i, s in enumerate(sequence):
            h_t = self.controller(s, r_t)
            M_t, ww_t, a_t = self.writehead(h_t, M_t, ww_t)
            r_t, rw_t = self.readhead(h_t, M_t, rw_t)
            s_out = torch.clamp(self.r_to_s(r_t), 0., 1.)
            sequence_out[i] = s_out
            a_t_list.append(a_t.data.cpu().numpy())
            r_t_list.append(r_t.data.cpu().numpy())
            ww_t_list.append(ww_t.data.cpu().numpy())
            rw_t_list.append(rw_t.data.cpu().numpy())

        return sequence_out, [
            np.array(l) for l in [ww_t_list, rw_t_list, a_t_list, r_t_list]
        ]
