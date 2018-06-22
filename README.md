
This project shows the fundamental mechanisms of neural networks that utilize external memory on the basis of NTMs.

Verified setup for training and reproducing the results: python 3.5.2, torch 0.4.0, numpy 1.14.3.

# Overview of Repository

`dataloader`: producing input and target sequences that we feed into the NTM for training it to solve the copy task.

`checkpoints`: checkpoints for models that were trained with up to 100000 sequences. 

`ntm`: NTM implementation consisting of `controller`, `head` and `ntm`.

# Abstract Network Architecture

```
NeuralTM(
    (controller): Controller(
        (layer1): Linear(in=30, out=100, bias=True)
    )
    (r_to_s): Linear(in=20, out=10, bias=True)
    (readhead): ReadHead(
        (produce_k_t): Linear(in=100, out=20, bias=True)
        (produce_beta_t): Linear(in=100, out=1, bias=True)
        (produce_g_t): Linear(in=100, out=1, bias=True)
        (produce_s_t): Linear(in=100, out=3, bias=True)
        (produce_gamma_t): Linear(in=100, out=1, bias=True)
    )
    (writehead): WriteHead(
        (produce_k_t): Linear(in=100, out=20, bias=True)
        (produce_beta_t): Linear(in=100, out=1, bias=True)
        (produce_g_t): Linear(in=100, out=1, bias=True)
        (produce_s_t): Linear(in=100, out=3, bias=True)
        (produce_gamma_t): Linear(in=100, out=1, bias=True)
        (produce_e_t): Linear(in=100, out=20, bias=True)
        (produce_a_t): Linear(in=100, out=20, bias=True)
    )
)
```

# Verification of Functionality

Inputs and outputs, add and read vectors, read and write weightings:

![](https://raw.githubusercontent.com/dichotomies/ntm-pytorch/master/results/verification.png)

Training the NTM with sequences of length up to 20. Testing its generelization on sequences with length 10, 20, 60, 100:

![](https://raw.githubusercontent.com/dichotomies/ntm-pytorch/master/results/generalization.png)

# Reproducing the Results

You can verify the model functionality and see the read and write mechanisms in action with `reproducing-results.ipynb`. The notebook also contains scripts for training the model and loading the pretrained models from `checkpoints`.
