
This project shows the fundamental mechanisms of neural networks that utilize external memory on the basis of NTMs.

Verified setup for training and reproducing the results: python 3.5.2, torch 0.4.0, numpy 1.14.3.

**Overview of Repository**

`dataloader`: producing input and target sequences that we feed into the NTM for training it to solve the copy task.

`checkpoints`: checkpoints for models that were trained with up to 100000 sequences. 

`ntm`: NTM implementation consisting of `controller`, `head` and `ntm`.

**Reproducing the Results**

You can verify the model functionality and see the read and write mechanisms in action with `reproducing-results.ipynb`. The notebook also contains scripts for training the model and loading the pretrained models from `checkpoints`.
