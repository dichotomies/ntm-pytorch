
**Introduction**

Recently, there has been a resurgence in models that are able to complete sequential tasks with multiple computational steps by utilizing explicit storage and a notion of attention. In particular, the model class, which is referred to as memory augmented neural networks, has outreached the capabilities of traditional LSTMs at solving algorithmic problems. For example, Graves et al. (2014) demonstrated that Neural Turing Machines (NTMs) can infer simple algorithms such as copying and sorting sequences. In a more recent study, Graves et al. (2016) demonstrated that differentiable neural computers (DNCs) can successfully answer synthetic questions designed to emulate reasoning and inference problems in natural language (The NTM is the predecessor to the DNC). These achievements make memory augmented networks an important fundament for learning machines that can store knowledge quickly and reason about it flexibly. 

This repo shows the fundamental mechanisms of neural networks that utilize external memory on the basis of NTMs (Due to their simple, intuitive and yet effective design, they make a good starting point for this.). 

**Overview of Repository**

`dataloader`: producing input and target sequences that we feed into the NTM for training it to solve the copy task.
`checkpoints`: checkpoints for models that were trained with up to 100000 sequences. 
`ntm`: NTM implementation consisting of `controller`, `head` and `ntm`.

**Reproducing the Results**

You can verify the model functionality and see the read and write mechanisms in action with `reproducing-results.ipynb`. The notebook also contains scripts for training the model and loading the pretrained models from `checkpoints`.
