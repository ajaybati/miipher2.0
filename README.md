# miipher2.0
Miipher2.0 is a custom implementation of the feature cleaner seen in [Miipher](https://arxiv.org/pdf/2303.01664.pdf) developed by Google to help restore degraded speech. It was used in the creation of LibriTTS-R. **We do not implement the vocoder and use an inhouse whisper-to-wav vocoder**.

The best checkpoint and associated files are here: [model_training-test4.py](https://github.com/ajaybati/miipher2.0/blob/main/model_training-test4.py), [dataloading_test1.py](https://github.com/ajaybati/miipher2.0/blob/main/dataloading_test1.py), and the associated tensorboard outputs: [tb_test4](https://github.com/ajaybati/miipher2.0/tree/main/tb_test4/miipher2). The checkpoint is available in google drive here: [Best checkpoints](https://drive.google.com/drive/folders/1ak6S3zPv-B0R8GXE-oKYWVJgf4ISNRG6?usp=sharing). CheckpointTest4.zip has the lower test loss but CheckpointTest4-iter3.zip has lower train loss.

Here is more information about Miipher and through process behind it: [Presentation](https://docs.google.com/presentation/d/1DbGY9jIiA8Gj6l3yPbz9z3JfDsNTqOHB59Fs_dq-DrM/edit?usp=sharing)


Here are some general notes on [SOTA techniques](https://docs.google.com/document/d/1JGwh8YdrtCTJpfHypn6vUG1Jn8HyIBXotkGXMsK72ko/edit?usp=sharing) used out there. 


## Methodology
There are 3 components that Miipher uses that we need to replace: 
- PNG-BERT - represents linguistic information 
- custom Conformer-based speaker encoder - conditioned to preserve speaker identity
- w2v-BERT - robust speech embeddings

We chose [PL-BERT](https://github.com/yl4579/PL-BERT) in palce of PNG-BERT, [ECAPA-TDNN](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/ecapa_tdnn) in place of their custom speaker encoder, and [Whisper](https://github.com/openai/whisper) embeddings in place of w2v-BERT.

We do not implement the fixed point iteration of the paper.

## Implementation Details
We had to tweak the architecture a bit (scaled down) for better results, but we believe on the right dataset and right set of hyperparameters, this model will have better results. Here is what the current successful architecture looks like: 


[<img src="./images/featurecleaner.png" width="250"/>](./images/featurecleaner.png)

We ignore the speaker identity aspect, feed the PL-BERT embeddings directly to the cross attention, and only stack 2x.


## Usage
The PL-BERT embeddings and speaker encoder embeddings are calculated on the fly during training. The Whisper embeddings, however, are preprocessed and stored (WARNING: this takes up a lot of space). Ideally, better resources should enable all inputs to be processed on the fly during training. 

To check out how the data is loaded go to [dataloading_test1.py](https://github.com/ajaybati/miipher2.0/blob/main/dataloading_test1.py). This is the architecture used for the best checkpoint: [model_training-test4.py](https://github.com/ajaybati/miipher2.0/blob/main/model_training-test4.py). This is also the training file, make sure to use this (and load any necessary checkpoint) for further training.

The original architecture used is located in [model_training.py](https://github.com/ajaybati/miipher2.0/blob/main/model_training.py).


## Results

Results can be seen in [unseen](https://github.com/ajaybati/miipher2.0/tree/main/unseen) and [trainOutputs](https://github.com/ajaybati/miipher2.0/tree/main/trainOutputs). Unseen represents audio that is unseen to the model during training and trainOutputs represents audio that is seen to the model during training. In each of the subfolders, there are 3 audio files: original, noisy (passed directly through vocoder), and miipher (passed through the miipher2.0 model then the vocoder).