from conformer.conformer.encoder import ConformerBlock

import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecSpeakerLabelModel
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import math
import os
import requests
import boto3
import wave
import sys
import contextlib
import collections
from tqdm import tqdm
from tqdm.notebook import tqdm
import yaml
from transformers import AlbertConfig, AlbertModel

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from posnet import PostNet
from dataloading import MiipherLightningModule, collate_fn

sys.path.append("plbert/")
from hparams import create_hparams
from hparams import HParams

from pytorch_lightning.utilities import grad_norm


class Miipher2(pl.LightningModule):
    """
    Args:
        num_classes (int): Number of classification classes

    Inputs: inputs, input_lengths
        - plbertDim: feature dimension of pl-bert embeddings 
        - spEncDim: feature dimension of speaker encoder embeddings 
        - whisperLen: first dimension (time steps) of whisper embeddings
        - whisperDim: feature dimension of whisper embeddings
        - modelDim: hidden_size that most of the model operates on
        - crossAttHeads: number of heads in cross attention layer
        - conformerBlockSettings: Conformer Block hyper-parameters (look above in ConformerBlock section)
        - hparams: PostNet hyper-parameters, look here: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py

    Returns: outputs, output_lengths
        - **outputs**
        - **output_lengths** 
    """
    def __init__(self, plbertDim, spEncDim, whisperLen, whisperDim, modelDim,
                 crossAttHeads, crossAttDim, conformerBlockSettings, hparams, learning_rate=1e-4) -> None:
        super(Miipher2, self).__init__()

        self.learning_rate = learning_rate
        #init 3 linear layers
        self.plbert_lin = nn.Linear(plbertDim, modelDim)
        self.spEnc_lin = nn.Linear(spEncDim, modelDim)
        self.whisper_lin = nn.Linear(whisperDim, modelDim)

        #init film1 == film2 (kernel=3, stride=1)
        self.film1conv1 = nn.Conv1d(modelDim, modelDim, 3, stride=1)
        self.film1conv2 = nn.Conv1d(modelDim, modelDim, 3, stride=1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

        #init positional embedding + iter count
        self.film2conv1 = None #help
        self.film2conv2 = None #help

                         
        #Stack 4 times:
            #init cross attention module - input (128), hidden_size (512)
            #init Conformer Block - attention -> input (128), hidden_size (512)
            #init layer norm
        self.cross_attention = nn.ModuleList([])
        self.layer_norm = nn.ModuleList([])
        self.conformerBlock = nn.ModuleList([])
        for x in range(1):
            # nn.Linear(modelDim, crossAttDim), 
            # nn.Linear(modelDim, crossAttDim), 
            # nn.Linear(modelDim, crossAttDim), 
            self.cross_attention.append(nn.MultiheadAttention(modelDim, crossAttHeads, dropout=0.1, batch_first=True))
            self.layer_norm.append(nn.LayerNorm([whisperLen, modelDim]))
            self.conformerBlock.append(ConformerBlock(*conformerBlockSettings))

        #make sure to reset hparams while loading it. input dimension to postnet is whisperDim => n_mel_channels=whisperDim. 
        #maybe change postnet_embedding_dim
        self.postnet = PostNet(hparams)
        self.whisper_proj = nn.Linear(crossAttDim, whisperDim)

        #LOSS
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()


    
    def forward(self, plbert_embs, spEnc_embs, whisper_embs):
        """
        Inputs: plbert_embs, spEnc_embs, whisper_embs
            - plbert_embs (Tensor[batch_size, num_tokens, hidden_size]): png-bert replacement; Phoneme-level embeddings from transcript
            - spEnc_embs (Tensor[batch_size, hidden_size]): Speaker Encoder Embeddings (speakerNet/Ecapa)
            - whisper_embs (Tensor[batch_size, seq_len, hidden_size]): w2v-bert replacement; Whisper encoder-level embeddings of audio
    
        Returns: outputs
            - outputs (Tensor[batch_size, seq_len, hidden_size]): predicted, clean whisper-level embeddings
        """
        #all linear output will be hidden_dim=modelDim
        plbert_out = self.plbert_lin(plbert_embs) #(b, num_tokens, model_dim)
        spEnc_out = self.spEnc_lin(spEnc_embs)
        whisper_out = self.whisper_lin(whisper_embs)

        #be careful about this, currently we have pltbert_out=(num_tokens, modelDim) and spEnc_out=(modelDim)
        #if we want to change the second dimension, we would need to transpose the output since pytorch is channels_first unlike tf
        plbert_out = torch.permute(plbert_out, (0,2,1)) #(b, modelDim, num_tokens)
        film1_out = self.film1conv2(self.leakyrelu(self.film1conv1(plbert_out))+spEnc_out.reshape((*spEnc_out.shape,1))) #(b, modelDim, num_tokens-r), where num_tokens is reduced by r because of kernel
        film1_out = torch.permute(film1_out, (0,2,1)) #(b, num_tokens-r, modelDim)
        for x in range(1):
            att = self.cross_attention[x]
            query, key, value = whisper_out, film1_out, film1_out
            att_out, _ = att(query, key, value) #be careful here
            # print(att_out.shape)
            whisper_out = whisper_out + att_out
            layer_out = self.layer_norm[x](whisper_out)
            conf_out = self.conformerBlock[x](layer_out)
            whisper_out = whisper_out + conf_out

        whisper_out = self.whisper_proj(whisper_out)
        whisper_permute = torch.permute(whisper_out, (0,2,1))
        post_out = self.postnet(whisper_permute)
        output = whisper_out + torch.permute(post_out, (0,2,1))
        # output = self.whisper_proj(output)
        return output

    def norm_loss(self, gt, preds):
        """
        Inputs: gt, preds
            - gt(Tensor[batch_size, seq_len, hidden_size])
            - preds(Tensor[batch_size, seq_len, hidden_size])
    
        Returns: loss (reduction='mean')
            - loss (Tensor[1]) = (1 norm + 2 norm + spectral convergence), reduced
        """
        norm1 = self.l1_loss(preds, gt)
        norm2 = self.l2_loss(preds, gt)
        spectr = norm2/((gt**2).sum()/np.prod(list(gt.shape)))
        loss = norm1 + norm2 + spectr
        return loss, norm1, norm2, spectr
    
    def training_step(self, train_batch, batch_idx):
        plbertembs, whisper_noisy, speakerembs, whisper_clean = train_batch
        logits = self.forward(plbertembs, speakerembs, whisper_noisy)
        loss, norm1, norm2, spectr = self.norm_loss(logits, whisper_clean)
        # print("-"*10+"DEBUGGING"+"-"*10)
        # print(logits)
        # print("*"*20)
        # print(whisper_clean)
        # print("*"*20)
        # print(loss, norm1, norm2, spectr)
        # print("-"*10+"DEBUGGING done"+"-"*10)
        self.log_dict(
            {'train_loss':loss,
             'train_norm1_loss':norm1,
             'train_norm2_loss':norm2,
             'train_spectral':spectr}, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
        )
        return norm2
    
    def validation_step(self, val_batch, batch_idx):
        plbertembs, whisper_noisy, speakerembs, whisper_clean = val_batch
        logits = self.forward(plbertembs, speakerembs, whisper_noisy)
        loss, norm1, norm2, spectr = self.norm_loss(logits, whisper_clean)
        self.log_dict(
            {'test_loss':loss,
             'test_norm1_loss':norm1,
             'test_norm2_loss':norm2,
             'test_spectral':spectr}, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=7)}}


    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)



MODEL_DIM = 256
whisperLen = 1500
whisper_dim = 768
plbert_dim = 768
speakernet_dim = 192
crossAttHeads = 8 #changed from 16->8
crossAttDim = 256

hparams = create_hparams()
hparams.n_mel_channels = whisper_dim
hparams.postnet_embedding_dim = 512

conformerBlockSettings = HParams(encoder_dim = MODEL_DIM,
    num_attention_heads = 4, #changed from 8->4
    feed_forward_expansion_factor = 2, #changed from 4->2
    conv_expansion_factor = 2,
    feed_forward_dropout_p = 0.1,
    attention_dropout_p = 0.1,
    conv_dropout_p = 0.1,
    conv_kernel_size = 11, #changed from 31->11
    half_step_residual = True)
conformerBlockSettings = conformerBlockSettings.tup()

miipher = Miipher2(plbert_dim, speakernet_dim, whisperLen, whisper_dim, MODEL_DIM,
                 crossAttHeads, crossAttDim, conformerBlockSettings, hparams)

checkpoint_callback = ModelCheckpoint(dirpath='checkpoints_scale_lr/',
                                      filename='{epoch}-{step}-{test_loss:.2f}',
                                      verbose=True,
                                      monitor="test_loss",
                                      mode='min',
                                      save_top_k=5)

logger = TensorBoardLogger("tb_logs_scale_lr", name="miipher2")
trainer = pl.Trainer(
    accelerator='gpu',
    logger=logger,
    callbacks=[checkpoint_callback],
    min_epochs=1,
    max_epochs=30,
)
data_module = MiipherLightningModule(4, collate_fn) #changed batch size from 8->4
# torch.autograd.set_detect_anomaly(True)

# lr_finder = trainer.tuner.lr_find(miipher, data_module)


# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()
# print("*"*20)
# print(new_lr)
# print("*"*20)
# miipher.learning_rate = 1
trainer.fit(miipher, data_module, ckpt_path="checkpoints_scale_lr/epoch=17-step=103842-test_loss=0.59.ckpt")