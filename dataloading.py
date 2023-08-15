import sys
sys.path.append("plbert/")

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from IPython.display import Audio, display
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
import yaml
from transformers import AlbertConfig, AlbertModel

from phonemize import phonemize
from phonemizer.backend import EspeakBackend
from transformers import TransfoXLTokenizer
from text_normalize import normalize_text, remove_accents
from text_utils import TextCleaner

import whisper
import librosa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl



"""
PL-BERT SECTION
"""
plbert_root = "plbert/"
log_dir = plbert_root+"Checkpoint/"
config_path = os.path.join(log_dir, "config.yml")
plbert_config = yaml.safe_load(open(config_path))

albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
plbert = AlbertModel(albert_base_configuration)

files = os.listdir(log_dir)
ckpts = []
for f in os.listdir(log_dir):
    if f.startswith("step_"): ckpts.append(f)

iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
iters = sorted(iters)[-1]

checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')

state_dict = checkpoint['net']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    if name.startswith('encoder.'):
        name = name[8:] # remove `encoder.`
        new_state_dict[name] = v

plbert.load_state_dict(new_state_dict)

#I may need to make this function be able to batch input together (or I can later create a collate function)
#pad token = '$'
def tokenize(sents, global_phonemizer, tokenizer, text_cleaner):
    batched = []
    max_id_length = 0
    lengths = []
    for sent in sents:
        pretextcleaned = ' '.join(phonemize(sent, global_phonemizer, tokenizer)['phonemes'])
        cleaned = text_cleaner(pretextcleaned)
        batched.append(torch.LongTensor(cleaned))
        max_id_length = max(max_id_length, len(cleaned))
    phoneme_ids = torch.zeros((len(sents), max_id_length)).long()
    mask = torch.zeros((len(sents), max_id_length)).long()
    for i, c in enumerate(batched):
        phoneme_ids[i,:len(c)] = c
        mask[i,:len(c)] = 1
    return phoneme_ids, mask
def get_pltbert_embs(s, global_phonemizer, tokenizer, text_cleaner):
    """
    Input: list of texts

    Output: output of pretrained Albert model - (batch_size, num_tokens, 768)
    """
    phoneme_ids, attention_mask = tokenize(s, global_phonemizer, tokenizer, text_cleaner)
    return plbert(phoneme_ids, attention_mask=attention_mask).last_hidden_state

global_phonemizer = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True) #make sure brew install espeak and export location of .dylib
tokenizer = TransfoXLTokenizer.from_pretrained(plbert_config['dataset_params']['tokenizer'])
text_cleaner = TextCleaner()

"""
ECAPA SECTION (speaker encoding)
"""

ecapa = torch.jit.load("ecapa2_traced.jit")
def batch_spenc(batch):
    #batched input
    kwarg = {'input_signal': batch.cuda(), 'input_signal_length': torch.tensor([480000]*len(batch)).cuda()}
    return kwarg


"""
DATA LOADERS
"""
from torch.utils.data import Dataset
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
class MiipherDataset(Dataset):
    def __init__(self, noisy_filepath, clean_filepath, text_filepath, whisperEmbfilepathNoisy, whisperEmbfilepathClean):
        self.noisyfilepaths = noisy_filepath
        self.cleanfilepaths = clean_filepath
        self.sentpaths = text_filepath
        self.spencembs = []
        self.all_sents = []
        
        #change to enable batched input
        for x in tqdm(range(len(self.noisyfilepaths))):
            noisy, clean, text = self.noisyfilepaths[x], self.cleanfilepaths[x], self.sentpaths[x]

            #Text: load the text from the file as a string
            with open(text, 'r') as f:
                self.all_sents.append(f.read().strip())
                
            #Speaker Encoder: load audio file and save the loaded array
            # ref_path = noisy
            # audio, sr = librosa.load(ref_path, sr=16000)
            # audio = np.array([audio])
            # audio_signal = torch.tensor(whisper.pad_or_trim(audio))
            self.spencembs.append(noisy)

        batchNoises = []
        for batchNoisy in tqdm(whisperEmbfilepathNoisy):
            batchNoises.append(np.load(batchNoisy, mmap_mode='r'))
        num_batches = sum([a.shape[0] for a in batchNoises])
        self.whisper_noisy = []
        for loaded in batchNoises:
            for i in tqdm(range(len(loaded))):
                self.whisper_noisy.append(loaded[i,:,:])
        
        batchCleans = []
        for batchClean in tqdm(whisperEmbfilepathClean):
            batchCleans.append(np.load(batchClean, mmap_mode='r'))
        num_batches = sum([a.shape[0] for a in batchNoises])
        self.whisper_clean = []
        for loaded in batchCleans:
            for i in tqdm(range(len(loaded))):
                self.whisper_clean.append(loaded[i,:,:])
                
    def __len__(self):
        return len(self.noisyfilepaths)

    def __getitem__(self, idx):
        return self.whisper_noisy[idx], self.spencembs[idx], self.all_sents[idx], self.whisper_clean[idx]

def collate_fn(batch):
    """
    batch = list of tuples(whisper_noisy, speaker wav, raw sentences, whisper_clean)
    """
    whisper_noisy, speaker_wav, raw_sents, whisper_clean = [],[],[],[]
    for a,b,c,d in batch:
        whisper_noisy.append(torch.tensor(a, dtype=torch.float32))
        
        ref_path = b
        audio, sr = librosa.load(ref_path, sr=16000)
        audio = np.array([audio])
        audio_signal = torch.tensor(whisper.pad_or_trim(audio))
        speaker_wav.append(audio_signal)
        
        raw_sents.append(c) 
        whisper_clean.append(torch.tensor(d,dtype=torch.float32))
    whisper_noisy = torch.stack(whisper_noisy)
    whisper_clean = torch.stack(whisper_clean)

    plbertembs = get_pltbert_embs(raw_sents, global_phonemizer, tokenizer, text_cleaner)

    
    speaker_wav = torch.stack(speaker_wav).squeeze()
    batched = batch_spenc(speaker_wav)
    speakerembs = ecapa(**batched)[-1]

    return plbertembs.cpu().detach(), whisper_noisy.cpu().detach(), speakerembs.cpu().detach(), whisper_clean.cpu().detach()


""" 
Lightning Module 
"""

import pytorch_lightning as pl
class MiipherLightningModule(pl.LightningDataModule):
    def __init__(self, batch_size, collate_fn):
        super().__init__()
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.whisperEmbTrainClean = 'whisperEmbs/clean_trainset_embs'
        self.whisperEmbTrainNoisy = 'whisperEmbs/noisy_trainset_embs'
        self.whisperEmbTestClean = 'whisperEmbs/clean_testset_embs'
        self.whisperEmbTestNoisy = 'whisperEmbs/noisy_testset_embs'
        self.train_wav_clean = 'miipherTestDataset/train/clean_trainset_wav'
        self.train_wav_noisy = 'miipherTestDataset/train/noisy_trainset_wav'
        self.train_txt = 'miipherTestDataset/train/trainset_txt'
        self.test_wav_clean = 'miipherTestDataset/test/clean_testset_wav'
        self.test_wav_noisy = 'miipherTestDataset/test/noisy_testset_wav'
        self.test_txt = 'miipherTestDataset/test/testset_txt'
    
    def setup(self, stage):
        #TRAIN
        whisperembclean = [f"{self.whisperEmbTrainClean}/{a}" for a in sorted([f for f in os.listdir(self.whisperEmbTrainClean) if 'batch' in f], key=lambda x: int(x.split('_')[-1].split('.')[0]))]
        whisperembnoisy = [f"{self.whisperEmbTrainNoisy}/{a}" for a in sorted([f for f in os.listdir(self.whisperEmbTrainNoisy) if 'batch' in f], key=lambda x: int(x.split('_')[-1].split('.')[0]))]
        clean = [f"{self.train_wav_clean}/{a}" for a in sorted(os.listdir(self.train_wav_clean))]
        noisy = [f"{self.train_wav_noisy}/{a}" for a in sorted(os.listdir(self.train_wav_noisy))]
        text = [f"{self.train_txt}/{a}" for a in sorted(os.listdir(self.train_txt))]
        self.miipher_train = MiipherDataset(noisy, clean, text, whisperembnoisy, whisperembclean)

        #TEST
        whisperembclean = [f"{self.whisperEmbTestClean}/{a}" for a in sorted([f for f in os.listdir(self.whisperEmbTestClean) if 'batch' in f], key=lambda x: int(x.split('_')[-1].split('.')[0]))]
        whisperembnoisy = [f"{self.whisperEmbTestNoisy}/{a}" for a in sorted([f for f in os.listdir(self.whisperEmbTestNoisy) if 'batch' in f], key=lambda x: int(x.split('_')[-1].split('.')[0]))]
        clean = [f"{self.test_wav_clean}/{a}" for a in sorted(os.listdir(self.test_wav_clean))]
        noisy = [f"{self.test_wav_noisy}/{a}" for a in sorted(os.listdir(self.test_wav_noisy))]
        text = [f"{self.test_txt}/{a}" for a in sorted(os.listdir(self.test_txt))]
        self.miipher_test = MiipherDataset(noisy, clean, text, whisperembnoisy, whisperembclean)
    
    def train_dataloader(self):
        # print("-"*10 + "Data Loading Sanity Check" + "-"*10)
        # _, a, b, _ = self.miipher_train[5]
        # print(a,b)
        # print("-"*10 + "Data Loading Sanity Check DONE" + "-"*10)
        return DataLoader(self.miipher_train, batch_size=self.batch_size, collate_fn=collate_fn)
    
    def val_dataloader(self):
        # print("-"*10 + "Data Loading Sanity Check" + "-"*10)
        # _, a, b, _ = self.miipher_test[5]
        # print(a,b)
        # print("-"*10 + "Data Loading Sanity Check DONE" + "-"*10)
        return DataLoader(self.miipher_test, batch_size=self.batch_size, collate_fn=self.collate_fn)