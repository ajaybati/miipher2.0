import glob
import pandas as pd
import numpy as np
import sys
import whisper
import time
import os
import torch
import librosa
from tqdm import tqdm


# ---------------------------------------------------------
def mkdir(path):
    folder = os.path.dirname(path)
    if folder == "":
        return
    if not os.path.exists(folder):
        os.makedirs(folder)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class Batch:
    def __init__(
        self, nbatch, save_batches
    ):
        self.nbatch = nbatch
        self.max_frames = 3000
        self.max_frames_per_batch = 3000
        self.nmel = 80
        self.fs = 16000
        self.save_batches = save_batches
        self.reset()
        self.batched_embs = np.zeros((self.save_batches, self.nbatch, 1500, 768))
        self.batch_batch_counter = 0
        self.num_files_saved = 0

    # ---------------------------------------------------------
    def reset(
        self,
    ):
        self.batch_counter = 0
        self.mel = np.zeros(
            (self.nbatch, self.nmel, self.max_frames_per_batch), dtype=np.float32
        )
        self.meta = []
        for b in range(self.nbatch):
            self.meta.append([])

    # ---------------------------------------------------------
    def save(self, path, full_batches=True, enc=None):
        self.num_files_saved += 1
        if full_batches:
            with open(f'{path}/batch{self.save_batches}_{self.num_files_saved}.npy', 'wb') as f:
                np.save(f, self.batched_embs.reshape((self.batched_embs.shape[0]*self.batched_embs.shape[1],1500,768)))
        else:
            with open(f'{path}/batch{self.save_batches}_{self.num_files_saved}.npy', 'wb') as f:
                np.save(f, enc)
        self.batched_embs = np.zeros((self.save_batches, self.nbatch, 1500, 768))
        self.batch_batch_counter = 0
        
        # for b in range(self.nbatch):
        #     for nfram, f_in in self.meta[b]:
        #         f_out = f_in.replace(path_in, path_out).replace(".wav", ".npy")
        #         mkdir(f_out)
        #         np.save(f_out, enc[b, : nfram // 2, :])


# ---------------------------------------------------------
# ---------------------------------------------------------
def process_data(flist, B, save_path):
    mkdir(save_path)
    model = whisper.load_model("small.en")
    N = len(flist)
    for i, f_in in enumerate(tqdm(flist)):
        wav, _ = librosa.load(f_in, sr=B.fs, duration=20)
        
        if wav.shape[0] < int(B.fs * 0.5):  # do not process files shorter than 0.5s
            continue
        mel = whisper.log_mel_spectrogram(wav)
        nfram = mel.shape[-1]
        if nfram > B.max_frames:  # do not process files longer than max_frames
            continue

        # write mel and metadata to batch
        wav = whisper.pad_or_trim(wav)
        mel = whisper.log_mel_spectrogram(wav)
        B.mel[B.batch_counter, :, :] = mel
        B.meta[B.batch_counter].append(
            (
                nfram,
                flist[i],
            )
        )
        B.batch_counter += 1
        
        if (B.batch_counter == B.nbatch):  # process data if all <nbatch> batches are full
            with torch.no_grad():
                enc = model.embed_audio(torch.from_numpy(B.mel).cuda()) #.cuda()
                enc = (
                    enc.cpu().numpy() #.cpu().detach().numpy()
                )  # shape = (nbatch, nfram, n_enc)
                B.batched_embs[B.batch_batch_counter, :,:,:] = enc
                B.reset()
            B.batch_batch_counter += 1

        if B.batch_batch_counter == B.save_batches:
            B.save(save_path)

        
    # process last batch
    with torch.no_grad():
        if B.batch_batch_counter > 0:
            B.batched_embs = B.batched_embs[:B.batch_batch_counter, :,:,:]
            B.save(save_path)
        if B.batch_counter > 0:
            enc = model.embed_audio(torch.from_numpy(B.mel).cuda()) #.cuda()
            enc = enc.cpu().numpy()  #.cpu().detach().numpy() , shape = (nbatch, nfram, n_enc)
            # B.batched_embs[B.batch_batch_counter, :,:,:] = enc
            # B.batch_batch_counter += 1
            B.save(save_path, full_batches=False, enc=enc[:B.batch_counter])


# NOTE:
# run with:
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=<gpu_num> python whisper_encode_batch.py
