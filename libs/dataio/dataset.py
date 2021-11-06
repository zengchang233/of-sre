import os
import math
import random
import logging
import warnings
import csv
import sys
warnings.filterwarnings('ignore')
sys.path.insert(0, "../../")
import glob

import numpy as np
import oneflow as of
from oneflow.utils.data import Dataset
import soundfile as sf
from scipy import signal

from libs.utils.utils import read_config

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to oneflow tensor
    audio, _ = sf.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats,axis=0).astype(np.float)

    return feat

class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):
        self.max_frames = max_frames
        self.max_audio  = max_frames * 160 + 240
        self.noisetypes = ['noise','speech','music']
        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        return np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):
        rir_file    = random.choice(self.rir_files)
        rir, _      = sf.read(rir_file)
        rir         = np.expand_dims(rir.astype(np.float),0)
        rir         = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]

class SpeechTrainDataset(Dataset):
    def __init__(self, opts):
        # read config from opts
        frame_range = opts['frames'] # frame number range in training
        self.lower_frame_num = frame_range[0]
        self.higher_frame_num = frame_range[1]
        TRAIN_MANIFEST = opts['train_manifest']
        self.rate = opts['rate']
        self.win_len = opts['win_len']
        self.win_shift = opts['win_shift']
        feat_type = opts['feat_type']
        #  musan_path = opts.get('musan', None)
        #  rirs_path = opts.get('rirs', None)
        #  self.augment_wav = AugmentWAV(musan_path, rirs_path, frame_range[-1])
        #  self.augment = False
        #  if (not musan_path is None) and (not rirs_path is None):
        #      self.augment = True

        if 'repeat' in opts:
            repeat = opts['repeat']
        else:
            repeat = True
        self.labels = []

        # read audio file path from manifest
        self.dataset = []
        current_sid = -1
        total_duration = 0

        count = 0
        
        with open(TRAIN_MANIFEST, 'r') as f:
            reader = csv.reader(f)
            for sid, _, filename, duration, samplerate in reader:
                if sid != current_sid:
                    self.dataset.append([])
                    current_sid = sid
                self.dataset[-1].append((filename, float(duration), int(samplerate)))
                total_duration += eval(duration)
                count += 1
        self.n_spk = len(self.dataset)

        # split dev dataset
        self.split_train_dev(opts['dev_number'])

        # compute the length of dataset according to mean duration and total duration
        total_duration -= self.dev_total_duration
        mean_duration_per_utt = (np.mean(frame_range) - 1) * self.win_shift + self.win_len
        if repeat:
            self.count = math.floor(total_duration / mean_duration_per_utt) # make sure each sampling point in data will be used
        else:
            self.count = count - opts['dev_number']

        # set feature extractor according to feature type
        #  if 'kaldi' in feat_type:
        #      from libs.dataio.feature import KaldiFeatureExtractor as FeatureExtractor
        #  elif 'wave' in feat_type:
        #      from libs.dataio.feature import RawWaveform as FeatureExtractor
        #  else:
        from libs.dataio.feature import FeatureExtractor
        try:
            feature_opts = read_config("conf/data/{}.yaml".format(feat_type))
        except:
            feature_opts = read_config("../../conf/data/{}.yaml".format(feat_type)) # for test
        self.feature_extractor = FeatureExtractor(self.rate, feat_type.split("_")[-1], feature_opts)

    def split_train_dev(self, dev_number = 1000):
        self.dev = []
        self.dev_total_duration = 0
        self.dev_number = dev_number
        i = 0
        while i < dev_number:       
            spk = random.randint(0, self.n_spk - 1)
            if len(self.dataset[spk]) <= 1:
                continue 
            utt_idx = random.randint(0, len(self.dataset[spk]) - 1)
            utt = self.dataset[spk][utt_idx]
            self.dev.append((utt, spk))
            self.dev_total_duration += utt[1]
            del self.dataset[spk][utt_idx]
            i += 1

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        idx = idx % self.n_spk
        return idx

    def collate_fn(self, batch):
        frame = random.randint(self.lower_frame_num, self.higher_frame_num) # random select a frame number in uniform distribution
        duration = (frame - 1) * self.win_shift + self.win_len # duration in time of one training speech segment
        samples_num = int(duration * self.rate) # duration in sample point of one training speech segment
        wave = []
        for sid in batch:
            speaker = self.dataset[sid]
            y = []
            n_samples = 0
            while n_samples < samples_num:
                aid = random.randrange(0, len(speaker))
                audio = speaker[aid]
                t, sr = audio[1], audio[2]
                samples_len = int(t * sr)
                if n_samples == 0:
                    start = int(random.uniform(0, t - 1.0) * sr)
                else:
                    start = 0
                #  start = int(random.uniform(0, t) * sr) # random select start point of speech
                _y, _ = self._load_audio(audio[0], start = start, stop = samples_len) # read speech data from start point to the end
                if _y is not None:
                    y.append(_y)
                    n_samples += len(_y)
            y = np.hstack(y)[:samples_num]
            #  if self.augment:
                #  augtype = random.randint(0,4)
                #  if augtype == 1:
                #      y = self.augment_wav.reverberate(y)
                #  elif augtype == 2:
                #      y = self.augment_wav.additive_noise('music',y)
                #  elif augtype == 3:
                #      y = self.augment_wav.additive_noise('speech',y)
                #  elif augtype == 4:
            #          y = self.augment_wav.additive_noise('noise',y)
            wave.append(y)
        feature = self.feature_extractor(wave)
        labels = of.tensor(batch)
        return feature, labels

    def _load_audio(self, path, start = 0, stop = None, resample = True):
        y, sr = sf.read(path, start = start, stop = stop, dtype = 'float32', always_2d = True)
        y = y[:, 0]
        return y, sr

    def get_dev_data(self):
        idx = 0
        while idx < self.dev_number:
            (wav_path, _, __), spk = self.dev[idx]
            data, _ = self._load_audio(wav_path)
            feat = self.feature_extractor([data])
            yield feat, of.LongTensor([spk])
            idx += 1

    def __call__(self):
        idx = 0
        wavlist = []
        spk = []
        for ind, i in enumerate(self.dataset):
            wavlist.extend(i)
            spk.extend([ind] * len(i))
        while idx < len(wavlist):
            wav_path, _, __ = wavlist[idx]
            data, _ = self._load_audio(wav_path)
            feat = self.feature_extractor(data)
            yield feat, spk[idx], os.path.basename(wav_path).replace('.wav', '.npy')
            idx += 1 

if __name__ == "__main__":
    from oneflow.utils.data import DataLoader
    #  from libs.utils.utils import BalancedBatchSamplerV2
    opts = read_config("../../conf/data.yaml")
    train_dataset = SpeechTrainDataset(opts)
    #  batch_sampler = BalancedBatchSamplerV2(train_dataset.n_spk, len(train_dataset), 500, 5)
    train_loader = DataLoader(train_dataset, shuffle = True, collate_fn = train_dataset.collate_fn, num_workers = 0)
    trainiter = iter(train_loader)
    feature, label = next(trainiter)
    print(feature.shape)
    print(label)
    #  output = of.unique(label)
    #  print(len(output))
