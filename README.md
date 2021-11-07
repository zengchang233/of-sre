# OneFlow-SRE
Speaker recognition implemented by oneflow library.

## Introduction

This repo contains several usual models including x-vector (TDNN, ETDNN, ECAPA-TDNN), r-vector (ResNet18 variants, ResNet34 variants).

It also contains several different training methodology such as few-shot learning (ProtoNet), discriminative learning (softmax variants)

### Requirements

```
oneflow>=0.6.0
python_speech_features
librosa
soundfile
tqdm
numpy
matplotlib
scipy
```

### feature

- [x] stft
- [x] log stft
- [x] fbank
- [x] log fbank
- [x] mfcc

### dataloader

- [x] variable length
- [x] balance batch sampler
- [ ] GPU feature extraction (once OneFlow supports acoustic feature extraction, I will implement it)

### Front-end

- [x] TDNN
- [x] ETDNN
- [ ] ECAPA-TDNN
- [ ] ResNet18
- [ ] ResNet34

### Pooling

- [x] statistics pooling
- [ ] attentive statistics pooling
- [ ] multi-head self-attentive pooling

### Loss function

- [x] softmax
- [x] am-softmax
- [ ] angular prototypical loss
- [ ] triplet loss

### Back-end

- [x] cosine similarity
- [ ] PLDA

### Trainer

- [x] Base trainer
- [x] NNet trainer
- [ ] Neural backend trainer

### Demos

#### Datasets

- [x] VoxCeleb 1&2
- [ ] CNCeleb 1&2

### Note

- Now it is a preliminary implementation without optimization so that the training speed is very slow (I think dataloader is the bottleneck)
- Once OneFlow update their package, I will optimize my code for speaker recognition.
- If you have any problem with this code, please contact me via zengchang.elec@gmail.com
