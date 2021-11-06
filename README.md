# OneFlow-SRE
Speaker recognition implemented by oneflow library.

## Introduction

This repo contains several usual models including x-vector (TDNN, ETDNN, ECAPA-TDNN), r-vector (ResNet18 variants, ResNet34 variants).

It also contains several different training methodology such as few-shot learning (ProtoNet), discriminative learning (softmax variants)

### feature

- [x] stft
- [x] log stft
- [x] fbank
- [x] log fbank
- [x] mfcc

### dataloader

- [ ] variable length
- [ ] balance batch sampler
- [ ] GPU feature extraction (once OneFlow supports acoustic feature extraction, I will implement it)

### Front-end

- [ ] TDNN
- [ ] ETDNN
- [ ] ECAPA-TDNN
- [ ] ResNet18
- [ ] ResNet34

### Pooling

- [ ] statistics pooling
- [ ] attentive statistics pooling
- [ ] multi-head self-attentive pooling

### Loss function

- [ ] softmax
- [ ] am-softmax
- [ ] angular prototypical loss
- [ ] triplet loss

### Back-end

- [ ] cosine similarity
- [ ] PLDA

### Trainer

- [x] Base trainer
- [x] NNet trainer
- [ ] Neural backend trainer

### Demos

#### Datasets

- [ ] VoxCeleb 1&2
- [ ] CNCeleb 1&2
