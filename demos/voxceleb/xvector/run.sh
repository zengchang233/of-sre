#! /bin/bash

stage=0
debug=false

. ./parse_options.sh

voxceleb1_dev=/home/smg/zengchang/data/voxceleb1/dev

if [ $stage -le -1 ]; then
    # prepare manifest.csv file, only implement once.
    python ../preprocess.py --path $voxceleb1_dev
fi

# train a frontend module
if [ $stage -le 0 ]; then
    # options:
    # --feat-type {python,kaldi}_{fbank,mfcc,spectrogram}
    # --input-dim 1 for resnet, feature dimension for xvector
    # --arch resnet, tdnn, etdnn, ... take ../../../libs/nnet/*.py as reference
    # --loss CrossEntropy, AMSoftmax, TripletLoss, ...
    # --bs batch size
    # --resume resume path
    # --device gpu or cpu
    # --mode depreciated
    if [ $debug == "true" ]; then
        python -m ipdb local/nnet/trainer.py --feat-type python_logfbank --arch tdnn --input-dim 80 \
            --device cuda --bs 64 --loss AMSoftmax # --resume exp/Wed_Nov_10_11_38_09_2021/net_9.pth
        echo "frontend training done!"
    else
        export CUDA_VISIBLE_DEVICES="0"
        if [ $CUDA_VISIBLE_DEVICES == "0" ]; then
            python local/nnet/trainer.py --feat-type python_logfbank --arch tdnn --input-dim 80 \
                --device cuda --bs 64 --loss AMSoftmax --resume exp/Wed_Nov_10_11_38_09_2021
        else
            nj=`echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1- | wc -l`
            python3 -m oneflow.distributed.launch --nproc_per_node $nj local/nnet/trainer.py \
                --feat-type python_logfbank --arch tdnn --input-dim 80 --device cuda --bs 64 --loss AMSoftmax
        echo "frontend training done!"
    fi
fi

##### PyTorch Result #####
# model config: 
    # TDNN layers: [512,512,512,512,1500], [[-2,-1,0,1,2],[-2,0,2],[-3,0,3],[0],[0]]
    # FC layers: fc1+activation+bn, fc2 (without activation and bn), fc3 + softmax
# feature: 80 dims kaldi fbank (log = true)
# data repeat: false
# loss: AMSoftmax (s=20,m=0.25)
# voxceleb1 dev as training set, voxceleb1 test as test set
    # no augmentation, no repeat: EER: 5.10% (xvector in kaldi EER is 5.302% as reference)
    # no augmentation, repeat: EER: 4.97% (xvector in kaldi EER is 5.302% as reference)
    # augmentation, no repeat: EER: 4.07% (xvector in kaldi EER is 5.302% as reference)
    # augmentation, repeat: EER: % (xvector in kaldi EER is 5.302% as reference)
    # augmentation, ASP (implemented by myself), no repeat: 3.94% (using asv subtools implementation, the result is 4.05%)
    # augmentation, MultiHeadAttentionPooling, no repeat: 3.87%
    # augmentation, MultiResolutionAttentionPooling, no repeat: 3.99%

##### OneFlow Result #####
# tdnn model config: 
    # TDNN layers: [512,512,512,512,1500], [[-2,-1,0,1,2],[-2,0,2],[-3,0,3],[0],[0]]
    # FC layers: fc1+activation+bn, fc2 (without activation and bn), fc3 + softmax
# feature: 80 dims python_speech_features logfbank
# data repeat: false
# loss: AMSoftmax (s=20,m=0.25)
# voxceleb1 dev as training set, voxceleb1 test as test set
    # no augmentation, no repeat: EER: 5.63% (xvector in kaldi EER is 5.302% as reference)
    # no augmentation, repeat: EER: % (xvector in kaldi EER is 5.302% as reference)
    # augmentation, no repeat: EER: 4.31% (xvector in kaldi EER is 5.302% as reference)
    # augmentation, repeat: EER: % (xvector in kaldi EER is 5.302% as reference)

# etdnn model config: 
    # TDNN layers: [512,512,512,512,512,512,512,1500], [[-2,-1,0,1,2],[-2,0,2],[-3,0,3],[0],[0],[0],[0],[0]]
    # FC layers: fc1+activation+bn, fc2 (without activation and bn), fc3 + softmax
# feature: 80 dims python_speech_features logfbank
# data repeat: false
# loss: AMSoftmax (s=20,m=0.25)
# voxceleb1 dev as training set, voxceleb1 test as test set
    # no augmentation, no repeat: EER: 5.28% (xvector in kaldi EER is 5.302% as reference)
    # no augmentation, repeat: EER: % (xvector in kaldi EER is 5.302% as reference)
    # augmentation, no repeat: EER: 4.16% (xvector in kaldi EER is 5.302% as reference)
    # augmentation, repeat: EER: % (xvector in kaldi EER is 5.302% as reference)

# evaluation on test set without backend (or using cosine backend)
if [ $stage -le 1 ]; then
    expdir=`ls -lht ./exp | grep "^d" | head -1 | rev | cut -d' ' -f1 | rev`
    python local/evaluation.py -e $expdir -m best_dev_model.pth -d cuda -l far
    echo "scoring with only frontend done!"
fi

# train a backend module
# if [ $stage -le 1 ]; then
#     python local/backend/backend_trainer.py
# fi
