# Tacotron 2 for Portuguese TTS
This is an adaptation of NVIDIA's tacotron2 repository to train and experiment with portuguese TTS models, following the work ["A Corpus of Neutral Voice Speech in Brazilian Portuguese"](https://www.smt.ufrj.br/gpa/propor2022). More info about the data and a notebook to directly generate speech from portuguese pretrained models can be found at [kaggle](https://www.kaggle.com/datasets/mediatechlab/gneutralspeech). To interact and contribute with with our data, models and code, please follow the instructions below.
 
---------------------------
                    


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup (Portuguese)
1. Download and extract the [G Neutral Speech Male Dataset](https://www.kaggle.com/datasets/mediatechlab/gneutralspeech)
2. Clone this repo: `git clone https://github.com/mediatechlab/tacotron2.git`
3. CD into this repo: `cd tacotron2`
4. Initialize submodule: `git submodule init; git submodule update`
5. Update .wav paths: `sed -i -- 's,DUMMY,gneutral_speech_dataset_folder/wavs,g' filelists/*.txt`
    - Alternatively, set `load_mel_from_disk=True` in `hparams.py` and update mel-spectrogram paths 
6. Install [PyTorch 1.0]
7. Install [Apex]
8. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our [Tacotron 2 Portuguese Model](https://drive.google.com/file/d/1HWlWM9lObk10NogCajYx2ILqbMWdBXo7/view?usp=sharing). 
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo
1. Download our published [Tacotron 2 and Waveglow Portuguese Models](https://drive.google.com/drive/folders/1OgP5foSPDsQBw1I64ZriS6vt3Pf9wj3L)
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb 

### *\*NEW\** Google Colab
Now you can try out tts in portuguese with studio quality in google colab with this [notebook](https://colab.research.google.com/drive/1Kz5ktn355ekeuMpDSXHjpx0pM_5dDwsn) (subject to our [terms of use](https://www.smt.ufrj.br/~gpa/terms_of_use.pdf)).

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

[Tacotron2-MMI](https://github.com/bfs18/tacotron2)

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) as described in our code.

We are inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)
Tacotron PyTorch implementation.

We are thankful to the Tacotron 2 paper authors, specially Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
