# Youtube 영상 한국어 및 영어 자막 생성

goorm KDT NLP 4조(언어의 마술4조)의 3차 프로젝트 페이지입니다.

프로젝트에 대한 더욱 자세한 내용은 `presentation.pptx`를 참고해주세요.

## About files and folders

```
/ckpt
    Pretrained weights for STT and NMT models
/static/videos
    youtube videos will be saved in this folder
templates
    Templates with jinja2
fastapi.ipynb
    A notebook file which launchs fastapi backend and models
models.py
    STT and NMT models
yt.py
    For downloading youtube videos, generating scripts and translating.
```
## 목차

1. 프로젝트 개요
2. 프로젝트 진행 프로세스
3. 프로젝트 결과
4. 자체 평가 및 보완

## 1. 프로젝트 개요

- **Task** : 한국어 YouTube 영상의 한국어 자막 생성 및 한영 번역(STT & NMT)
- **Evaluation methnod** : WER & BLEU score
- **Library & Framework** : Python 3.7.13, Hugging Face - transformers 4.20.1

프로젝트의 목표는 한국어 YouTube 영상에서 한국어 자막을 생성하고, 이를 영어 자막으로 번역하는 것입니다.

## 2. 프로젝트 진행 프로세스

## 2.1 STT(Speech to Text) model

- **사용 모델**: Hugging Face - wav2vec2 (pretrained weights `'w11wo/wav2vec2-xls-r-300m-korean'`)

- **학습 데이터셋**: KsponSpeech(AIHub 한국어 음성), AIHub 방송 컨텐츠 대화체 음성인식 일부(TS2)

### 2.2 NMT(Neural Machine Translation) model

- **사용 모델**: Encoder-Decoder model
    - Encoder: KoBERT(pretrained weights `'monologg/kobert'`)
    - Decoder: DistilGPT-2
- **학습 데이터셋**
    - AI Hub-한영번역 말뭉치 중 구어체, 대화체 데이터(50만 개)
    - AI Hub- 일상생활 구어체 한영 번역 병렬 말뭉치 데이터(120만 개)
    - Korpora-영화자막 한영병렬 말뭉치(120만 개)

## 3. 프로젝트 결과

TODO list

- WER, BLEU score로 프로젝트 결과를 보여주자.
- 데모 영상을 띄우자.
- 자체 평가 및 보완 작성

## 4. 자체 평가 및 보완
