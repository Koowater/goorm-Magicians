# 기계독해(MRC, Machine Reading Comprehension) competition

goorm KDT NLP 4조(언어의 마술4조)의 2차 프로젝트 페이지입니다.

프로젝트에 대한 더욱 자세한 내용은 `presentation.pptx`를 참고해주세요.

## About files and folders

```
/data
    datasets or codes for training and validation
/examples
    documentations
train_with_wandb.ipynb
    training notebook with wandb
train_with_wandb_sweep.ipynb
    training notebook with wandb sweep for hyperparameter searching
dp.py
    data provider, preprocessor and postprocessor
utils.py
    evaluation metrics and custom loss functions
```
## 목차

1. 프로젝트 개요
2. 프로젝트 진행 프로세스
3. 프로젝트 결과
4. 자체 평가 및 보완

## 1. 프로젝트 개요

- **Task** : 신문 기사 기계독해
- **Evaluation methnod** : Levenshtein distance(edit distance)
- **Library & Framework** : Python 3.7.13, Hugging Face - transformers 4.20.1

프로젝트의 목표는 신문 기사 기계독해 성능을 올리기 위한 test dataset에 대한 편집거리 최소화입니다. Training dataset은 AIHub 내의 dataset과 별도 제공된 training dataset으로 제한되었습니다.

## 2. 프로젝트 진행 프로세스

성능 개선을 위하여 크게 네 가지로 해결 사항들을 설정하고 구현하였습니다.

- **Preprocessing**
- **Postprocessing**
- **LM training**
- **Dataset**

## 2.1 Preprocessing

### Doc stride

신문 기사 지문의 길이는 LM 입력 시퀀스의 최대 길이를 초과하므로 이를 해결하기 위해 **doc stride**를 적용하였습니다. 적용 방법은 Hugging Face 공식 문서를 참조하였습니다.

## 2.2 Postprocessing

후처리 과정은 다음과 같습니다.

1. *start, end position 교차 여부 확인*
    
    end가 start보다 앞에 존재하면 잘못된 추론이라고 판단하였습니다.

2. *추론된 start, end position으로 정답 문자열 생성*

3. *[UNK] token 복원*

    offset mapping을 통해 원본 context에서 원본 문자열을 참조하였습니다.

4. *Special token 삭제*

5. *pred answer의 max_len 초과 여부 확인*

    모델이 추론하는 정답의 길이를 제한하면 편집거리가 감소하였습니다. 최적의 max_len은 추후 실험을 통해 찾아냈습니다.

6. *답이 빈 문자열이라면 제거*

7. *답이 2개 이상 남아있다면 start, end logit의 합이 더 큰 경우를 최종 답으로 선정*

8. *정답 문자열의 끝 조사 제거, 특수문자 사이 공백 제거*

    추론된 정답의 마지막에 조사가 포함되는 경우가 다수 발견되어서, 이를 제거하고 불필요한 공백을 제거했습니다.

## 2.3 LM training

`monologg/koelectra-base-v3-finetuned-korquad` in Hugging Face models

KorQuAD Dataset에 사전학습된 ELECTRA model을 사용하였습니다. WandB sweep을 적용하여 최적의 hyperparameter를 탐색하였으며, 학습 후 별도의 validation dataset에 대한 편집거리를 측정하여 가장 좋은 성능을 보여주는 max_len을 찾았습니다.

## 2.4 Dataset

실험을 통해 AIHub 내의 행정 문서 데이터셋을 기존 학습 데이터셋과 함께 학습하는 것이 가장 좋은 성능을 나타내는 것을 확인했습니다. 이를 최종 학습 데이터셋으로 선정하였습니다.

## 3. 프로젝트 결과

[https://www.kaggle.com/competitions/k-digital-goorm-4-korean-mrc/leaderboard](https://www.kaggle.com/competitions/k-digital-goorm-4-korean-mrc/leaderboard)

6개의 팀 중 2위를 달성하였습니다.

## 4. 자체 평가 및 보완

- KoBigBird model을 사용하면 doc stride 없이 긴 지문을 적용할 수 있었습니다.
- LM이 추론한 start, end position과 logits를 통해 5개의 정답 후보를 생성하여 이 중에서 최종 정답을 선정하는 알고리즘이 가장 보편적인 정답 문자열 생성 방법이었다는 것을 1등 팀의 발표를 통해 알게 되었습니다.