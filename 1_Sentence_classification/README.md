# 문장 분류기 구현(Sentence classification)

goorm KDT NLP 4조(언어의 마술4조)의 2차 프로젝트 페이지입니다.

프로젝트에 대한 더욱 자세한 내용은 `presentation.pptx`를 참고해주세요.

## About files and folders

```
train_and_eval.ipynb
    train the model and save a csv file to evaluate test dataset
```

## 목차

1. 프로젝트 개요
2. 프로젝트 진행 프로세스
3. 프로젝트 결과
4. 자체 평가 및 보완

## 1. 프로젝트 개요

- **Task** : 영어 문장의 긍정, 부정 판단
- **Evaluation methnod** : kaggle competition score(accuracy)
- **Library & Framework** : Python 3.7.13, Hugging Face - transformers 4.20.1

프로젝트의 목표는 Yelp review dataset의 긍정, 부정을 판단하는 문장 분류기를 구현하여 best score에 도달하는 것입니다.

## 2. 프로젝트 진행 프로세스

## 2.1 버그 수정

프로젝트의 첫 번째 과제는 주어진 baseline code의 버그를 찾는 것입니다. 입력 데이터가 어떻게 처리되는지 살펴본 결과, test data collator에서 input_ids가 정렬되고 있어 labels와 순서가 맞지 않아 데이터 검증 시 정확도가 올바르게 측정되지 않는 버그를 찾아 수정하였습니다.

```python
def collate_fn_style_test(samples):
    input_ids = samples
    max_len = max(len(input_id) for input_id in input_ids)

    # --- Bug ---
    # sorted_indices = np.argsort([len(input_id) for input_id in input_ids])[::-1]
    sorted_indices = range(len(input_ids))

    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])
    input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in sorted_indices],
                             batch_first=True)
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids
```

## 2.2 LM 훈련

여러 LM을 훈련하여 성능을 측정한 결과, `DeBERTaV3`을 최종 모델로 선정하였습니다.

## 2.3 Data augmentation(데이터 증강)

다음과 같은 데이터 증강을 활용하였습니다. 전체 학습 데이터셋의 10%에 적용했습니다. 

- Back translation
- Contextual word embeddings
    - insert
    - substitute
- Synonym augmentation
- Random word augmentation

## 2.4 Data preprocessing(데이터 전처리)

- 학습 데이터셋 내의 중복 데이터를 제거하여 학습 시간을 절약하였습니다.
- 검증 데이터셋 내에 학습 데이터셋 내의 문장이 발견되어 이를 제거하였습니다. 이 과정을 통해 검증 신뢰도를 높일 수 있었습니다.

## 3. 프로젝트 결과

[https://www.kaggle.com/competitions/goorm-project-1-text-classification/leaderboard](https://www.kaggle.com/competitions/goorm-project-1-text-classification/leaderboard)

- Score 0.99를 달성하여 6개의 팀 중 2위를 달성하였습니다.

## 4. 자체 평가 및 보완

- WandB sweep을 통한 hyperparameter search를 적용하였다면 정확도 향상에 더욱 도움이 되었을 것입니다.