# MDER-MA: 멀티모달 감정 인식 (Multimodal Emotion Recognition)

ERD-MA 데이터셋의 텍스트(발화 텍스트) + 이미지(스펙트로그램)를 결합한 멀티모달 감정 인식 시스템이다.
사전학습된 텍스트/이미지 인코더를 활용하고, 다양한 fusion 방식과 백본 조합을 실험해서 최적 구조를 찾는다.

## 프로젝트 구조

```
├── common.py                          # 공통 모듈 (데이터 로딩, 전처리, 학습 파이프라인, 시각화)
├── model1_concat_fusion.ipynb         # mBERT + EfficientNet-B0 (Concat)
├── model2_cross_attention_fusion.ipynb # mBERT + EfficientNet-B0 (Cross-Attention)
├── model3_gated_fusion.ipynb          # mBERT + EfficientNet-B0 (Gated) ★ Best
├── model4_xlm_roberta_vit.ipynb       # XLM-RoBERTa + ViT-B/16 (Gated)
├── model5_klue_bert_convnext.ipynb    # KLUE-BERT + ConvNeXt-Small (Optimized Gated) ★ Best
├── model6_mdeberta_swin.ipynb         # mDeBERTa-v3 + Swin-Tiny (Gated)
├── requirements.txt
└── data/
    ├── ERD-MA Text/                   # 감정별 발화 텍스트 (.txt)
    └── ERD-MA Spectrogram/            # 감정별 스펙트로그램 (.jpg/.png)
```

## 환경

- Python 3.12, PyTorch 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB)
- transformers 5.5.0, timm 1.0.26

---

## 데이터 분석 및 전처리

### 원본 데이터

ERD-MA 데이터셋은 텍스트 파일과 음성에서 추출한 스펙트로그램 이미지가 감정별 폴더에 저장되어 있다. 파일명 규칙이 `감정_성별_번호_세션_발화번호`인데, 이게 일관적이지 않아서 전처리가 꽤 필요했다.

### 전처리 과정

**성별 미표기 파일 제거** — 파일명에 male/female이 없는 파일이 9개 있었다. `sad0` ~ `sad8` 이런 식으로 성별 정보가 아예 빠져 있어서 메타데이터 분석에 쓸 수 없다. 성별-감정 교차 분석을 위해 제거했다.

**chunk 파일 제거** — `sad_chunk_1` ~ `sad_chunk_20` 같은 청크 파일 20개가 섞여 있었다. 원본 발화를 잘라낸 파편이라 하나의 완전한 감정 표현이 아니다. 학습 데이터로 부적합하니까 제외했다.

**화이트(공백) 이미지 필터링** — 스펙트로그램인데 실제로 열어보면 거의 완전히 하얀 이미지가 있다. RGB 각 채널이 250 이상인 픽셀 비율이 5%를 넘으면 비정상 데이터로 판단해서 제거한다. 무음 구간이거나 변환 과정에서 오류가 난 것으로 보인다.

**빈 텍스트 제거** — 텍스트 파일이 있긴 한데 내용이 비어있는 경우도 있다. 텍스트 모달리티가 없으면 멀티모달 학습이 안 되니까 제외한다.

**파일명 오타 보정** — `fmale`을 `female`로 자동 보정한다. 이게 없으면 성별 필터에서 누락된다.

### 전처리 후 데이터

필터링 결과 **1,236개** 쌍이 남았다.

| 감정    | 개수 |
| ------- | ---- |
| Happy   | 312  |
| Sad     | 341  |
| Angry   | 271  |
| Neutral | 312  |

성별은 남성 659개, 여성 577개로 분포된다.

클래스 불균형 처리는 언더샘플링도 실험해봤는데 (최소 클래스 271개에 맞춰서 전부 271개로 줄임 → 1,084개), 데이터가 1,236개밖에 안 되는 상황에서 152개를 더 줄이니까 오히려 Sad recall이 급락했다. 그래서 전체 데이터를 사용하되 `class_weights`(역빈도 가중치)로 손실 함수에서 보정하는 방식으로 갔다.

---

## 모델 구조 설명

### 공통 구조

모든 모델은 동일한 구조를 따른다:

- **텍스트 인코더** → projection (Linear + ReLU + Dropout)
- **이미지 인코더** → projection (Linear + ReLU + Dropout)
- **Fusion** (모달리티 결합)
- **Classifier** (MLP → 4-class 분류)

인코더는 사전학습 가중치를 불러오고, 전체 파인튜닝한다. differential learning rate을 적용해서 사전학습 인코더는 낮은 lr(1e-5 ~ 2e-5), 새로 추가한 레이어는 높은 lr(5e-4 ~ 1e-3)로 학습한다.

### Fusion 방식 비교 (Model 1 ~ 3)

세 가지 fusion 방식의 핵심 차이:

**Concatenation Fusion (Model 1)** — 텍스트 [CLS] 벡터와 이미지 GAP 벡터를 projection 후 그냥 이어붙인다(concat). 가장 단순한 baseline인데, 두 모달리티를 동일 비중으로 합치기 때문에 한쪽 모달리티의 노이즈도 그대로 반영된다.

**Cross-Attention Fusion (Model 2)** — BERT 전체 시퀀스(T×768)와 EfficientNet 공간 특성맵(HW×1280)을 교차 어텐션으로 결합한다. Text→Image, Image→Text 양방향으로 어텐션을 걸어서 모달리티 간 세밀한 상호작용을 학습한다. 이론적으로는 가장 강력한데, MultiheadAttention 2개 + LayerNorm 2개가 추가돼서 소규모 데이터에서는 오히려 오버피팅이 심했다.

**Gated Fusion (Model 3)** — 텍스트/이미지 벡터를 concat한 뒤 sigmoid gate 네트워크를 통과시켜서 `gate * text + (1-gate) * image`로 결합한다. 게이트가 샘플별로 어떤 모달리티에 더 의존할지 자동 결정한다. 구조가 단순하면서도(Linear 2개) 적응적이라서 이 데이터셋에서 가장 좋은 성능을 냈다.

### 백본 조합 비교 (Model 3 ~ 6)

Gated Fusion이 가장 좋았으니까, 같은 fusion 방식에서 백본만 바꿔서 비교했다:

| 모델    | 텍스트 인코더              | 이미지 인코더    | 특징                                                  |
| ------- | -------------------------- | ---------------- | ----------------------------------------------------- |
| Model 3 | mBERT (multilingual-cased) | EfficientNet-B0  | 다국어 BERT + 경량 CNN                                |
| Model 4 | XLM-RoBERTa                | ViT-B/16         | 100개 언어 대규모 학습 + Vision Transformer           |
| Model 5 | KLUE-BERT                  | ConvNeXt-Small   | 한국어 특화 BERT + 모던 CNN (ImageNet-22k pretrained) |
| Model 6 | mDeBERTa-v3                | Swin Transformer | 디센탱글드 어텐션 SOTA + 계층적 ViT                   |

Model 5는 원래 ConvNeXt-Tiny로 시작했는데, KLUE-BERT와 조합했을 때 학습이 전혀 수렴하지 않았다(4 epoch 동안 25% 정체). ConvNeXt-Small(ImageNet-22k pretrained)로 변경하고, 텍스트 풀링을 CLS+Mean 혼합으로, gate를 softmax 기반 modality-wise gate로 재설계한 뒤에야 정상 학습이 시작됐다.

Model 6의 mDeBERTa-v3는 가중치가 fp16으로 저장되어 있어서 연산 중 overflow → NaN이 발생했다. `text_encoder.float()`로 fp32 캐스팅하고, projection 앞에 LayerNorm을 추가해서 해결했다.

---

## 성능 향상 전략

### 데이터 증강 (Image Augmentation)

학습용 이미지에만 augmentation을 적용한다. 스펙트로그램 특성에 맞게 과하지 않게 설정했다:

- `RandomHorizontalFlip(p=0.5)` — 좌우 반전
- `RandomAffine(degrees=5, translate=(0.05, 0.05))` — 미세 회전/이동
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)` — 밝기/대비 변동

검증/테스트에는 Resize + Normalize만 적용한다.

### Dropout / Regularization

- Projection 레이어와 Classifier에 Dropout 0.3 적용
- AdamW의 weight_decay=0.01로 L2 정규화
- Gradient clipping (max_norm=1.0)으로 학습 안정화

Model 1에서 dropout을 0.4로 올려봤는데, 데이터가 적은 상태에서 정규화를 너무 강하게 걸면 학습 자체가 부족해진다. 0.3이 적절했다. 다만 Model 5처럼 backbone이 큰 경우(ConvNeXt-Small 50M)에는 0.4 + weight_decay 0.02 조합이 오히려 과적합 억제에 효과적이었다.

### Learning Rate Scheduling

두 가지 스케줄러를 실험했다:

- **CosineAnnealingLR** — 에포크 단위로 lr을 cosine 곡선으로 감소. Model 3에서 사용했고 가장 안정적이었다.
- **Warmup + Cosine** — 초반 N 스텝 동안 lr을 선형으로 올린 후 cosine 감소. Model 1에서는 warmup 100 스텝으로 인코더 초반 불안정을 방지했고, Model 5에서는 warmup 150 스텝으로 ConvNeXt-Small의 느린 수렴을 보완했다.

인코더와 classifier의 lr을 분리하는 differential LR이 핵심이다. 사전학습 가중치는 천천히(1e-5~2e-5), 새 레이어는 빠르게(5e-4~1e-3) 학습시킨다.

### Class Imbalance 처리

**Weighted Cross-Entropy Loss** — 클래스별 역빈도 가중치를 계산해서 손실 함수에 반영한다. 샘플이 적은 Angry(271개)에 더 큰 가중치를 줘서 학습 시 균형을 맞춘다.

```python
cnt = train_df['label_id'].value_counts().sort_index().values
w = 1.0 / cnt
w = w / w.sum() * num_classes
class_weights = torch.FloatTensor(w).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Stratified Split** — train/val/test 분리 시 클래스 비율을 유지하는 stratified split을 사용한다. 어느 한 감정이 특정 셋에 몰리는 걸 방지한다.

언더샘플링(최소 클래스에 맞춰 나머지 감소)도 해봤는데, 1,236개에서 1,084개로 줄이니까 Sad recall이 0.80→0.61로 급락했다. 데이터가 이미 적은 상황에서는 weighted loss가 더 낫다.

---

## 실험 결과

### 모델별 성능 비교 (Model 1 ~ 3)

| 모델        | Fusion          | Test Acc   | F1 macro   | Best Val Acc |
| ----------- | --------------- | ---------- | ---------- | ------------ |
| Model 1     | Concatenation   | 0.8602     | 0.8606     | 0.8649       |
| Model 2     | Cross-Attention | 0.8495     | 0.8505     | 0.8757       |
| **Model 3** | **Gated**       | **0.9140** | **0.9136** | **0.9243**   |

Model 3 (Gated Fusion)이 압도적으로 좋다. Test Accuracy 91.4%, F1 macro 91.4%.

### 오버피팅 패턴

학습곡선을 보면 차이가 확실하다:

- Model 1: val_loss가 0.37에서 1.14로 폭증. train_acc 0.98인데 val은 0.86에서 멈춤 (gap 12%)
- Model 2: val_loss가 0.47에서 0.78로 상승. train_acc 0.99 vs val 0.88 (gap 11%)
- Model 3: val_loss가 0.19에서 0.31로 소폭 상승. train_acc 0.98 vs val 0.92 (gap 6%)

Gated의 단순한 구조(Linear 2개짜리 gate)가 소규모 데이터셋에서 오히려 일반화가 잘 된다.

### 클래스별 성능 (Model 3 기준)

| 감정    | Precision | Recall | F1   |
| ------- | --------- | ------ | ---- |
| Happy   | 0.95      | 0.87   | 0.91 |
| Sad     | 0.95      | 0.80   | 0.87 |
| Angry   | 0.85      | 1.00   | 0.92 |
| Neutral | 0.90      | 1.00   | 0.95 |

Angry와 Neutral은 거의 완벽하다. Sad가 recall 0.80으로 가장 낮은데, Sad→Angry 오분류가 6건 있었다. 슬픔과 분노의 음성 특성이 겹치는 부분이 있어서 발생하는 것으로 보인다.

### Model 4 ~ 6 성능 비교

Gated Fusion이 가장 효과적이었으므로, 동일한 fusion 구조에서 백본 조합만 변경해서 비교했다.

| 모델        | 텍스트 인코더 | 이미지 인코더      | Test Acc   | F1 macro   | 비고                              |
| ----------- | ------------- | ------------------ | ---------- | ---------- | --------------------------------- |
| Model 3     | mBERT         | EfficientNet-B0    | 0.9140     | 0.9136     | 기준선                            |
| Model 4     | XLM-RoBERTa   | ViT-B/16           | -          | -          | OOM (364M params, 8.5GB GPU 한계) |
| **Model 5** | **KLUE-BERT** | **ConvNeXt-Small** | **0.9516** | **0.9518** | **최고 성능**                     |
| Model 6     | mDeBERTa-v3   | Swin-Tiny          | 0.9516     | 0.9520     | fp16→fp32 변환 필요               |

**Model 4 (XLM-RoBERTa + ViT-B/16)** — 364M 파라미터로 RTX 5060 Laptop(8.5GB)에서 OOM이 발생했다. batch_size를 2까지 줄여도 epoch 6에서 CUDA memory fragmentation으로 중단됐다. 인코더를 freeze하면 메모리 문제는 해결되지만 성능이 0.74로 급락해서, 이 GPU 환경에서는 실용적이지 않다.

**Model 5 (KLUE-BERT + ConvNeXt-Small)** — 한국어 특화 BERT와 ImageNet-22k pretrained ConvNeXt-Small 조합이다. 원래 ConvNeXt-Tiny로 시작했는데 GPU가 남아돌고 학습 지표도 아쉬워서(4 epoch 동안 25% 정체) 아키텍처를 재설계했다:

- 텍스트 풀링: [CLS] 단독 → CLS + Mean pooling 혼합 (0.5:0.5)
- Gate: element-wise sigmoid → modality-wise softmax (2차원)
- Residual fusion: gate collapse 방지를 위해 `fused + 0.1*(text+image)` 추가
- 활성화 함수: ReLU → GELU, projection 앞에 LayerNorm 추가
- 스케줄러: warmup_cosine (warmup 150 steps), 에포크 30, patience 8

결과적으로 Test Accuracy 0.9516, F1 macro 0.9518로 전체 모델 중 최고 성능을 달성했다. 특히 Neutral recall 1.00, Angry recall 0.98로 거의 완벽했고, Happy/Sad도 0.94/0.90으로 Model 3보다 크게 개선됐다.

**Model 6 (mDeBERTa-v3 + Swin-Tiny)** — mDeBERTa-v3의 가중치가 fp16으로 저장되어 있어서 연산 중 overflow → NaN loss가 발생했다. `text_encoder.float()`로 fp32 캐스팅 + LayerNorm 추가로 해결했다. 해결 후 Test Accuracy 0.9516, F1 macro 0.9520으로 Model 5와 동일한 수준의 성능을 달성했다. batch_size=2로 15 epoch 전체 학습이 가능했고, Angry recall 1.00, Neutral recall 0.98로 안정적이다. 다만 Sad recall이 0.86으로 Model 5(0.90)보다 낮았다.

### 클래스별 성능 비교 (Model 3 vs Model 5 vs Model 6)

| 감정    | Model 3 F1 | Model 5 F1 | Model 6 F1 |
| ------- | ---------- | ---------- | ---------- |
| Happy   | 0.91       | 0.94       | 0.95       |
| Sad     | 0.87       | 0.93       | 0.92       |
| Angry   | 0.92       | 0.95       | 0.95       |
| Neutral | 0.95       | 0.99       | 0.99       |

Model 3에서 가장 약했던 Sad(F1 0.87)가 Model 5에서 0.93, Model 6에서 0.92로 크게 개선됐다. KLUE-BERT가 한국어 텍스트의 감정 뉘앙스를 더 잘 포착하고, ConvNeXt-Small/Swin-Tiny가 스펙트로그램의 시각적 패턴을 더 세밀하게 추출한 결과로 보인다. Model 5와 Model 6은 전혀 다른 백본 조합인데 거의 동일한 성능(0.9516)을 달성했다는 점이 흥미로운데, Gated Fusion 구조가 백본의 차이를 효과적으로 흡수한다는 의미다.

### 성능 극대화: train+val 통합 학습

Model 5, 6에서는 최종 성능 극대화를 위해 validation 데이터를 학습 데이터에 포함시켜 재학습했다. 기존에 val은 early stopping 판단용으로만 사용되었는데, 최종 모델에서는 train(865개) + val(185개) = 1,050개로 학습량을 ~21% 늘렸다. test 데이터(186개)는 절대 학습에 포함하지 않는다.

| 모델    | 기본 Test Acc | train+val Test Acc | 변화    |
| ------- | ------------- | ------------------ | ------- |
| Model 5 | 0.9516        | **0.9731**         | +2.15%p |
| Model 6 | 0.9516        | **0.9731**         | +2.15%p |

두 모델 모두 동일한 폭의 성능 향상을 보였다. 데이터가 1,236개밖에 안 되는 상황에서 185개(~21%)를 추가하는 것만으로 이 정도 향상이 나온다는 건, 기존 모델이 데이터 부족으로 성능이 제한되었다는 의미다.

### 추가 실험: SpecAugment + Mixup

Model 5 train+val 데이터에 스펙트로그램 특화 augmentation을 추가 적용했다:

- **SpecAugment** — 주파수/시간 축 랜덤 마스킹 (freq_mask=15, time_mask=25, 마스크 2개 중첩)
- **Mixup (alpha=0.4)** — 이미지만 Beta 분포 lambda로 선형 보간, 텍스트는 유지

| 설정                 | Test Acc   | F1 macro   | 비고                    |
| -------------------- | ---------- | ---------- | ----------------------- |
| train+val 단독       | **0.9731** | **0.9728** | 최고 성능               |
| + SpecAug+Mixup 15ep | 0.9409     | 0.9416     | underfitting            |
| + SpecAug+Mixup 30ep | 0.9570     | 0.9576     | epoch 22 early stopping |

에포크를 15→30으로 늘리고 early stopping(patience=5)을 추가하니 0.9409→0.9570으로 개선됐지만, train+val 단독(0.9731)에는 못 미쳤다.

성능 하락 원인: 기본 augmentation(flip, affine, jitter) + SpecAugment + Mixup = 3중 정규화가 1,050개 데이터에서 과하다. 특히 Mixup은 스펙트로그램의 주파수 축 의미 구조를 파괴해서 자연 이미지와 다른 결과가 나온다. SpecAugment도 감정별 특징 주파수 대역을 마스킹하면 핵심 정보가 손실된다.

---

## 성능 개선 과정

### 시도했던 것들

**클래스 불균형 수정 (언더샘플링)** — 최소 클래스 Angry(271개)에 맞춰서 전부 271개로 줄여봤다. 총 1,084개. 결과적으로 Sad recall이 0.80에서 0.61로 급락했다. 데이터가 이미 적은 상황에서 70개(Sad)를 더 빼니까 다양성이 줄어든 거다. 다시 전체 사용 + weighted loss로 복구했다.

**encoder_lr 조정** — 2e-5에서 1e-5로 줄여봤는데, cosine 스케줄러랑 조합하면 초반 lr이 너무 높다가 급감해서 val loss가 폭발했다. 2e-5로 복구하니까 안정적이었다.

**weight_decay 강화** — 0.01에서 0.02로 올려봤는데, 데이터가 줄어든 상태에서 정규화를 동시에 강화하니까 역효과였다. 0.01이 적절하다.

**dropout 조정** — 0.3에서 0.4로 올려봤다. Concat이나 Cross-Attention에서는 별 효과 없었고, Gated에서는 0.3이 최적이었다.

**스케줄러 변경** — warmup_cosine도 시도했는데, Model 3처럼 mBERT+EfficientNet 조합에서는 plain cosine이면 충분했다. 반면 Model 5(KLUE-BERT+ConvNeXt-Small)에서는 warmup 150 스텝이 필수적이었다. ConvNeXt-Small은 backbone이 크고 수렴이 느려서 초반 lr이 너무 높으면 gradient가 불안정해진다.

**ConvNeXt-Tiny → ConvNeXt-Small 변경 (Model 5)** — KLUE-BERT + ConvNeXt-Tiny 조합에서 학습이 전혀 수렴하지 않았다. 4 epoch 동안 train accuracy가 25%에서 움직이지 않았는데, ConvNeXt-Tiny(28M)의 feature 표현력이 KLUE-BERT(110M)와 불균형이었던 것으로 보인다. ConvNeXt-Small(50M, ImageNet-22k pretrained)로 바꾸고 아키텍처를 전면 재설계한 뒤에 95.16%까지 도달했다.

**mDeBERTa-v3 fp16 NaN 문제 (Model 6)** — mDeBERTa-v3는 HuggingFace에서 가중치가 fp16으로 저장되어 있다. 디센탱글드 어텐션 연산 중 값이 overflow → NaN이 전파되는 문제가 있었다. `self.text_encoder.float()`로 전체 가중치를 fp32로 캐스팅하고, projection 앞에 LayerNorm을 추가해서 해결했다.

**XLM-RoBERTa + ViT-B/16 메모리 문제 (Model 4)** — 364M 파라미터 모델은 RTX 5060 Laptop(8.5GB)에서 batch_size=2로도 epoch 6에서 OOM이 발생했다. AdamW의 optimizer states(momentum + variance)가 파라미터당 2배 메모리를 추가 할당하는데, 첫 번째 optimizer.step() 시점에 한꺼번에 ~2.9GB가 잡힌다. 인코더를 freeze하면 trainable params가 0.66M으로 줄어서 메모리 문제는 해결되지만, feature extraction만으로는 성능이 0.74에 그쳤다.

### 가장 효과적이었던 것

결국 **fusion 방식 자체를 바꾼 것**이 가장 큰 성능 향상을 가져왔다. Concat(0.86) → Gated(0.91)로 5%p 상승. 그 다음으로 **백본 조합 변경**이 효과적이었다. mBERT+EfficientNet(0.91) → KLUE-BERT+ConvNeXt-Small(0.95)로 4%p 추가 상승. 하이퍼파라미터 튜닝보다 구조적 선택이 중요했다.

---

## 분석 및 결론

이 프로젝트에서는 fusion 방식 3가지(Concat, Cross-Attention, Gated)와 백본 조합 4가지를 실험했다.

**Fusion 방식**: Cross-Attention은 이론적으로 모달리티 간 세밀한 상호작용을 학습하는 구조인데, 실제로는 파라미터가 많아서 1,236개 소규모 데이터셋에서 오버피팅이 심했다. Gated Fusion은 sigmoid gate 하나로 모달리티 기여도를 조절하는 단순한 구조인데, 이게 오히려 일반화를 잘 한다. gate가 하는 역할이 핵심이다 — 스펙트로그램 품질이 낮거나 텍스트가 더 명확한 샘플에서는 텍스트 쪽에 가중치를 주고, 반대 상황에서는 이미지 쪽에 가중치를 준다.

**백본 선택**: 한국어 데이터에는 다국어 mBERT보다 한국어 특화 KLUE-BERT가 더 효과적이었다. 이미지 쪽에서도 EfficientNet-B0(5.3M)보다 ConvNeXt-Small(50M, ImageNet-22k pretrained)이 스펙트로그램의 주파수-시간 패턴을 더 잘 포착했다. 다만 모델이 너무 크면(XLM-RoBERTa + ViT-B/16, 364M) GPU 메모리 한계에 부딪혀서, 백본 크기와 GPU 리소스 사이의 균형이 중요하다.

**학습 안정화**: 백본이 커질수록 하이퍼파라미터 민감도가 올라간다. ConvNeXt-Small은 warmup이 필수였고, mDeBERTa-v3는 fp16→fp32 캐스팅이 필요했다. 단순히 backbone을 교체하는 것만으로는 안 되고, 각 조합에 맞는 학습 전략(LR 스케줄, dropout, gate 설계)을 같이 조정해야 한다.

**데이터 전처리**: 화이트 이미지, 성별 미표기 파일, chunk 파일을 걸러내지 않으면 노이즈 데이터가 학습을 방해한다. 클래스 불균형은 weighted loss로 충분했고, 언더샘플링은 이 규모에서는 오히려 역효과였다.

**데이터 증강의 한계**: SpecAugment + Mixup 추가 시 0.9731→0.9570으로 하락했다. 에포크를 늘려도 train+val 단독보다 낮았는데, 소규모 데이터에서 3중 정규화가 과하고 Mixup이 스펙트로그램 주파수 구조를 파괴 한 것 같다.

**최종 성능 요약**:

| 모델                | Fusion          | 백본                       | Test Acc   | F1 macro   |
| ------------------- | --------------- | -------------------------- | ---------- | ---------- |
| Model 1             | Concatenation   | mBERT + EfficientNet-B0    | 0.8602     | 0.8606     |
| Model 2             | Cross-Attention | mBERT + EfficientNet-B0    | 0.8495     | 0.8505     |
| Model 3             | Gated           | mBERT + EfficientNet-B0    | 0.9140     | 0.9136     |
| Model 4             | Gated           | XLM-RoBERTa + ViT-B/16     | OOM        | -          |
| Model 5             | Optimized Gated | KLUE-BERT + ConvNeXt-Small | 0.9516     | 0.9518     |
| Model 5 (train+val) | Optimized Gated | KLUE-BERT + ConvNeXt-Small | **0.9731** | **0.9728** |
| Model 6             | Gated           | mDeBERTa-v3 + Swin-Tiny    | 0.9516     | 0.9520     |
| Model 6 (train+val) | Gated           | mDeBERTa-v3 + Swin-Tiny    | **0.9731** | **0.9732** |
