"""
MDER-MA 멀티모달 감정 인식 - 공통 모듈
데이터 로딩, 전처리, 학습 파이프라인, 시각화 등 공통 기능 제공
"""

import os, copy, gc, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import timm

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 상수
# ============================================================
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Neutral']
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for i, e in enumerate(EMOTIONS)}

DEFAULT_DATA_CONFIG = {
    'data_root': 'data',
    'spectrogram_type': 'spectrogram',       # 'spectrogram' | 'mel'
    'text_model_name': 'bert-base-multilingual-cased',
    'image_model_name': 'efficientnet_b0',
    'max_text_length': 128,
    'image_size': 224,
    'test_size': 0.15,
    'val_size': 0.15,
    'random_seed': 42,
    'num_classes': 4,
    'white_thresh': 250,                     # 흰색 판정 임계값 (0~255)
    'white_ratio': 0.05,                     # 이 비율 초과 시 제거
    'balance_classes': False,                # class_weights로 균형 처리 (언더샘플링 비활성)
}


# ============================================================
# 유틸리티
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    return device


def is_white_image(img_path, white_thresh=250, white_ratio=0.05):
    """
    이미지의 흰색 공백 비율 검사.
    전체 픽셀 중 RGB 각 채널이 모두 white_thresh 이상인 비율이
    white_ratio를 초과하면 True (= 제거 대상).
    """
    try:
        img = np.array(Image.open(img_path).convert('RGB'))
        white_mask = np.all(img >= white_thresh, axis=2)
        ratio = white_mask.sum() / white_mask.size
        return ratio > white_ratio
    except Exception:
        return True


# ============================================================
# 데이터 로딩 & 전처리
# ============================================================
def load_data(config=None):
    """
    폴더 스캔 → 파일명 기반 라벨 추출 → 텍스트-스펙트로그램 매칭 → 화이트 이미지 필터링.
    Returns: 깔끔한 DataFrame
    """
    if config is None:
        config = DEFAULT_DATA_CONFIG

    data_root = config['data_root']
    spec_folder = ('ERD-MA Mel-Spectrograms_'
                   if config['spectrogram_type'] == 'mel'
                   else 'ERD-MA Spectrogram')
    text_dir = os.path.join(data_root, 'ERD-MA Text')
    spec_dir = os.path.join(data_root, spec_folder)
    wt = config.get('white_thresh', 250)
    wr = config.get('white_ratio', 0.05)

    def _has_gender(stem):
        """파일명에 male/female이 포함되어야 유효 (fmale 오타 포함)"""
        low = stem.lower()
        return 'female' in low or 'male' in low or 'fmale' in low

    def _has_chunk(stem):
        """파일명에 chunk이 포함되면 비정상 데이터"""
        return 'chunk' in stem.lower()

    # 텍스트 파일 수집
    all_text = {}
    skip_no_gender = 0
    skip_chunk = 0
    for emotion in EMOTIONS:
        folder = os.path.join(text_dir, emotion)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith('.txt'):
                stem = os.path.splitext(f)[0]
                if _has_chunk(stem):
                    skip_chunk += 1
                    continue
                if not _has_gender(stem):
                    skip_no_gender += 1
                    continue
                path = os.path.join(folder, f)
                file_label = stem.split('_')[0]
                if stem not in all_text or file_label == emotion:
                    all_text[stem] = path

    # 스펙트로그램 파일 수집
    all_spec = {}
    for emotion in EMOTIONS:
        folder = os.path.join(spec_dir, emotion)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(('.jpg', '.png')):
                stem = os.path.splitext(f)[0]
                if _has_chunk(stem) or not _has_gender(stem):
                    continue
                path = os.path.join(folder, f)
                file_label = stem.split('_')[0]
                if stem not in all_spec or file_label == emotion:
                    all_spec[stem] = path

    print(f'필터링: 성별 미표기 {skip_no_gender}개, chunk 파일 {skip_chunk}개 제외')

    # 매칭
    common_stems = sorted(set(all_text.keys()) & set(all_spec.keys()))
    print(f'텍스트 파일: {len(all_text)}')
    print(f'스펙트로그램 파일: {len(all_spec)}')
    print(f'매칭된 쌍: {len(common_stems)}')

    # 화이트 이미지 필터링 + 빈 텍스트 제거
    white_count = 0
    empty_text_count = 0
    invalid_gender_count = 0
    samples = []
    for stem in tqdm(common_stems, desc='전처리 (화이트 이미지 필터링)'):
        parts = stem.split('_')
        label = parts[0]
        if label not in LABEL2ID:
            continue

        spec_path = all_spec[stem]
        if is_white_image(spec_path, wt, wr):
            white_count += 1
            continue

        with open(all_text[stem], 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not text:
            empty_text_count += 1
            continue

        gender = parts[1] if len(parts) > 1 else 'unknown'
        gender = gender.replace('fmale', 'female')  # 오타 보정

        # 최종 성별 검증 (male/female 아니면 제외)
        if gender not in ('male', 'female'):
            invalid_gender_count += 1
            continue

        samples.append({
            'id': stem,
            'text': text,
            'spec_path': spec_path,
            'label': label,
            'label_id': LABEL2ID[label],
            'gender': gender,
        })

    df = pd.DataFrame(samples)
    df['text_length'] = df['text'].str.len()

    print(f'\n--- 전처리 결과 ---')
    print(f'제거: 화이트 이미지 {white_count}개, 빈 텍스트 {empty_text_count}개, 성별 불명 {invalid_gender_count}개')
    print(f'필터링 후 데이터: {len(df)}개')
    print(df['label'].value_counts().reindex(EMOTIONS))

    # 클래스 균형: 최소 클래스 수에 맞춰 언더샘플링
    if config.get('balance_classes', True):
        min_count = df['label'].value_counts().min()
        balanced = []
        for label in EMOTIONS:
            label_df = df[df['label'] == label]
            if len(label_df) > min_count:
                label_df = label_df.sample(n=min_count, random_state=config['random_seed'])
            balanced.append(label_df)
        df = pd.concat(balanced).reset_index(drop=True)
        print(f'\n--- 클래스 균형 조정 (언더샘플링 → {min_count}개/클래스) ---')

    print(f'\n최종 데이터셋: {len(df)}개')
    print(f'\n--- Class Distribution ---')
    print(df['label'].value_counts().reindex(EMOTIONS))
    print(f'\n--- Gender Distribution ---')
    print(df['gender'].value_counts())

    return df


def visualize_data(df):
    """데이터 분석 시각화 (클래스분포, 텍스트길이, 성별분포, 샘플 스펙트로그램)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    # 감정 클래스 분포
    counts = df['label'].value_counts().reindex(EMOTIONS)
    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title('Emotion Class Distribution', fontsize=13)
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold')

    # 텍스트 길이 분포
    for i, em in enumerate(EMOTIONS):
        axes[1].hist(df[df['label'] == em]['text_length'],
                     alpha=0.6, label=em, bins=20, color=colors[i])
    axes[1].set_title('Text Length Distribution', fontsize=13)
    axes[1].set_xlabel('Characters')
    axes[1].legend()

    # 성별-감정 분포
    ct = pd.crosstab(df['gender'], df['label'])[EMOTIONS]
    ct.plot(kind='bar', ax=axes[2], color=colors)
    axes[2].set_title('Gender-Emotion Distribution', fontsize=13)
    axes[2].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.show()

    # 감정별 샘플 스펙트로그램
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for i, em in enumerate(EMOTIONS):
        row = df[df['label'] == em].iloc[0]
        axes[i].imshow(Image.open(row['spec_path']))
        preview = row['text'][:40] + ('...' if len(row['text']) > 40 else '')
        axes[i].set_title(f"{em}\n{preview}")
        axes[i].axis('off')
    plt.suptitle('Sample Spectrograms per Emotion', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ============================================================
# Dataset & Transforms
# ============================================================
class MultimodalEmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        enc = self.tokenizer(
            row['text'], max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt')
        img = Image.open(row['spec_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'image': img,
            'label': torch.tensor(row['label_id'], dtype=torch.long)
        }


def get_transforms(img_size=224):
    """학습용(augmentation 포함) / 검증용 이미지 변환"""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def prepare_datasets(df, config=None, device=None):
    """
    DataFrame → Stratified Split → Dataset 생성 + 클래스 가중치
    Returns: (train_ds, val_ds, test_ds, class_weights, tokenizer)
    """
    if config is None:
        config = DEFAULT_DATA_CONFIG
    if device is None:
        device = torch.device('cpu')

    seed = config['random_seed']

    # Stratified split
    train_df, temp_df = train_test_split(
        df, test_size=config['test_size'] + config['val_size'],
        stratify=df['label_id'], random_state=seed)
    rel_test = config['test_size'] / (config['test_size'] + config['val_size'])
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df['label_id'], random_state=seed)

    print(f'Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['text_model_name'])

    # Transforms
    train_tf, val_tf = get_transforms(config['image_size'])

    # Datasets
    train_ds = MultimodalEmotionDataset(train_df, tokenizer, config['max_text_length'], train_tf)
    val_ds = MultimodalEmotionDataset(val_df, tokenizer, config['max_text_length'], val_tf)
    test_ds = MultimodalEmotionDataset(test_df, tokenizer, config['max_text_length'], val_tf)

    # 클래스 가중치
    cnt = train_df['label_id'].value_counts().sort_index().values.astype(float)
    w = 1.0 / cnt
    w = w / w.sum() * len(EMOTIONS)
    class_weights = torch.FloatTensor(w).to(device)
    print(f'Class weights: {class_weights}')

    return train_ds, val_ds, test_ds, class_weights, tokenizer


# ============================================================
# 학습 & 평가 파이프라인
# ============================================================
def train_and_evaluate(model, train_ds, val_ds, test_ds,
                       train_config, class_weights, device):
    """
    모델 학습 + 검증 + 테스트.
    - Differential LR (encoder vs classifier)
    - Scheduler (cosine / plateau / warmup_cosine)
    - Early Stopping + Gradient Clipping
    Returns: dict{history, test_acc, test_f1_macro, test_f1_weighted, predictions, ...}
    """
    model = model.to(device)
    bs = train_config['batch_size']
    pin = (device.type == 'cuda')

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=pin)
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=pin)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential LR
    enc_params, cls_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'text_encoder' in name or 'image_encoder' in name:
            enc_params.append(p)
        else:
            cls_params.append(p)

    optimizer = optim.AdamW([
        {'params': enc_params, 'lr': train_config['encoder_lr']},
        {'params': cls_params, 'lr': train_config['classifier_lr']}
    ], weight_decay=train_config['weight_decay'])

    # Scheduler
    total_steps = len(train_loader) * train_config['num_epochs']
    stype = train_config['scheduler']

    if stype == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_config['num_epochs'])
        step_at = 'epoch'
    elif stype == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5)
        step_at = 'epoch'
    elif stype == 'warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_config.get('warmup_steps', 50),
            num_training_steps=total_steps)
        step_at = 'step'
    else:
        scheduler = None
        step_at = None

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state = None
    patience_cnt = 0

    for epoch in range(train_config['num_epochs']):
        # ---- Train ----
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{train_config['num_epochs']}")

        for batch in pbar:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            out = model(ids, mask, imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config['max_grad_norm'])
            optimizer.step()

            if step_at == 'step' and scheduler is not None:
                scheduler.step()

            t_loss += loss.item() * labels.size(0)
            pred = out.argmax(1)
            t_total += labels.size(0)
            t_correct += pred.eq(labels).sum().item()
            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                acc=f'{t_correct/t_total:.4f}')

        t_loss /= t_total
        t_acc = t_correct / t_total

        # ---- Validate ----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                imgs = batch['image'].to(device)
                labels = batch['label'].to(device)

                out = model(ids, mask, imgs)
                loss = criterion(out, labels)
                v_loss += loss.item() * labels.size(0)
                v_total += labels.size(0)
                v_correct += out.argmax(1).eq(labels).sum().item()

        v_loss /= v_total
        v_acc = v_correct / v_total

        # Scheduler step (epoch-level)
        if step_at == 'epoch' and scheduler is not None:
            if stype == 'plateau':
                scheduler.step(v_loss)
            else:
                scheduler.step()

        # Log
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        print(f'  => train_loss={t_loss:.4f} train_acc={t_acc:.4f} '
              f'val_loss={v_loss:.4f} val_acc={v_acc:.4f}')

        # Early Stopping
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= train_config['patience']:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Test ----
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

            out = model(ids, mask, imgs)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1_macro = f1_score(all_labels, all_preds, average='macro')
    test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print(f'\n{"="*50}')
    print(f'Best Val Acc:       {best_val_acc:.4f}')
    print(f'Test Accuracy:      {test_acc:.4f}')
    print(f'Test F1 (macro):    {test_f1_macro:.4f}')
    print(f'Test F1 (weighted): {test_f1_weighted:.4f}')
    print(f'{"="*50}')
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS))

    return {
        'history': history,
        'test_acc': test_acc,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted,
        'predictions': all_preds,
        'true_labels': all_labels,
        'best_val_acc': best_val_acc,
    }


# ============================================================
# 결과 시각화
# ============================================================
def plot_training_curves(result, model_name='Model'):
    """Loss & Accuracy 학습 곡선"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    h = result['history']
    epochs = range(1, len(h['train_loss']) + 1)

    ax1.plot(epochs, h['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, h['val_loss'], 'r--', label='Val', linewidth=2)
    ax1.set_title(f'{model_name} - Loss', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, h['train_acc'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, h['val_acc'], 'r--', label='Val', linewidth=2)
    ax2.set_title(f'{model_name} - Accuracy', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion(result, model_name='Model'):
    """Confusion Matrix"""
    cm = confusion_matrix(result['true_labels'], result['predictions'])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
    ax.set_title(
        f'{model_name}\nAcc={result["test_acc"]:.4f}  F1={result["test_f1_macro"]:.4f}',
        fontsize=13)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.show()


def print_report(result, model_name='Model'):
    """Classification Report 출력"""
    print(f'\n{"="*60}')
    print(f'  {model_name} - Classification Report')
    print(f'{"="*60}')
    print(classification_report(
        result['true_labels'], result['predictions'],
        target_names=EMOTIONS, digits=4))
