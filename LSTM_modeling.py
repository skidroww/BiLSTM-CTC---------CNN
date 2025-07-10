import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==============================================================================
# 1. 설정 및 하이퍼파라미터
# ==============================================================================
BASE_DIR = 'C:/Users/bit/Desktop/sign_language_sen_data'
CSV_PATH = os.path.join(BASE_DIR, 'labels.csv')
NPY_DIR = os.path.join(BASE_DIR, '좌표')
MODEL_SAVE_PATH = 'C:/Users/bit/Desktop/best_sentence_model.pth' # 가장 성능이 좋은 모델이 저장될 경로
VOCAB_SAVE_PATH = 'C:/Users/bit/Desktop/vocab.json'

# 모델 하이퍼파라미터
INPUT_DIM = 498
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 8  # GPU 메모리에 따라 조절
NUM_EPOCHS = 500
LEARNING_RATE = 0.0001
NOISE_LEVEL = 0.005 # 데이터 증강 노이즈 수준

# 조기 종료(Early Stopping) 설정
PATIENCE = 25 # 15 에포크 동안 검증 손실이 개선되지 않으면 학습 중단

# ==============================================================================
# 2. 어휘 사전 생성
# ==============================================================================
df = pd.read_csv(CSV_PATH, delimiter=',')
all_words = set(word for sentence in df['transcript'] for word in sentence.split(' '))

word_list = sorted(list(all_words))
special_tokens = ['<pad>', '<blank>']
word_list = special_tokens + word_list
word_to_index = {word: i for i, word in enumerate(word_list)}
index_to_word = {i: word for i, word in enumerate(word_list)}
VOCAB_SIZE = len(word_list)

print(f"어휘 사전 생성 완료. 단어 수: {VOCAB_SIZE}")

# ==============================================================================
# 3. 데이터 목록 생성 및 분할
# ==============================================================================
data_manifest = []
for index, row in df.iterrows():
    folder_name = row['folderpath'].replace('.영상', '')
    transcript = row['transcript']
    npy_files = [f for f in os.listdir(NPY_DIR) if f.startswith(folder_name) and f.endswith('.npy')]
    for npy_file in npy_files:
        npy_path = os.path.join(NPY_DIR, npy_file)
        data_manifest.append({'npy_path': npy_path, 'transcript': transcript})

# 데이터를 훈련용과 검증용으로 분할 (8:2 비율)
train_manifest, val_manifest = train_test_split(data_manifest, test_size=0.2, random_state=42)
print(f"데이터 분할 완료. 훈련 데이터: {len(train_manifest)}개, 검증 데이터: {len(val_manifest)}개")

# ==============================================================================
# ✨ 3-1. 데이터 증강 함수 정의
# ==============================================================================
def temporal_warp(sequence, warp_factor_range=(0.8, 1.2)):
    """시간적 왜곡: 시퀀스의 길이를 무작위로 줄이거나 늘립니다."""
    original_len, num_features = sequence.shape
    warp_factor = np.random.uniform(warp_factor_range[0], warp_factor_range[1])
    new_len = int(original_len * warp_factor)

    warped_sequence = np.zeros((new_len, num_features))
    x = np.arange(original_len)
    new_x = np.linspace(0, original_len - 1, new_len)

    for i in range(num_features):
        interp_func = np.interp(new_x, x, sequence[:, i])
        warped_sequence[:, i] = interp_func
    return warped_sequence

def masking(sequence, mask_type='time', max_length_ratio=0.15, num_masks=1):
    """마스킹: 시퀀스의 일부를 0으로 만들어 정보를 가립니다."""
    seq_len, num_features = sequence.shape
    
    if mask_type == 'time':
        # 시간 축을 따라 마스킹
        max_mask_len = int(seq_len * max_length_ratio)
        if max_mask_len == 0: return sequence
        for _ in range(num_masks):
            mask_len = np.random.randint(1, max_mask_len)
            start = np.random.randint(0, seq_len - mask_len)
            sequence[start:start+mask_len, :] = 0
            
    elif mask_type == 'feature':
        # 특징 축을 따라 마스킹 (특정 관절 정보를 전체 시간 동안 제거)
        num_feature_mask = int(num_features * max_length_ratio)
        if num_feature_mask == 0: return sequence
        for _ in range(num_masks):
            masked_features = np.random.choice(num_features, num_feature_mask, replace=False)
            sequence[:, masked_features] = 0
            
    return sequence

def scale_translate(sequence, scale_range=(0.9, 1.1), translate_range=(-0.05, 0.05)):
    """크기/위치 조절: 좌표를 미세하게 확대/축소하거나 이동시킵니다."""
    # 위치 정보(166개)에만 적용. 속도/가속도 정보는 제외
    num_pos_features = 166 
    
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    translate_factor = np.random.uniform(translate_range[0], translate_range[1], num_pos_features)
    
    sequence[:, :num_pos_features] = sequence[:, :num_pos_features] * scale_factor + translate_factor
    return sequence

# ==============================================================================
# 4. Dataset 클래스 (✨ 데이터 증강 기능 추가)
# ==============================================================================
class SentenceDataset(Dataset):
    def __init__(self, manifest, word_to_index_map, is_train=False, noise_level=0.0):
        self.manifest = manifest
        self.word_to_index = word_to_index_map
        self.is_train = is_train
        self.noise_level = noise_level

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        sample = self.manifest[idx]
        npy_path = sample['npy_path']
        transcript = sample['transcript']
        sequence_data = np.load(npy_path).astype(np.float32)

    # 훈련용 데이터에만 데이터 증강 적용
        if self.is_train:
        # 확률을 0.3으로 낮추고, 증강 강도도 약하게 조절
            #if np.random.rand() < 0.5:
            #    sequence_data = scale_translate(sequence_data, scale_range=(0.95, 1.1))
        
            if np.random.rand() < 0.5:
                sequence_data = temporal_warp(sequence_data, warp_factor_range=(0.8, 1.2))

            if np.random.rand() < 0.5:
                sequence_data = masking(sequence_data, mask_type='time', max_length_ratio=0.15)
        
        # 기존의 노이즈 추가
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, sequence_data.shape).astype(np.float32)
                sequence_data += noise

    # ◀◀◀ 이 부분의 들여쓰기를 수정해야 합니다.
    # 이 로직은 is_train 값과 상관없이 항상 실행되어야 합니다.
        label_sequence = [self.word_to_index[word] for word in transcript.split(' ')]
        return torch.FloatTensor(sequence_data), torch.LongTensor(label_sequence)

# ==============================================================================
# 5. EarlyStopping 클래스
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ==============================================================================
# 6. 모델 및 Collate 함수 정의
# ==============================================================================
class SentenceLSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SentenceLSTM_CNN, self).__init__()
        
        # 1D CNN 레이어 추가
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        # LSTM의 입력 사이즈는 CNN의 최종 out_channels와 동일해야 함
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x shape: (N, T, C) -> (N, C, T) for Conv1d
        # N: 배치 크기, T: 시퀀스 길이, C: 특징 차원
        x = x.permute(0, 2, 1)
        
        # CNN 통과
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # LSTM 입력을 위해 다시 (N, T, C) 형태로 변경
        x = x.permute(0, 2, 1)
        
        # LSTM 통과
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs.permute(1, 0, 2) # (T, N, C) for CTC Loss

def pad_collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=word_to_index['<pad>'])
    sequence_lengths = torch.LongTensor([len(seq) for seq in sequences])
    label_lengths = torch.LongTensor([len(lab) for lab in labels])
    return padded_sequences, padded_labels, sequence_lengths, label_lengths

# ==============================================================================
# 7. 메인 학습 로직
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 및 데이터로더 생성
    train_dataset = SentenceDataset(train_manifest, word_to_index, is_train=True, noise_level=NOISE_LEVEL)
    val_dataset = SentenceDataset(val_manifest, word_to_index, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    # 모델, 손실 함수, 옵티마이저, 조기 종료 초기화
    model = SentenceLSTM_CNN(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, VOCAB_SIZE).to(device)
    criterion = nn.CTCLoss(blank=word_to_index['<blank>'], reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    
    # ◀◀◀ [추가] 학습률 스케줄러 설정 ◀◀◀
    # 검증 손실(val_loss)이 5 에포크 동안 개선되지 않으면 학습률을 0.1배로 줄임
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=MODEL_SAVE_PATH)
    
    

    print("\n--- 모델 학습을 시작합니다 ---")
    for epoch in range(NUM_EPOCHS):
        # 훈련
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for sequences, labels, seq_lengths, label_lengths in train_pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            
            log_probs = model(sequences)
            loss = criterion(log_probs, labels, seq_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)

        # 검증
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for sequences, labels, seq_lengths, label_lengths in val_pbar:
                sequences, labels = sequences.to(device), labels.to(device)
                
                log_probs = model(sequences)
                loss = criterion(log_probs, labels, seq_lengths, label_lengths)
                
                val_loss += loss.item()
                val_pbar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        # 조기 종료 확인
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("조기 종료 조건이 충족되어 학습을 중단합니다.")
            break

    print("\n--- 모델 학습 완료 ---")
    print(f"가장 성능이 좋았던 모델이 '{MODEL_SAVE_PATH}' 경로에 저장되었습니다.")
    
    # 어휘 사전을 json 파일로 저장
    with open(VOCAB_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(word_to_index, f, ensure_ascii=False, indent=4)
    print(f"어휘 사전이 '{VOCAB_SAVE_PATH}' 경로에 저장되었습니다.")