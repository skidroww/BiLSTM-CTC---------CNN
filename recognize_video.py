import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from PIL import ImageFont, ImageDraw, Image
from utils import Vector_Normalization
from pyctcdecode import build_ctcdecoder

# --- 1. CNN+LSTM 모델 클래스 정의 (학습 때와 동일) ---
class SentenceLSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SentenceLSTM_CNN, self).__init__()
        
        # 1D CNN 레이어
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        # LSTM의 입력 사이즈는 CNN의 최종 out_channels와 동일
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x shape: (N, T, C) -> (N, C, T) for Conv1d
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

# --- 2. 설정 및 로드 ---
DESKTOP_PATH = "C:/Users/bit/Desktop"
MODEL_PATH = os.path.join(DESKTOP_PATH, "best_sentence_model.pth")
VOCAB_PATH = os.path.join(DESKTOP_PATH, "vocab.json")

# ▼▼▼▼▼ 인식하고 싶은 동영상 파일의 경로를 여기에 입력하세요 ▼▼▼▼▼
#VIDEO_FILE_PATH = "C:/Users/bit/Desktop/sign_language_sen_data/음식을 하다가 손가락을 칼에 베였어요/KETI_SL_0000009215.MTS"
VIDEO_FILE_PATH = "C:/Users/bit/Desktop/KakaoTalk_20250709_161131809.mp4"
# ▲▲▲▲▲ 인식하고 싶은 동영상 파일의 경로를 여기에 입력하세요 ▲▲▲▲▲

# 어휘 사전 로드
try:
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        word_to_index = json.load(f)
    index_to_word = {i: word for word, i in word_to_index.items()}
    VOCAB_SIZE = len(word_to_index)
    print("어휘 사전 로드 완료.")
except FileNotFoundError:
    print(f"에러: {VOCAB_PATH} 파일을 찾을 수 없습니다.")
    exit()

# 모델 로드
INPUT_DIM = 498
HIDDEN_DIM = 256
NUM_LAYERS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceLSTM_CNN(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, VOCAB_SIZE).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"모델 로드 완료. Device: {device}")
except FileNotFoundError:
    print(f"에러: {MODEL_PATH} 파일을 찾을 수 없습니다.")
    exit()



# --- 3. Beam Search Decoder 생성 ---
# 어휘 사전에서 특수 토큰(<pad>, <blank>)은 제외하고 레이블 목록 생성
labels = [word for word in index_to_word.values() if word not in ['<pad>', '<blank>']]

# 디코더 빌드 (alpha, beta 등 파라미터는 그대로 둡니다)
beam_search_decoder = build_ctcdecoder(
    labels,
    kenlm_model_path=None,
    alpha=0,
    beta=0.6
)
print("Beam Search Decoder 생성 완료.")



# --- 4. 영상 처리 및 추론 ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO_FILE_PATH)
if not cap.isOpened():
    print(f"에러: '{VIDEO_FILE_PATH}' 동영상을 열 수 없습니다.")
    exit()
    

    


keypoints_buffer = []
previous_features = None
previous_velocity = None # ◀◀◀ [추가] 이전 프레임의 속도를 저장할 변수


print("\n--- 영상 처리를 시작합니다 ---")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 특징 추출 로직 (GUI와 동일)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)
    results_hands = hands.process(image_rgb)
    
    final_features = np.zeros(166)
    if results_pose.pose_landmarks:
        pose_lm = results_pose.pose_landmarks.landmark
        shoulder_center_x = (pose_lm[11].x + pose_lm[12].x) / 2
        shoulder_center_y = (pose_lm[11].y + pose_lm[12].y) / 2
        shoulder_width = np.linalg.norm([pose_lm[11].x - pose_lm[12].x, pose_lm[11].y - pose_lm[12].y]) + 1e-6
        pose_indices = [11, 12, 13, 14, 15, 16, 23, 24]
        pose_features = []
        for idx in pose_indices:
            pose_features.append((pose_lm[idx].x - shoulder_center_x) / shoulder_width)
            pose_features.append((pose_lm[idx].y - shoulder_center_y) / shoulder_width)
        left_hand_features, right_hand_features = np.zeros(75), np.zeros(75)
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                handedness = results_hands.multi_handedness[i].classification[0].label
                joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                v, angle_label = Vector_Normalization(joint)
                hand_features = np.concatenate([v.flatten(), angle_label.flatten()])
                if handedness == "Left": left_hand_features = hand_features
                elif handedness == "Right": right_hand_features = hand_features
        final_features = np.concatenate([pose_features, left_hand_features, right_hand_features])
    
    velocity = np.zeros_like(final_features) if previous_features is None else final_features - previous_features
    acceleration = np.zeros_like(velocity) if previous_velocity is None else velocity - previous_velocity # ◀◀◀ [추가] 가속도 계산

    combined_features = np.concatenate([final_features, velocity, acceleration]) # ◀◀◀ [수정] 계산된 가속도 사용
    keypoints_buffer.append(combined_features)
    
    # 현재 상태를 이전 상태로 업데이트
    previous_features = final_features
    previous_velocity = velocity # ◀◀◀ [추가] 현재 속도를 다음 프레임을 위해 저장


    # (선택) 처리 중인 영상 보기
    cv2.imshow('Processing Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("--- 영상 처리 완료 ---")
cap.release()
cv2.destroyAllWindows()

def add_spacing_post_process(sentence, vocab):
    """
    예측된 문장과 어휘 사전을 기반으로 띄어쓰기를 추가하는 후처리 함수
    """
    # 어휘 사전을 길이순으로 내림차순 정렬 (가장 긴 단어부터 찾기 위함)
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    
    result = []
    while len(sentence) > 0:
        # 현재 문장의 시작 부분과 일치하는 가장 긴 단어를 찾음
        found = False
        for word in sorted_vocab:
            if sentence.startswith(word):
                result.append(word)
                sentence = sentence[len(word):] # 찾은 단어만큼 문자열을 잘라냄
                found = True
                break
        # 어휘 사전에서 단어를 찾지 못한 경우 (예: 알 수 없는 문자)
        if not found:
            # 한 글자씩 잘라서 추가하고 넘어감
            result.append(sentence[0])
            sentence = sentence[1:]
            
    return " ".join(result)


# --- 5. 최종 예측 ---
if keypoints_buffer:
    input_tensor = torch.FloatTensor(np.array(keypoints_buffer)).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(input_tensor)

    # CPU로 이동 후 softmax를 적용하여 확률값으로 변환
    probs = F.softmax(log_probs.squeeze(1).cpu(), dim=-1)

    # 디코더 입력을 위한 확률값 구조 변경
    word_probs = probs[:, 2:]
    blank_prob = probs[:, 1].unsqueeze(1)
    probs_for_decoder = torch.cat([word_probs, blank_prob], dim=1)
    
    # ◀◀◀ .numpy()를 추가하여 텐서를 배열로 변환 (핵심 수정) ◀◀◀
    predicted_sentence = beam_search_decoder.decode(probs_for_decoder.numpy())

    vocab_for_spacing = [word for word in index_to_word.values() if word not in ['<pad>', '<blank>']]
    final_sentence = add_spacing_post_process(predicted_sentence, vocab_for_spacing)
    
    
    print("\n========================================")
    print(f"최종 예측 문장: {final_sentence}")
    print("========================================")
else:
    print("영상에서 특징을 추출하지 못했습니다.")