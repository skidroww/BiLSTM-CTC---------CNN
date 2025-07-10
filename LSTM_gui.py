import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder

# ==============================================================================
# 1. 설정 및 경로
# ==============================================================================
MODEL_PATH = 'C:/Users/bit/Desktop/best_sentence_model.pth'
VOCAB_PATH = 'C:/Users/bit/Desktop/vocab.json'

# ==============================================================================
# 2. 모델 클래스 정의 (학습 때와 동일)
# ==============================================================================
class SentenceLSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SentenceLSTM_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs.permute(1, 0, 2)

# ==============================================================================
# 3. 띄어쓰기 후처리 함수
# ==============================================================================
def add_spacing_post_process(sentence, vocab):
    """
    예측된 문장과 어휘 사전을 기반으로 띄어쓰기를 추가하는 후처리 함수
    """
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    result = []
    while len(sentence) > 0:
        found = False
        for word in sorted_vocab:
            if sentence.startswith(word):
                result.append(word)
                sentence = sentence[len(word):]
                found = True
                break
        if not found:
            result.append(sentence[0])
            sentence = sentence[1:]
    return " ".join(result)

# ==============================================================================
# 4. GUI 애플리케이션 클래스
# ==============================================================================
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # --- 모델 및 디코더 로드 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_vocab_and_model()
        self.init_decoder()

        # --- MediaPipe 초기화 ---
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(
            model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # --- 비디오 및 GUI 설정 ---
        # self.vid = cv2.VideoCapture(0) # 웹캠 사용시
        self.vid = cv2.VideoCapture('http://192.168.0.136:4747/video') # DroidCam 사용시
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # --- 상태 표시 레이블 ---
        self.status_label = tk.Label(window, text="상태: 대기 중", font=("Arial", 16))
        self.status_label.pack(anchor=tk.CENTER, expand=True)
        self.sentence_label = tk.Label(window, text="인식된 문장:", font=("Arial", 20), wraplength=600)
        self.sentence_label.pack(anchor=tk.CENTER, expand=True)

        # --- 추론 관련 변수 ---
        self.keypoints_buffer = []
        self.is_signing = False
        self.idle_counter = 0
        self.IDLE_THRESHOLD = 0.5
        self.IDLE_TIME_THRESHOLD = 30
        self.previous_features = None
        self.previous_velocity = None
        
        # ▼▼▼ 안정화 필터 변수 추가 ▼▼▼
        self.prediction_history = []
        self.STABLE_THRESHOLD = 3  # 3번 연속 일치해야 최종 예측으로 인정
        self.stable_prediction = "..." # 화면에 표시될 안정화된 최종 예측
        
        self.delay = 15
        self.update()
        self.window.mainloop()

    def load_vocab_and_model(self):
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            self.word_to_index = json.load(f)
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        print("어휘 사전 로드 완료.")

        INPUT_DIM = 498
        HIDDEN_DIM = 256
        NUM_LAYERS = 2
        VOCAB_SIZE = len(self.word_to_index)
        
        self.model = SentenceLSTM_CNN(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, VOCAB_SIZE).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        print(f"모델 로드 완료. Device: {self.device}")

    def init_decoder(self):
        labels = [word for word in self.index_to_word.values() if word not in ['<pad>', '<blank>']]
        self.beam_search_decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=None,
            alpha=0,
            beta=0.6
        )
        print("Beam Search Decoder 생성 완료.")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results_pose = self.pose.process(image_rgb)
            results_hands = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True
            
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

                left_hand_features = np.zeros(75)
                right_hand_features = np.zeros(75)

                if results_hands.multi_hand_landmarks:
                    for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                        handedness = results_hands.multi_handedness[i].classification[0].label
                        joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        v = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :] - joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                        angle = np.arccos(np.einsum('nt,nt->n', v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
                        angle = np.degrees(angle)
                        hand_features = np.concatenate([v.flatten(), angle])
                        if handedness == "Left": left_hand_features = hand_features
                        elif handedness == "Right": right_hand_features = hand_features
                
                final_features = np.concatenate([pose_features, left_hand_features, right_hand_features])
                velocity = np.zeros_like(final_features) if self.previous_features is None else final_features - self.previous_features
                movement = np.linalg.norm(velocity)
                acceleration = np.zeros_like(velocity) if self.previous_velocity is None else velocity - self.previous_velocity
                combined_features = np.concatenate([final_features, velocity, acceleration])

                if movement > self.IDLE_THRESHOLD:
                    if not self.is_signing:
                        self.is_signing = True
                        self.keypoints_buffer = []
                        self.status_label.config(text="상태: 인식 중...")
                    self.keypoints_buffer.append(combined_features)
                    self.idle_counter = 0
                else:
                    if self.is_signing:
                        self.idle_counter += 1
                        if self.idle_counter > self.IDLE_TIME_THRESHOLD:
                            self.is_signing = False
                            self.status_label.config(text="상태: 처리 중...")
                            
                            if len(self.keypoints_buffer) > 10:
                                input_tensor = torch.FloatTensor(np.array(self.keypoints_buffer)).unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    log_probs = self.model(input_tensor)
                                
                                probs = F.softmax(log_probs.squeeze(1).cpu(), dim=-1)
                                word_probs = probs[:, 2:]
                                blank_prob = probs[:, 1].unsqueeze(1)
                                probs_for_decoder = torch.cat([word_probs, blank_prob], dim=1)
                                
                                predicted_sentence = self.beam_search_decoder.decode(probs_for_decoder.numpy())
                                
                                 # ▼▼▼ 안정화 필터 로직 적용 ▼▼▼
                                self.prediction_history.append(predicted_sentence)
                                if len(self.prediction_history) > self.STABLE_THRESHOLD:
                                    self.prediction_history.pop(0) # 가장 오래된 예측 제거
                                
                                # 띄어쓰기 후처리 적용
                                vocab_for_spacing = [word for word in self.index_to_word.values() if word not in ['<pad>', '<blank>']]
                                final_sentence = add_spacing_post_process(predicted_sentence, vocab_for_spacing)
                                
                                self.sentence_label.config(text=f"인식된 문장: {final_sentence}")

                            self.keypoints_buffer = []
                            self.status_label.config(text="상태: 대기 중")
                
                self.previous_features = final_features
                self.previous_velocity = velocity

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def on_closing(self):
        self.vid.release()
        self.window.destroy()

# ==============================================================================
# 5. 애플리케이션 실행
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "수어 문장 인식 GUI")
