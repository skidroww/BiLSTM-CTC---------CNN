import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from utils import Vector_Normalization

# --- 설정값 ---
BASE_DIR = 'C:/Users/bit/Desktop/sign_language_sen_data'
CSV_PATH = os.path.join(BASE_DIR, 'labels.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '좌표')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ⭐ 처리할 동영상 확장자 목록 (필요시 추가)
VIDEO_EXTENSIONS = ['.mov', '.mp4', '.avi', '.mkv','mts']

# --- MediaPipe 초기화 ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

FEATURE_DIM = 166

# --- 데이터 처리 시작 ---
print("--- 문장 데이터 특징 추출을 시작합니다 ---")

try:
    df = pd.read_csv(CSV_PATH, delimiter=',')
except FileNotFoundError:
    print(f"에러: {CSV_PATH} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# CSV의 각 행(각 폴더)을 순회
for index, row in df.iterrows():
    folder_path_str = row['folderpath']
    full_folder_path = os.path.join(BASE_DIR, folder_path_str)

    if not os.path.isdir(full_folder_path):
        print(f"경고: '{full_folder_path}' 폴더를 찾을 수 없습니다. 건너뜁니다.")
        continue

    # ⭐ 폴더 내의 모든 파일을 확인
    video_files_in_folder = [f for f in os.listdir(full_folder_path) if any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)]
    
    if not video_files_in_folder:
        print(f"경고: '{full_folder_path}' 폴더에 처리할 영상 파일이 없습니다.")
        continue

    # ⭐ 폴더 내의 각 영상 파일을 순회하며 처리
    for video_index, video_filename in enumerate(tqdm(video_files_in_folder, desc=f"'{folder_path_str}' 폴더 처리 중")):
        video_path = os.path.join(full_folder_path, video_filename)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"경고: '{video_path}' 비디오를 열 수 없습니다.")
            continue

        keypoints_buffer = []
        previous_features = None
        previous_velocity = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- 특징 추출 로직 (이전과 동일) ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results_pose = pose.process(image_rgb)
            results_hands = hands.process(image_rgb)
            final_features = np.zeros(FEATURE_DIM)

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
                        v, angle_label = Vector_Normalization(joint)
                        hand_features = np.concatenate([v.flatten(), angle_label.flatten()])
                        if handedness == "Left": left_hand_features = hand_features
                        elif handedness == "Right": right_hand_features = hand_features
                final_features = np.concatenate([pose_features, left_hand_features, right_hand_features])

            velocity = np.zeros_like(final_features) if previous_features is None else final_features - previous_features
            acceleration = np.zeros_like(velocity) if previous_velocity is None else velocity - previous_velocity
            combined_features = np.concatenate([final_features, velocity, acceleration])
            keypoints_buffer.append(combined_features)
            previous_features = final_features
            previous_velocity = velocity
        
        cap.release()

        # ⭐ .npy 파일 저장 (예: 화상을입었어요_0.npy, 화상을입었어요_1.npy)
        if keypoints_buffer:
            concept_name = folder_path_str
            npy_filename = f"{concept_name}_{video_index}.npy"
            save_path = os.path.join(OUTPUT_DIR, npy_filename)
            np.save(save_path, np.array(keypoints_buffer))

print("\n--- 모든 비디오 처리 및 특징 추출 완료 ---")