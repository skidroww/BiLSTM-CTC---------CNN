import numpy as np

# 3D 벡터 정규화 버전
def Vector_Normalization(joint):
    # Compute angles between joints
    # ⭐ :2 슬라이싱을 제거하여 x, y, z 좌표를 모두 사용
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :] # Child joint
    v = v2 - v1 
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    # ⭐ 여기는 3D 벡터에 대해서도 동일하게 작동합니다.
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

    angle = np.degrees(angle) # Convert radian to degree

    angle_label = np.array([angle], dtype=np.float32)

    return v, angle_label