import numpy as np

# --- ステップ1: 6種類の「お手本」を定義 ---
# TODO:
# 1. あなたの「お手本」となる6感情のVA座標と、その時の9個のパラメータを定義してください。
# 2. ここではダミーとして「喜び・怒り・悲しみ・落ち着き・驚き・恐怖」の6つを仮定します。

# (eyeOpen, pupil, pupilAngle, upperAngle, upperCover, lowerCover, mouthCurve, mouthHeight, mouthWidth)
KEYFRAME_PARAMS = {
    # お手本1: 喜び (Happy)
    "happy": np.array([0.25, 0.65, -10, -20, 0, 0.2, 40, 3.0, 4.0]),
    
    # お手本2: 怒り (Angry)
    "angry": np.array([0.9, 0.8, 5, 20, 0.15, 0.2, -15, 0.3, 0.9]),
    
    # お手本3: 悲しみ (Sad)
    "sad": np.array([0.8, 0.6, -5, -15, 0.18, 0.15, -18, 0.1, 0.8]),
    
    # お手本4: 落ち着き (Calm)
    "calm": np.array([0.15, 0.7, -13, -26, 0, 0, 12, 0.3, 1.2]),
    
    # お手本5: 驚き (Surprise)
    "surprise": np.array([1, 0.4, 10, 25, 0.0, 0.0, 15, 3, 0.65]),
    
    # お手本6: 恐怖 (Fear)
    "fear": np.array([1, 0.55, 0, -29, 0.07, 0, -11, 2.05, 0.8])
}

# 上記の6感情に対応する「VA座標」
# TODO: この座標も、あなたの研究の定義に合わせて書き換えてください。
KEYFRAME_VA = {
    "happy": np.array([4.5, 3.5]),    # (V, A)
    "angry": np.array([-4.0, 4.0]),
    "sad":   np.array([-4.0, -3.0]),
    "calm":  np.array([4.0, -3.0]),
    "surprise": np.array([1.0, 4.5]),
    "fear":  np.array([-2.5, 3.5])
}

# 補間のなめらかさを決めるパワー（通常 2.0 でOK）
IDW_POWER = 2.0
# 距離がゼロ（お手本と完全一致）の時のための微小値
EPSILON = 1e-9

def get_interpolated_expression(target_v, target_a):
    """
    ターゲットのVA座標に基づき、6つのキーフレームから表情パラメータを補間します。
    (逆距離加重法 - IDW)
    """
    
    target_va = np.array([target_v, target_a])
    
    total_weight = 0.0
    weighted_params = np.zeros(9) # 9個のパラメータ（中身は全部0）
    
    for emotion_name in KEYFRAME_VA:
        
        # 1. ターゲットとお手本のVA座標との「距離」を計算
        key_va = KEYFRAME_VA[emotion_name]
        distance = np.linalg.norm(target_va - key_va) # ユークリッド距離
        
        # ゼロ除算を避ける
        if distance < EPSILON:
            # 距離がほぼゼロ ＝ お手本と完全一致
            return KEYFRAME_PARAMS[emotion_name]
            
        # 2. 距離の逆数から「重み」を計算
        # 距離が近いほど、重みは爆発的に大きくなる
        weight = 1.0 / (distance ** IDW_POWER)
        
        # 3. 重みを加算していく
        total_weight += weight
        weighted_params += weight * KEYFRAME_PARAMS[emotion_name]
        
    # 4. 全ての重み付きパラメータを、重みの合計で割って「加重平均」を出す
    final_params = weighted_params / total_weight

    # 目の開き具合を強制的に変更
    if final_params[0] >= 0.35:
        final_params[0] = 1.0
    else:
        final_params[0] = 0.2
    
    return final_params

# --- ステップ3: 実行例 ---
# LLMが「少し不安」(V=-1.0, A=1.5) と判断したと仮定
target_valence = 3
target_arousal = 4

new_face_params = get_interpolated_expression(target_valence, target_arousal)

print(f"ターゲット座標: V={target_valence}, A={target_arousal}")
print("--- 生成された9個のパラメータ ---")
print(f"eyeOpenness: {new_face_params[0]:.4f}")
print(f"pupilSize: {new_face_params[1]:.4f}")
print(f"pupilAngle: {new_face_params[2]:.4f}")
print(f"upperEyelidAngle: {new_face_params[3]:.4f}")
print(f"upperEyelidCoverage: {new_face_params[4]:.4f}")
print(f"lowerEyelidCoverage: {new_face_params[5]:.4f}")
print(f"mouthCurve: {new_face_params[6]:.4f}")
print(f"mouthHeight: {new_face_params[7]:.4f}")
print(f"mouthWidth: {new_face_params[8]:.4f}")