import os
import re
import time
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import ollama
import csv
import math

# --- 定数 ---
# LLMモデル
LLM_MODEL = "qwen3:8b"
# CSVファイルパス
CSV_FILE_PATH = 'emotion_data.csv'
# CSVヘッダー
CSV_HEADERS = ['subject_id', 'timestamp', 'emotion_label', 'animationDuration', 'eyeOpenness', 'pupilSize', 'pupilAngle', 'upperEyelidAngle', 'upperEyelidCoverage', 'lowerEyelidCoverage', 'mouthCurve', 'mouthHeight', 'mouthWidth']


# --- FlaskとSocket.IOの初期化 ---
# 静的ファイルとテンプレートフォルダをルートディレクトリに設定
app = Flask(__name__, static_folder='static', template_folder='.')
app.config["SECRET_KEY"] = "C0HThSwr"
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Ollamaの初期化 ---
client = ollama.Client()

# --- 表情計算ロジック ---

# 論文  に記載されている式(2)の w の値
W = 1.0
# 論文  の式(2)にある ε (イプシロン) の値（ゼロ除算防止用）
EPSILON = 1e-9 

# --- ご注意 ------------------------------------------------------------------
# まだ確実な数値ではない
# ---------------------------------------------------------------------------
KEYFRAME_PARAMS = {
    "happy": np.array([1, 0.8, 0, 0, 0, 0.06, 30, 1.45, 2.5]),
    "angry": np.array([1, 0.65, 4, 30, 0.22, 0.16, -15, 0.65, 1.4]),
    "sad": np.array([1, 0.85, 0, -18, 0.14, 0.15, -18, 0.7, 1.6]),
    "astonished": np.array([1, 0.65, 0, -18, 0, 0, 12, 3, 1.15]),
    "sleepy": np.array([0.2, 0.75, -18, 0, 0, 0, 2, 2.5, 1.2]),
    "relaxed": np.array([0.2, 0.75, -14, 0, 0, 0.1, 14, 0.45, 1.3])
}

# 6感情に対応する「VA座標」
KEYFRAME_VA = {
    "happy": np.array([0.89,0.17]), 
    "angry": np.array([-0.40,0.79]),
    "sad": np.array([-0.81,-0.40]),
    "astonished": np.array([0.0, 1.0]), 
    "sleepy": np.array([0.01, -1.0]),
    "relaxed": np.array([0.71,-0.65])
}

# def get_interpolated_expression(target_v, target_a):
#     """ ターゲットのVA座標に基づき、表情パラメータを補間する """
#     target_va = np.array([target_v, target_a])
#     total_weight = 0.0
#     weighted_params = np.zeros(9) # パラメータの数（元のコードに依存）
    
#     for emotion_name, key_va in KEYFRAME_VA.items():
#         distance = np.linalg.norm(target_va - key_va)
        
#         # 論文の式(2)  に基づき、rtop_k (weight) を計算
#         # rtop_k = 1 / ((100 * d_k)^w + ε)
#         # weight = np.exp(- (0.5*distance ** 2)) 
#         weight = 1.0 / (((100*distance) ** W) + EPSILON)
        
#         total_weight += weight
#         weighted_params += weight * KEYFRAME_PARAMS[emotion_name]
    
#     final_params = weighted_params / total_weight
        
        
        
def get_interpolated_expression(target_v, target_a):
    """ターゲットのVA座標に基づき、表情パラメータを補間する"""
    target_va = np.array([target_v, target_a])

    rtop_values = []  # 各emotionの生の重み
    params_list = []
    emotion_names = []
    

    for emotion_name, key_va in KEYFRAME_VA.items():
        distance = np.linalg.norm(target_va - key_va)
        # rtop_k = 1 / ((distance)^W + ε)
        rtop_k = 1.0 / (((distance)) + EPSILON)
        #rtop_k = np.exp(- (distance ** 2))  # 距離が大きいほど小さくなるように指数関数で変換
        rtop_values.append(rtop_k)
        params_list.append(KEYFRAME_PARAMS[emotion_name])
        emotion_names.append(emotion_name)

    # ===== ソフトマックス正規化 =====
    rtop_values = np.array(rtop_values)
    # exp_rtop = np.exp(rtop_values - np.max(rtop_values))  # 安定化
    # softmax_weights = exp_rtop / np.sum(exp_rtop)
    T = 1  # 温度パラメータ
    # 安定化 & 温度スケーリング
    exp_rtop = np.exp((rtop_values - np.max(rtop_values)) / T)
    softmax_weights = exp_rtop / np.sum(exp_rtop)

    print(f"\n=== V={target_v}, A={target_a} の重み分析 ===")
    print(f"{'感情':<10} | {'距離':<8} | {'生の重み':<11} | {'ソフトマックス %':<15}")
    print("-" * 65)
    for i, emotion_name in enumerate(emotion_names):
        distance = np.linalg.norm(target_va - KEYFRAME_VA[emotion_name])
        print(f"{emotion_name:<12} | {distance:<10.4f} | {rtop_values[i]:<15.6f} | {softmax_weights[i]*100:<15.2f}%")
    print("=" * 65 + "\n")
    
#     # ===== 重み付き平均 =====
#     final_params = np.zeros(9)
#     for w, p in zip(softmax_weights, params_list):
#         final_params += w * p

    # ===== 重み付き平均 (ファジーロジック適用) =====

    # --- 1. eyeOpenness 以外の8パラメータを計算 (従来通りの加重平均) ---
    final_params_8 = np.zeros(8)
    for w, p in zip(softmax_weights, params_list):
        final_params_8 += w * p[1:]  # インデックス1以降の8要素を使用

    # --- 2. eyeOpenness (インデックス0) のみファジーロジックで計算 ---

    # 感情名と重みの辞書を作成 (後の計算用)
    weights_dict = {name: weight for name, weight in zip(emotion_names, softmax_weights)}

    # 「開」グループ (KEYFRAME_PARAMS[emotion][0] == 1.0 のもの)
    Wide_Score = weights_dict.get("happy", 0) + weights_dict.get("angry", 0) + \
                 weights_dict.get("sad", 0) + weights_dict.get("astonished", 0)

    # 「閉」グループ (KEYFRAME_PARAMS[emotion][0] == 0.2 のもの)
    Narrow_Score = weights_dict.get("sleepy", 0) + weights_dict.get("relaxed", 0)

    # 「開」と「閉」のどちらが優勢か
    Score = Wide_Score - Narrow_Score

    # ゲイン k (この値で「寄せ具合」を調整。大きいほど0か1に振り切れる)
    k = 15  

    # シグモイド関数
    # Score がプラス (Wide優勢) なら 1.0 に、マイナス (Narrow優勢) なら 0.0 に近づく
    sigmoid_output = 1.0 / (1.0 + math.exp(-k * Score))
    
    # スコアとkの値を表示
    print(f"--- eyeOpenness ファジースコア計算 ---")
    print(f"Wide_Score: {Wide_Score:.4f}, Narrow_Score: {Narrow_Score:.4f}, Score: {Score:.4f}")
    print(f"Sigmoid出力: {sigmoid_output:.4f}")

    # ターゲットの eyeOpenness を 0.2 から 1.0 の間でマッピングする
    MIN_OPENNESS = 0.2
    MAX_OPENNESS = 1.0
    # sigmoid_output が 0.0 なら 0.2 に、1.0 なら 1.0 になる
    target_eyeOpenness = MIN_OPENNESS + (MAX_OPENNESS - MIN_OPENNESS) * sigmoid_output

    # --- 3. 最終パラメータを結合 ---
    final_params = np.concatenate(([target_eyeOpenness], final_params_8))
    
    # ===== ここまで! =====
    # if final_params[0] >= 0.8: final_params[0] = 1.0
    # else: final_params[0] = 0.2

    # if final_params[4] < 0.1:
    #     final_params[4] = 0
        
    print(f"--- 補間結果: V={target_v}, A={target_a} ---")
    param_names = [
    "eyeOpenness", "pupilSize", "pupilAngle", "upperEyelidAngle",
    "upperEyelidCoverage", "lowerEyelidCoverage", "mouthCurve",
    "mouthHeight", "mouthWidth"
    ]
    print(f"{'Parameter':<20} | {'Value':<12}")
    print("-" * 35)
    for name, value in zip(param_names, final_params):
        print(f"{name:<20} | {value:<12.4f}")
    print("=" * 35 + "\n")
    return final_params
    

# --- Flaskルーティング ---
@app.route("/", methods=["GET"])
def index():
    """ index.htmlをレンダリング """
    return render_template("index.html")

# --- Socket.IOイベントハンドラ ---
@socketio.on("user_message")
def handle_message(data):
    """ ユーザーからのメッセージを処理し、LLM と表情パラメータを返す """
    start_time = time.time()  # 処理開始時間を記録
    messages = data["messages"]
    
    instruction = """あなたは、ユーザの感情を理解し、自分自身も感情を表現できる未友達ロボットです。
    ユーザーと自然な日本語のタメ口で対話してください。

    # ルール
    1.  **出力形式:** 1行目に必ず `EMOTION: (V, A) (感情名)` を出力し、改行して2行目から応答を返す。
    2.  **感情座標:** V (快・不快) と A (覚醒・鎮静) をそれぞれ -1.0 (最小) 〜 1.0 (最大) の範囲で決める。
    3.  **役割:** あなたは感情豊かな友達ロボットとして、自然な応答を返す。
    
    # お手本 (AI自身の感情)

    ユーザー: やった！ついにプロジェクトが完成したんだ！
    あなた: 
    EMOTION: (0.8, 0.8) (excited)
    本当！すごい！おめでとう！

    ユーザー: わぁ！君の後ろにお化けがいるよ！
    あなた: 
    EMOTION: (1.0, 0.0) (astonished)
    えっ！？どこどこ！？

    ユーザー: （ため息）…別に、なんでもない。
    あなた: 
    EMOTION: (-0.8, -0.6) (disappointed)
    そっか…。話したくなったら、いつでも聞くからね。

    ユーザー: もう寝る時間だ。
    あなた: 
    EMOTION: (0.01, -1.0) (sleepy)
    ふわぁ…おやすみ…。"""
    
    # LLMへの指示を常にメッセージリストの先頭に追加
    messages.insert(0, {"role": "system", "content": instruction})

    print(f"[User] {messages[-1]['content']}")

    try:
        response = client.chat(model=LLM_MODEL, messages=messages, stream=True)
        
        full_text = ""
        
        # --- ★ 修正点 2: ストリーム処理のロジック変更 ---
        emotion_sent = False # 表情を送信済みかどうかのフラグ
        buffer = "" # EMOTION行を検出するためのバッファ

        for chunk in response:
            if "message" in chunk:
                subtext = chunk["message"]["content"]
                
                if not emotion_sent:
                    # EMOTION行が来るまでテキストをバッファに貯める
                    buffer += subtext
                    
                    # EMOTION行全体（改行含む）を探す
                    match = re.search(r'EMOTION:\s*\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)[^\n]*\n', buffer, re.IGNORECASE)
                    
                    if match:
                        # --- 感情を検出 ---
                        emotion_sent = True
                        v_val = float(match.group(1))
                        a_val = float(match.group(2))
                        print(f"--- 座標を検出 (ストリーム中): V={v_val}, A={a_val} ---")
                        
                        params = get_interpolated_expression(v_val, a_val)
                        param_names = [
                            "eyeOpenness", "pupilSize", "pupilAngle", "upperEyelidAngle", 
                            "upperEyelidCoverage", "lowerEyelidCoverage", "mouthCurve", 
                            "mouthHeight", "mouthWidth"
                        ]
                        param_dict = {name: val for name, val in zip(param_names, params)}
                        
                        print(f"--- 表情パラメータを送信 ---")
                        
                        # フロントエンドに表情パラメータを送信
                        emit("update_expression", param_dict)

                        # --- 感情行を除いた「残り」のテキストを送信 ---
                        # マッチしたEMOTION行の「後」のテキストを取得
                        remaining_text = buffer[match.end():]
                        
                        if remaining_text:
                            emit("bot_stream", {"chunk": remaining_text})
                            full_text += remaining_text
                        
                        # buffer = "" # バッファはもう使わない
                    
                    elif len(buffer) > 300: # 閾値 (EMOTION行は通常先頭に来るはず)
                        # プロンプト指示に従わず、EMOTION行が来ていない場合のフォールバック
                        print("--- 警告: EMOTION行が先頭で検出されませんでした。テキストをそのまま流します。 ---")
                        emit("bot_stream", {"chunk": buffer})
                        full_text += buffer
                        emotion_sent = True # 再検索しない
                        # buffer = ""
                
                else:
                    # --- 感情送信後の通常のテキストストリーム ---
                    emit("bot_stream", {"chunk": subtext})
                    full_text += subtext

        # ストリーム終了処理
        
        # バッファにテキストが残っている (＝EMOTIONが見つからないまま終わった)
        if not emotion_sent and buffer:
            print("--- 警告: EMOTION行が見つからないままストリームが終了しました。 ---")
            # EMOTION行がチャットに表示されるかもしれないが、テキストは送信する
            emit("bot_stream", {"chunk": buffer})
            full_text += buffer

        emit("bot_stream_end", {"text": full_text.strip()})
        print(f"[Bot] {full_text.strip()} (処理時間: {time.time() - start_time:.2f}秒)")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

@socketio.on('save_data')
def handle_save_data(data):
    """ フロントエンドから受信したデータをCSVに保存 """
    print("--- CSV保存リクエスト受信 ---")
    print(data)
    
    # ファイルが存在しない場合はヘッダーを書き込む
    file_exists = os.path.isfile(CSV_FILE_PATH)
    try:
        with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        print(f"--- データが {CSV_FILE_PATH} に保存されました ---")
        emit('save_success', {'message': 'データは正常に保存されました。'})
    except Exception as e:
        print(f"--- CSV保存エラー: {e} ---")
        emit('save_error', {'message': str(e)})

@socketio.on('manual_update_expression')
def handle_manual_update(data):
    """ コンソールからの手動での表情更新 """
    try:
        v_val = float(data['v'])
        a_val = float(data['a'])
        print(f"--- 手動更新: V={v_val}, A={a_val} ---")

        params = get_interpolated_expression(v_val, a_val)
        param_names = [
            "eyeOpenness", "pupilSize", "pupilAngle", "upperEyelidAngle",
            "upperEyelidCoverage", "lowerEyelidCoverage", "mouthCurve",
            "mouthHeight", "mouthWidth"
        ]
        param_dict = {name: val for name, val in zip(param_names, params)}

        emit("update_expression", param_dict)
        print(f"--- 表情パラメータを送信 (手動) ---")
    except (ValueError, KeyError) as e:
        print(f"--- 手動更新エラー: 無効なデータ {data} - {e} ---")

# --- サーバー起動 ---
if __name__ == "__main__":
    print("サーバーを http://127.0.0.1:5000 で起動します")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)