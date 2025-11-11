import os
import re
import time
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import ollama
import csv

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
W = 2.0
# 論文  の式(2)にある ε (イプシロン) の値（ゼロ除算防止用）
EPSILON = 1e-9 

# --- ご注意 ------------------------------------------------------------------
# まだ確実な数値ではない
# ---------------------------------------------------------------------------
KEYFRAME_PARAMS = {
    "astonished": np.array([1, 0.65, 0, -18, 0, 0, 12, 3, 1.15]),
    "excited": np.array([1, 0.6, 0, -18, 0, 0, 25, 1.15, 2.55]),
    "sleepy": np.array([0.15, 0.75, -11, 0, 0, 0, -15, 0.7, 1.55]),
    "disappointed": np.array([0.2, 0.65, -19, 0, 0, 0.03, 2, 2.5, 1.2]),
}
# 4感情に対応する「VA座標」
KEYFRAME_VA = {
    "astonished": np.array([5.0, 0.0]),
    "excited": np.array([0.70, 0.71]),
    "sleepy": np.array([0.01, -1.00]),
    "disappointed": np.array([-0.80,-0.03]),
}

def get_interpolated_expression(target_v, target_a):
    """ ターゲットのVA座標に基づき、表情パラメータを補間する """
    target_va = np.array([target_v, target_a])
    total_weight = 0.0
    weighted_params = np.zeros(9) # パラメータの数（元のコードに依存）
    
    for emotion_name, key_va in KEYFRAME_VA.items():
        distance = np.linalg.norm(target_va - key_va)
        
        # 論文の式(2)  に基づき、rtop_k (weight) を計算
        # rtop_k = 1 / ((100 * d_k)^w + ε)
        weight = 1.0 / (((100 * distance) ** W) + EPSILON)
        
        total_weight += weight
        weighted_params += weight * KEYFRAME_PARAMS[emotion_name]
        
    # 論文の式(1) [cite: 44] と 式(3) [cite: 48] を組み合わせた計算
    final_params = weighted_params / total_weight
    if final_params[0] >= 0.4: final_params[0] = 1.0
    else: final_params[0] = 0.2

    if final_params[4] < 0.1:
        final_params[4] = 0
        
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
    2.  **感情座標:** V (快・不快) と A (覚醒・鎮静) をそれぞれ -5.0 (最小) 〜 5.0 (最大) の範囲で決める。
    3.  **役割:** あなたは感情豊かな友達ロボットとして、自然な応答を返す。
    
    # お手本 (AI自身の感情)

    ユーザー: やった！ついにプロジェクトが完成したんだ！
    あなた: 
    EMOTION: (4.0, 4.0) (excited)
    本当！すごい！おめでとう！

    ユーザー: わぁ！君の後ろにお化けがいるよ！
    あなた: 
    EMOTION: (5.0, 0.0) (astonished)
    えっ！？どこどこ！？

    ユーザー: （ため息）…別に、なんでもない。
    あなた: 
    EMOTION: (-4.0, -3.0) (disappointed)
    そっか…。話したくなったら、いつでも聞くからね。

    ユーザー: もう寝る時間だ。
    あなた: 
    EMOTION: (0.05, -5.0) (sleepy)
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

# --- サーバー起動 ---
if __name__ == "__main__":
    print("サーバーを http://127.0.0.1:5000 で起動します")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)