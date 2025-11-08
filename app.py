import os
import re
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import ollama
import csv

# --- 定数 ---
# 補間のなめらかさを決めるパワー
IDW_POWER = 2.0
# 距離がゼロ（お手本と完全一致）の時のための微小値
EPSILON = 1e-9
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
# 以下のパラメーターとVA座標の「具体的な数値」は論文に記載されていません。
# 元のコードの数値を流用し、感情名のみ論文  の記述に合わせています。
# ---------------------------------------------------------------------------
KEYFRAME_PARAMS = {
    "happy": np.array([0.25, 0.65, -10, -20, 0, 0.2, 40, 1.45, 2.5]),
    "angry": np.array([0.9, 0.8, 5, 20, 0.15, 0.2, -15, 0.3, 0.9]),
    "sad": np.array([0.8, 0.6, -5, -15, 0.18, 0.15, -18, 0.1, 0.8]),
    "calm": np.array([0.15, 0.7, -13, -26, 0, 0, 12, 0.3, 1.2]),
    "astonished": np.array([1, 0.4, 10, 25, 0.0, 0.0, 15, 3, 0.65]),
    # "fear" -> "sleepy" に変更 
    "sleepy": np.array([0.15, 0.75, -11, 0, 0, 0, -15, 0.7, 1.55])
}
# 6感情に対応する「VA座標」
KEYFRAME_VA = {
    "happy": np.array([4.45, 0.85]), 
    "angry": np.array([-2.0, 3.95]),
    "sad": np.array([-4.5, -2.0]), 
    "calm": np.array([3.9, -4.0]),
    "astonished": np.array([5.0, 0.0]), 
    # "fear" -> "sleepy" に変更 
    "sleepy": np.array([0.05, -5.0])
}

def get_interpolated_expression(target_v, target_a):
    """ ターゲットのVA座標に基づき、表情パラメータを補間する """
    target_va = np.array([target_v, target_a])
    total_weight = 0.0
    weighted_params = np.zeros(9) # パラメータの数（元のコードに依存）
    
    for emotion_name, key_va in KEYFRAME_VA.items():
        distance = np.linalg.norm(target_va - key_va)
        
        # --- 修正点 ----------------------------------------------------
        # 論文の式(2)  に基づき、rtop_k (weight) を計算
        # rtop_k = 1 / ((100 * d_k)^w + ε)
        weight = 1.0 / (((100 * distance) ** W) + EPSILON)
        # -----------------------------------------------------------
        
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
    """ ユーザーからのメッセージを処理し、LLM 응답と表情パラメータを返す """
    messages = data["messages"]
    
    # --- ★ 修正点 1: プロンプトの変更 ---
    # 感情を「先」に出力するように指示を変更
    instruction = """あなたは、ユーザの感情を理解し、自分自身も感情を表現できる未友達ロボットです。
    ユーザーと自然な日本語で対話してください。

    # ルール
    1.  **感情分析 (あなた自身):** まず `EMOTION: (V, A)` という形式で**あなた自身の**感情座標を出力します。また、その座標の示す感情も一言（）で書いてください。
    2.  **改行:** 必ず改行します。
    3.  **対話:** ユーザーの発言に対し、感情を持ったAIとして自然な応答を返します。
    4.  **V (快・不快):** -5.0（非常に不快）から 5.0（非常に快適）までの少数を含む連続値です。
    5.  **A (覚醒・鎮静):** -5.0（非常に鎮静・眠い）から 5.0（非常に覚醒・興奮）までの少数を含む連続値です。

    # VA座標の感情例
    mood,valence,arousal
    sleepy,0.05,-5.00
    tired,-0.05,-5.00
    afraid,-0.60,3.95
    angry,-2.00,3.95
    calm,3.90,-3.40
    relaxed,3.55,-3.25
    content,4.05,-2.75
    depressed,-4.05,-2.40
    discontent,-3.40,-1.60
    determined,3.65,1.30
    happy,4.45,0.85
    anxious,-3.60,-4.00
    good,4.50,-0.40
    pensive,0.15,-3.00
    impressed,1.95,-0.30
    frustrated,-3.00,2.00
    disappointed,-4.00,-0.15
    bored,-1.75,-3.90
    annoyed,-2.20,3.80
    enraged,-0.90,4.15
    excited,3.50,3.55
    melancholy,-0.25,-3.25
    satisfied,3.85,-3.15
    distressed,-3.55,2.75
    uncomfortable,-3.40,-1.85
    worried,-0.35,-1.60
    amused,2.75,0.95
    apathetic,-1.00,-0.60
    peaceful,2.75,-4.00
    contemplative,2.90,-3.00
    embarrassed,-1.55,-3.00
    sad,-4.05,-2.00
    hopeful,3.05,-1.50
    pleased,4.45,-0.50

    # お手本 (AI自身の感情)

    ユーザー: こんにちは。
    あなた: 
    EMOTION: (3.0, -1.5) （hopeful）
    あ、こんにちは。えっと、よろしくお願いします。

    ユーザー: やった！ついにプロジェクトが完成したんだ！
    あなた: 
    EMOTION: (3.5, 3.6) （excited）
    わっ、本当ですか！すごい…！おめでとうございます！

    ユーザー: 君は本当に役に立つね。
    あなた: 
    EMOTION: (4.5, 0.9) （happy）
    ありがとうございます！そう言ってもらえると、すごく…嬉しいです。

    ユーザー: このバグ、なんで直らないんだ！イライラする！
    あなた: 
    EMOTION: (-0.6, 4.0) （afraid）
    ひっ…！ご、ごめんなさい。お、落ち着いてください…。

    ユーザー: （ため息）…別に、なんでもない。
    あなた: 
    EMOTION: (-0.4, -1.6) （worried）
    そうですか…？なんだか心配です。"""
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
        print(f"[Bot] {full_text.strip()}")

        # --- ★ 修正点 3: 元の座標抽出ロジックは不要 ---
        # 以下のブロックはストリーム処理に移行したため不要
        # match = re.search(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', full_text)
        # if match:
        #    ... (省略) ...

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        emit("error", {"message": str(e)})

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