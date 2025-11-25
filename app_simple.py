import os
import re
import time
import uuid
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import ollama
import csv
import math

# --- 定数 ---
# LLMモデル
LLM_MODEL = "qwen3:8b"

# CSVファイルパス（時間計測用）
TIMING_CSV_FILE = 'timing_data_simple.csv'
TIMING_CSV_HEADERS = ['timestamp', 'total_time', 'llm_time', 'param_time']

# --- FlaskとSocket.IOの初期化 ---
# 静的ファイルとテンプレートフォルダをルートディレクトリに設定
app = Flask(__name__, static_folder='static', template_folder='.')
app.config["SECRET_KEY"] = "C0HThSwr"
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Ollamaの初期化 ---
client = ollama.Client()

# --- 時間計測用CSV出力関数 ---
def save_timing_to_csv(total_time, llm_time, param_time):
    """処理時間をCSVファイルに保存する"""
    import datetime
    
    # ファイルが存在しない場合はヘッダーを書き込む
    file_exists = os.path.isfile(TIMING_CSV_FILE)
    
    try:
        with open(TIMING_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=TIMING_CSV_HEADERS)
            
            if not file_exists:
                writer.writeheader()
            
            # データを書き込み
            writer.writerow({
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_time': round(total_time, 4),
                'llm_time': round(llm_time, 4), 
                'param_time': round(param_time, 4)
            })
            
        print(f"CSV保存成功: {TIMING_CSV_FILE} - 全体:{total_time:.4f}s, LLM:{llm_time:.4f}s, パラメータ:{param_time:.4f}s")
        
    except Exception as e:
        print(f"CSV保存エラー: {e}")

# --- Flaskルーティング ---
@app.route("/", methods=["GET"])
def index():
    """ index.htmlをレンダリング """
    return render_template("index.html")

# --- Socket.IOイベントハンドラ ---
@socketio.on("user_message")
def handle_message(data):
    """ ユーザーからのメッセージを処理し、シンプルなテキスト応答を返す """
    start_time = time.time()  # 全体処理開始時間を記録
    llm_start_time = None     # LLM処理開始時間
    llm_end_time = None       # LLM処理終了時間
    messages = data["messages"]
    
    instruction = """あなたは、ユーザの感情を理解し、自分自身も感情を表現できる友達ロボットです。
    ユーザーと自然な日本語のタメ口で対話し，応答に相応しいあなた自身の感情を感情座標を用いて表現してください。

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
    
    # 会話履歴を無効化：現在のメッセージのみをLLMに送信
    # messages.insert(0, {"role": "system", "content": instruction}) # コメントアウト：履歴使用を無効化
    # 現在のメッセージのみでLLMに送信
    current_message = messages[-1]['content'] if messages else ""
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": current_message}]

    print(f"[User] {current_message}") # 履歴なしで現在のメッセージのみ表示

    try:
        llm_start_time = time.time()  # LLM処理開始時間を記録
        response = client.chat(model=LLM_MODEL, messages=messages, stream=True)
        
        full_text = ""
        
        # --- シンプルなテキストストリーム処理 ---
        for chunk in response:
            if "message" in chunk:
                subtext = chunk["message"]["content"]
                emit("bot_stream", {"chunk": subtext})
                full_text += subtext

        # ストリーム終了処理
        llm_end_time = time.time()  # LLM処理終了時間を記録
        
        emit("bot_stream_end", {"text": full_text.strip()})
        
        # 処理時間をCSV出力
        end_time = time.time()
        total_time = end_time - start_time
        llm_time = llm_end_time - llm_start_time if llm_start_time and llm_end_time else 0
        param_time = 0  # パラメータ計算なし
        
        save_timing_to_csv(total_time, llm_time, param_time) # CSVに時間データを保存
        print(f"[Bot] {full_text.strip()}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

# --- サーバー起動 ---
if __name__ == "__main__":
    print("サーバーを http://127.0.0.1:5000 で起動します")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)