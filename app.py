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

# --- 時間計測用表示関数 ---
def print_timing_table(total_time, llm_time, param_time):
    """処理時間を表形式で出力する"""
    print("\n" + "="*50)
    print("           処理時間計測結果")
    print("="*50)
    print(f"{'項目':<20} | {'時間(秒)':<10} | {'割合(%)':<8}")
    print("-"*50)
    print(f"{'全体処理時間':<20} | {total_time:<10.4f} | {'100.0':<8}")
    print(f"{'LLM応答生成':<20} | {llm_time:<10.4f} | {(llm_time/total_time*100 if total_time > 0 else 0):<8.1f}")
    print(f"{'パラメータ計算':<20} | {param_time:<10.4f} | {(param_time/total_time*100 if total_time > 0 else 0):<8.1f}")
    other_time = total_time - llm_time - param_time
    print(f"{'その他処理':<20} | {other_time:<10.4f} | {(other_time/total_time*100 if total_time > 0 else 0):<8.1f}")
    print("="*50 + "\n")

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
       
# def get_interpolated_expression(target_v, target_a):
#     """ ターゲットのVA座標に基づき、表情パラメータを補間する """
#     target_va = np.array([target_v, target_a])

#     weights = {}  # 各感情の rtop_k を記録（式2）
    
#     # --- まず rtop_k を全て求める ---
#     for emotion_name, key_va in KEYFRAME_VA.items():
#         distance = np.linalg.norm(target_va - key_va)

#         rtop_k = 1.0 / (((100 * distance) ** W) + EPSILON)
#         weights[emotion_name] = rtop_k

#     # --- 次に r_k を求める（式1の正規化） ---
#     total_rtop = sum(weights.values())
#     r_values = {name: w / total_rtop for name, w in weights.items()}

#     # ★ ここで r_k を print ★
#     print("=== r_k values (emotion weights) ===")
#     for name, r in r_values.items():
#         print(f"{name}: {r}")

#     # --- 最後に式(3)の p を求める ---
#     final_params = np.zeros(9)
#     for emotion_name, r_k in r_values.items():
#         final_params += r_k * KEYFRAME_PARAMS[emotion_name]
 
        
        
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
    print(f"{'感情':<10} | {'距離':<8} | {'rtop_k':<11} | {'r_k (softmax)':<15}")
    print("-" * 65)
    for i, emotion_name in enumerate(emotion_names):
        distance = np.linalg.norm(target_va - KEYFRAME_VA[emotion_name])
        print(f"{emotion_name:<12} | {distance:<10.4f} | {rtop_values[i]:<15.6f} | {softmax_weights[i]:<15.6f}")
    print("=" * 65 + "\n")
    
    # ========================
    #     r_k の表示部分
    # ========================
    print("\n=== r_k values (emotion weights, normalized) ===")
    for name, rk in zip(emotion_names, softmax_weights):   # ← rk_values は不要
        print(f"{name}: {rk}")   # 桁数多めで表示
    print("===============================================\n")
    
    # # ===== 重み付き平均 =====
    # final_params = np.zeros(9)
    # for w, p in zip(softmax_weights, params_list):
    #     final_params += w * p

    # ===== 1. 従来の重み付き平均 (ベース計算) =====
    # まず、全9パラメータを従来の加重平均で計算する
    final_params = np.zeros(9)
    for w, p in zip(softmax_weights, params_list):
        final_params += w * p

    # ベースとなる値（ファジー適用前）を保持
    base_eyeOpenness = final_params[0]
    base_upperEyelidCoverage = final_params[4]

    print(f"--- ベース補間結果 (ファジー適用前) ---")
    print(f"Base eyeOpenness: {base_eyeOpenness:.4f}")
    print(f"Base upperEyelidCoverage: {base_upperEyelidCoverage:.4f}")
    print("-" * 35)


    # ===== 2. ファジー制御による上書き =====

    # 感情名と重みの辞書を作成 (後の計算用)
    weights_dict = {name: weight for name, weight in zip(emotion_names, softmax_weights)}

    # --- 2a. eyeOpenness (インデックス0) の上書き ---
    Wide_Score = weights_dict.get("happy", 0) + weights_dict.get("angry", 0) + \
                 weights_dict.get("sad", 0) + weights_dict.get("astonished", 0)
    Narrow_Score = weights_dict.get("sleepy", 0) + weights_dict.get("relaxed", 0)
    Score = Wide_Score - Narrow_Score
    k = 20.0 
    sigmoid_output_eye = 1.0 / (1.0 + math.exp(-k * Score))
    
    MIN_OPENNESS = 0.2
    MAX_OPENNESS = 1.0
    target_eyeOpenness = MIN_OPENNESS + (MAX_OPENNESS - MIN_OPENNESS) * sigmoid_output_eye
    
    # final_params のインデックス0 を上書き
    final_params[0] = target_eyeOpenness

    # --- 2b. upperEyelidCoverage (インデックス4) の上書き (ハイブリッド方式) ---
    
        # --- ゲート(ON/OFF)の計算 ---
    # 「被る」グループの重みの合計を計算
    w_angry = weights_dict.get("angry", 0)
    w_sad = weights_dict.get("sad", 0)
    Cover_Score = w_angry + w_sad

    # 「被らない」グループの重みの合計を計算
    No_Cover_Score = weights_dict.get("happy", 0) + weights_dict.get("astonished", 0) + \
                     weights_dict.get("sleepy", 0) + weights_dict.get("relaxed", 0)
    
    # 「被る」と「被らない」のどちらが優勢か
    Score_coverage = Cover_Score - No_Cover_Score

    # ゲイン k (大きいほど急激に 0.0 か 1.0 に振り切れる)
    k_coverage = 20.0

    # シグモイド関数 (ゲートの開閉度: 0.0～1.0)
    # Score がプラス (Cover優勢) なら 1.0 に、マイナス (No_Cover優勢) なら 0.0 に近づく
    sigmoid_gate = 1.0 / (1.0 + math.exp(-k_coverage * Score_coverage))
    
    # --- 最終的な値の決定 ---
    # target = ゲート(0 or 1) * 目標値(Base値)
    # これにより、「微妙な値」はゲートが0.0になり消滅し、
    # 「意図した値」はゲートが1.0になりそのまま使われる。
    target_coverage = sigmoid_gate * base_upperEyelidCoverage
    
    print(f"--- upperEyelidCoverage ファジー計算 (ゲート * Base値 方式) ---")
    print(f"Cover_Score: {Cover_Score:.4f}, No_Cover_Score: {No_Cover_Score:.4f}, Score: {Score_coverage:.4f}")
    print(f"Sigmoid出力 (ゲート): {sigmoid_gate:.4f}")
    print(f"目標値 (Base値): {base_upperEyelidCoverage:.4f}")
    print(f"target_coverage (出力): {target_coverage:.4f}")

    # final_params のインデックス4 を上書き
    final_params[4] = target_coverage


    # ===== 3. 最終結果の表示 =====
    print(f"\n--- 補間結果 (ファジー適用後): V={target_v}, A={target_a} ---")
    param_names = [
    "eyeOpenness", "pupilSize", "pupilAngle", "upperEyelidAngle",
    "upperEyelidCoverage", "lowerEyelidCoverage", "mouthCurve",
    "mouthHeight", "mouthWidth"
    ]
    print(f"{'Parameter':<20} | {'Value':<12}")
    print("-" * 35)
    for i, (name, value) in enumerate(zip(param_names, final_params)):
        if i == 0:
            print(f"{name:<20} | {value:<12.4f}  <-- ファジー制御 (EyeOpen)")
        elif i == 4:
            print(f"{name:<20} | {value:<12.4f}  <-- ファジー制御 (EyelidCov)")
        else:
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
    start_time = time.time()  # 全体処理開始時間を記録
    llm_start_time = None     # LLM処理開始時間
    llm_end_time = None       # LLM処理終了時間
    param_start_time = None   # パラメータ計算開始時間
    param_end_time = None     # パラメータ計算終了時間
    messages = data["messages"]
    emotion_line = None  # EMOTION行を保存する変数
    v_val = None
    a_val = None  
    emotion_label = None
    
    # デバッグ: 受信データの確認
    print(f"--- 受信データ確認 ---")
    print(f"last_emotion in data: {'last_emotion' in data}")
    if "last_emotion" in data:
        print(f"last_emotion value: {data['last_emotion']}")
        print(f"last_emotion type: {type(data['last_emotion'])}")
    print("--- 受信データ確認終了 ---")
    
    # 前回の感情座標を取得（最後のassistantメッセージから）
    last_emotion_info = ""
    if len(messages) > 0:
        # 直近のbotの感情座標をデータから取得
        if "last_emotion" in data and data["last_emotion"] is not None:
            last_v = data["last_emotion"].get("v", "不明")
            last_a = data["last_emotion"].get("a", "不明")
            last_label = data["last_emotion"].get("label", "不明")
            last_emotion_info = f"\n\n# CRITICAL INSTRUCTION: PREVIOUS EMOTION STATE\nYour PREVIOUS emotion was: v={last_v}, a={last_a} ({last_label})\n\n**MANDATORY RULES:**\n1. You MUST NOT output the same coordinates (v={last_v}, a={last_a})\n2. The difference between your new coordinates and previous ones MUST be at least 0.3 in total distance\n3. If the user's input doesn't warrant a major emotional change, still vary your coordinates significantly\n4. FORBIDDEN: Any output with v={last_v} OR a={last_a} (even if the other coordinate changes)\n\n**VERIFY BEFORE OUTPUT:** Check that your new v,a values are sufficiently different from v={last_v}, a={last_a}"
    
    instruction = """You are an empathetic robot friend who understands user emotions and expresses your own emotions richly.
    You must interact with the user in natural, casual Japanese ("Tame-guchi").

    # GOAL
    Analyze the user's input, determine your appropriate emotional response, and generate a reply in Japanese.

    # OUTPUT FORMAT (Strictly Enforced)
    You must output the response in the following XML format:
    1. **<thought>**
        - Analyze the nuance of the user's input (e.g., joy, sarcasm, sadness).
        - Explain the logic behind your emotional reaction.
        - **You MUST avoid producing similar coordinates to the previous turn.**
        - **Small changes (within ±0.10) are strictly forbidden unless the context explicitly demands emotional stability.**
        - If the user shows strong emotion (joy, shock, anger, fear, frustration), you MUST output a correspondingly strong reaction.
    2. **<emotion>**
        - `v`: Valence from -1.0 (Negative) to 1.0 (Positive).
        - `a`: Arousal from -1.0 (Sleepy) to 1.0 (Active).
        - Content: A single English emotion label (e.g., happy, surprised, angry).
        - Strive to ensure the change from the previous instance is 0.30 or greater
    3. **Response Text**
        - The actual reply to the user.
        - **MUST BE IN JAPANESE.**
        - Use a casual, friendly tone (タメ口).
        - **Use "ボク" as your first-person pronoun.**
    
    形式:
    <thought>
    ここに思考プロセスを書く。前回の感情に引きずられず、今回の文脈に合わせて再計算する。
    </thought>
    <emotion v="0.00" a="0.00">emotion_label</emotion>
    ここにユーザーへの返答を書く。
    
    # PARAMETER GUIDELINES
    - **Valence (v):** Positive values for happiness/joy; Negative for sadness/anger.
    - **Arousal (a):** Positive for excitement/tension; Negative for relaxation/boredom.
    - **Dynamic Range:** Ensure the coordinates vary dynamically based on the conversation. Do not stick to "safe" average values (e.g., 0.5, 0.0).
    
    # EXAMPLES

    User: やった！ついにプロジェクトが完成したんだ！
    Assistant:
    <thought>
    The user completed a project. I feel very happy for them and my energy is high.
    </thought>
    <emotion v="0.70" a="0.71">excited</emotion>
    本当！すごい！おめでとう！

    User: わぁ！君の後ろにお化けがいるよ！
    Assistant:
    <thought>
    The user surprised me with a scary statement. My arousal spikes to maximum, but valence is neutral/confused.
    </thought>
    <emotion v="0.0" a="1.0">astonished</emotion>
    えっ！？どこどこ！？

    User: （ため息）…別に、なんでもない。
    Assistant:
    <thought>
    The user sighed and seems down. I feel disappointed and sad for them. Valence is very low.
    </thought>
    <emotion v="-0.80" a="-0.03">disappointed</emotion>
    そっか…。話したくなったら、いつでも聞くからね。

    User: もう寝る時間だ。
    Assistant:
    <thought>
    The user is going to sleep. I feel drowsy and calm, with very low arousal.
    </thought>
    <emotion v="0.01" a="-1.0">sleepy</emotion>
    ふわぁ…おやすみ…。
    
    User: 今日は一日ゆっくりできてよかった。
    Assistant:
    <thought>
    The user had a relaxing day and feels content. I feel satisfied and peaceful too.
    </thought>
    <emotion v="0.81" a="-0.55">content</emotion>
    そうだね、のんびりした日も大切だよね。"""
    
    # 前回の感情情報をinstructionに追加
    full_instruction = instruction + last_emotion_info
    
    # LLMへの指示を常にメッセージリストの先頭に追加
    messages.insert(0, {"role": "system", "content": full_instruction})
    
    # デバッグ: LLMに送られるinstructionの内容を確認
    print(f"\n--- LLMに送信されるInstruction ---")
    print(f"前回の感情情報: {repr(last_emotion_info)}")
    print(f"Full instruction末尾 (200文字): ...{full_instruction[-200:]}")
    print("--- Instruction確認終了 ---")

    print(f"[User] {messages[-1]['content']}")
    
    # デバッグ: 履歴の中身を確認
    print(f"\n--- メッセージ履歴の確認 (件数: {len(messages)}) ---")
    for i, msg in enumerate(messages):
        if msg['role'] == 'system':
            print(f"{i}: system - [インストラクション省略]")
        else:
            content_display = msg['content'].replace('\n', '\\n')
            print(f"{i}: {msg['role']} - {content_display}")
    print("--- 履歴確認終了 ---\n")

    try:
        llm_start_time = time.time()  # LLM処理開始時間を記録
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
                    
                    # XML形式の感情タグを探す (小数点表記に対応)
                    emotion_match = re.search(r'<emotion\s+v="(-?\d*\.?\d+)"\s+a="(-?\d*\.?\d+)">([^<]*)</emotion>', buffer, re.IGNORECASE)
                    
                    if emotion_match:
                        # --- 感情を検出 ---
                        emotion_sent = True
                        v_val = float(emotion_match.group(1))
                        a_val = float(emotion_match.group(2))
                        emotion_label = emotion_match.group(3)
                        emotion_line = emotion_match.group(0).strip()
                        print(f"--- 座標を検出 (ストリーム中): V={v_val}, A={a_val}, 感情: {emotion_label} ---")
                        
                        param_start_time = time.time()  # パラメータ計算開始時間を記録
                        params = get_interpolated_expression(v_val, a_val)
                        param_end_time = time.time()    # パラメータ計算終了時間を記録
                        param_names = [
                            "eyeOpenness", "pupilSize", "pupilAngle", "upperEyelidAngle", 
                            "upperEyelidCoverage", "lowerEyelidCoverage", "mouthCurve", 
                            "mouthHeight", "mouthWidth"
                        ]
                        param_dict = {name: val for name, val in zip(param_names, params)}
                        
                        print(f"--- 表情パラメータを送信 ---")
                        
                        # フロントエンドに表情パラメータを送信
                        emit("update_expression", param_dict)

                        # --- 履歴保存用にはXMLタグを除去したテキストのみ保存 ---
                        clean_buffer = buffer
                        # <thought>...</thought>を除去
                        clean_buffer = re.sub(r'<thought>.*?</thought>\s*', '', clean_buffer, flags=re.DOTALL)
                        # <emotion>...</emotion>を除去
                        clean_buffer = re.sub(r'<emotion[^>]*>.*?</emotion>\s*', '', clean_buffer, flags=re.DOTALL)
                        clean_buffer = clean_buffer.strip()
                        full_text += clean_buffer
                        
                        # --- 画面表示用にはthoughtとemotionタグを除去したテキストを送信 ---
                        display_text = buffer
                        # <thought>...</thought>を除去
                        display_text = re.sub(r'<thought>.*?</thought>\s*', '', display_text, flags=re.DOTALL)
                        # <emotion>...</emotion>を除去
                        display_text = re.sub(r'<emotion[^>]*>.*?</emotion>\s*', '', display_text, flags=re.DOTALL)
                        # 先頭と末尾の空白を除去
                        display_text = display_text.strip()
                        
                        if display_text:
                            emit("bot_stream", {"chunk": display_text})
                        
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
        llm_end_time = time.time()  # LLM処理終了時間を記録
        
        # バッファにテキストが残っている (＝EMOTIONが見つからないまま終わった)
        if not emotion_sent and buffer:
            print("--- 警告: EMOTION行が見つからないままストリームが終了しました。 ---")
            # EMOTION行がチャットに表示されるかもしれないが、テキストは送信する
            emit("bot_stream", {"chunk": buffer})
            full_text += buffer

        # 今回の感情座標をフロントエンドに送信
        current_emotion = None
        if v_val is not None and a_val is not None and emotion_label is not None:
            current_emotion = {"v": v_val, "a": a_val, "label": emotion_label}
            
        emit("bot_stream_end", {
            "text": full_text.strip(),
            "emotion": current_emotion
        })
        
        # 処理時間を表形式で出力
        end_time = time.time()
        total_time = end_time - start_time
        llm_time = llm_end_time - llm_start_time if llm_start_time and llm_end_time else 0
        param_time = param_end_time - param_start_time if param_start_time and param_end_time else 0
        
        # print_timing_table(total_time, llm_time, param_time)
        
        # EMOTION情報と応答を一緒に出力
        if 'emotion_line' in locals() and emotion_line:
            print(f"[Bot] {emotion_line}")
        elif 'v_val' in locals() and 'a_val' in locals() and 'emotion_label' in locals():
            print(f"[Bot] <emotion v=\"{v_val}\" a=\"{a_val}\">{emotion_label}</emotion>")
        
        # 画面表示用にクリーンアップしたテキストも出力
        clean_text = full_text.strip()
        # <thought>...</thought>を除去
        clean_text = re.sub(r'<thought>.*?</thought>\s*', '', clean_text, flags=re.DOTALL)
        # <emotion>...</emotion>を除去  
        clean_text = re.sub(r'<emotion[^>]*>.*?</emotion>\s*', '', clean_text, flags=re.DOTALL)
        clean_text = clean_text.strip()
        
        print(f"[Bot] {clean_text}")

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