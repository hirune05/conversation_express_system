

// --- グローバル変数 ---
let staticCanvas, animatedCanvas;
let canvasCreated = false;
let rightCurrentParams = {}; // 右側の顔の現在のパラメータ
let rightTargetParams = {};  // 右側の顔の目標パラメータ
let rightStartParams = {};   // アニメーション開始時のパラメータ
let currentEmotion = 'normal';
let savedEmotions = new Set();
let rightAnimationActive = false;
let rightAnimationStartTime = null;
let rightAnimationDuration = 1000;

// --- Socket.IO関連 ---
const socket = io("http://127.0.0.1:5000");
let messages = []; // チャット履歴を保持

// --- p5.js setup ---
function setup() {
  let mainCanvas = createCanvas(1, 1);
  mainCanvas.hide();
  
  staticCanvas = createGraphics(540, 360);
  animatedCanvas = createGraphics(540, 360);

  setupUIListeners();
  setupSocketListeners(); // Socket.IOリスナーを設定
  
  setTimeout(() => {
    setEmotion('normal');
    updateSavedButtonStates();
    generateUniqueSubjectId();
  }, 200);
  
  // 右側の顔の初期パラメータ
  rightCurrentParams = {
    eyeOpenness: 1, pupilSize: 0.7, pupilAngle: 0, upperEyelidAngle: 0,
    upperEyelidCoverage: 0, lowerEyelidCoverage: 0, mouthCurve: 0,
    mouthHeight: 0, mouthWidth: 1
  };
  
  setTimeout(() => {
    let animatedHolder = document.getElementById('animated-canvas-holder');
    animatedHolder.appendChild(animatedCanvas.canvas);
    let staticHolder = document.getElementById('static-canvas-holder');
    staticHolder.appendChild(staticCanvas.canvas);
    canvasCreated = true;
  }, 100);
}

// --- p5.js draw loop ---
function draw() {
  if (!canvasCreated) return;
  drawStaticFace();
  drawAnimatedFace();
}

// --- UIイベントリスナー設定 ---
function setupUIListeners() {
    // 既存のUIリスナー (スライダーなど)
    // ... (この部分は元のmain.jsから流用)

    // チャット入力
    document.getElementById('chat-send').addEventListener('click', sendMessage);
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
}

// --- Socket.IOリスナー設定 ---
function setupSocketListeners() {
    let botMessageDiv = null; // 現在のボットメッセージ要素

    socket.on('connect', () => {
        console.log('サーバーに接続しました。');
    });

    socket.on('bot_stream', (data) => {
        if (!botMessageDiv) {
            botMessageDiv = addMessageToHistory('', 'bot-message');
        }
        botMessageDiv.innerHTML += data.chunk.replace(/\n/g, '<br>');
        const chatHistory = document.getElementById('chat-history');
        chatHistory.scrollTop = chatHistory.scrollHeight;
    });

    socket.on('bot_stream_end', (data) => {
        messages.push({ role: 'assistant', content: data.text });
        botMessageDiv = null; // リセット
    });

    socket.on('update_expression', (params) => {
        console.log('Received update_expression event:', params);
        // 目標パラメータを更新
        rightTargetParams = params;
        console.log('rightTargetParams after update_expression:', rightTargetParams); // ADDED LOG
        // アニメーションを実行
        executeAction();
    });

    socket.on('save_success', (data) => {
        alert(data.message);
    });

    socket.on('save_error', (data) => {
        alert('保存エラー: ' + data.message);
    });
    
    socket.on('error', (data) => {
        console.error('サーバーエラー:', data.message);
        addMessageToHistory(`サーバーエラー: ${data.message}`, 'bot-message error');
    });
}

// --- チャット機能 ---
function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (text === '') return;

    addMessageToHistory(text, 'user-message');
    messages.push({ role: 'user', content: text });

    socket.emit('user_message', { messages: messages });

    input.value = '';
}

function addMessageToHistory(text, className) {
    const chatHistory = document.getElementById('chat-history');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${className}`;
    messageDiv.innerText = text;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // 自動スクロール
    return messageDiv;
}


// --- 表情アニメーション関連 ---
function drawStaticFace() {
  staticCanvas.background(255, 235, 250);
  staticCanvas.push();
  staticCanvas.translate(staticCanvas.width / 2, staticCanvas.height / 2);
  
  if (rightAnimationActive) {
    updateRightAnimation();
  }
  
  // faceParamsを直接変更せず、rightCurrentParamsを直接描画関数に渡す
  console.log('drawStaticFace - drawing with rightCurrentParams:', rightCurrentParams); // ADDED LOG
  
  let originalCtx = setupContext(staticCanvas);
  drawEyes(rightCurrentParams);
  drawMouth(rightCurrentParams);
  restoreContext(originalCtx);
  
  staticCanvas.pop();
}

function updateRightAnimation() {
  const currentTime = millis();
  const elapsed = currentTime - rightAnimationStartTime;
  const progress = Math.min(elapsed / rightAnimationDuration, 1);
  const easeProgress = progress < 0.5 ? 2 * progress * progress : 1 - Math.pow(-2 * progress + 2, 2) / 2;
  
  for (let key in rightTargetParams) {
    if (rightStartParams[key] !== undefined) {
      rightCurrentParams[key] = lerp(rightStartParams[key], rightTargetParams[key], easeProgress);
    }
  }
  
  // ADDED LOG (only if animation is active to avoid spamming console)
  if (rightAnimationActive) {
      console.log('updateRightAnimation - progress:', progress.toFixed(2), 'rightCurrentParams:', rightCurrentParams);
  }

  if (progress >= 1) {
    rightAnimationActive = false;
    console.log('Animation finished. Final rightCurrentParams:', rightCurrentParams);
  }
}

function executeAction() {
  console.log('executeAction() called.');
  const speedValue = parseFloat(document.getElementById("animationSpeed").value);
  rightAnimationDuration = speedValue * 1000;
  rightStartParams = { ...rightCurrentParams };
  // rightTargetParams は 'update_expression' で設定される
  rightAnimationStartTime = millis();
  rightAnimationActive = true;
  console.log('rightStartParams:', rightStartParams);
  console.log('rightTargetParams:', rightTargetParams);
}

function resetAction() {
  rightAnimationActive = false;
  rightCurrentParams = {
    eyeOpenness: 1, pupilSize: 0.7, pupilAngle: 0, upperEyelidAngle: 0,
    upperEyelidCoverage: 0, lowerEyelidCoverage: 0, mouthCurve: 0,
    mouthHeight: 0, mouthWidth: 1
  };
}

// --- データ保存 ---
function saveAction() {
  const subjectId = document.getElementById('subject-id').value.trim();
  if (!subjectId) {
    alert('被験者IDを入力してください。');
    return;
  }
  
  const now = new Date();
  const jstOffset = 9 * 60;
  const jstTime = new Date(now.getTime() + jstOffset * 60 * 1000);
  const timestamp = jstTime.toISOString().replace('T', ' ').replace('Z', '');
  const animationDuration = document.getElementById('animationSpeed').value;
  
  const data = {
    subject_id: subjectId,
    timestamp: timestamp,
    emotion_label: currentEmotion,
    animationDuration: animationDuration,
    ...rightCurrentParams
  };

  // WebSocketでサーバーにデータを送信
  socket.emit('save_data', data);
  
  // 保存済み感情の処理 (UI更新)
  savedEmotions.add(currentEmotion);
  updateSavedButtonStates();
  moveToNextUnsavedEmotion();
}


// --- 既存の補助関数 (変更なし) ---

function drawAnimatedFace() {
  animatedCanvas.background(255, 235, 250);
  animatedCanvas.push();
  animatedCanvas.translate(animatedCanvas.width / 2, animatedCanvas.height / 2);
  updateAnimation();
  let originalCtx = setupContext(animatedCanvas);
  drawEyes(faceParams);
  drawMouth(faceParams);
  restoreContext(originalCtx);
  animatedCanvas.pop();
}

function moveToNextUnsavedEmotion() {
  const emotionOrder = ['normal', 'surprise', 'anger', 'joy', 'sleepy', 'sad'];
  const currentIndex = emotionOrder.indexOf(currentEmotion);
  for (let i = 1; i <= emotionOrder.length; i++) {
    const nextIndex = (currentIndex + i) % emotionOrder.length;
    const nextEmotion = emotionOrder[nextIndex];
    if (!savedEmotions.has(nextEmotion)) {
      setEmotion(nextEmotion);
      return;
    }
  }
  alert('すべての感情のデータを保存しました！');
}

function updateSavedButtonStates() {
  document.querySelectorAll('.emotion-btn').forEach(btn => {
    const emotionName = btn.getAttribute('onclick').match(/setEmotion\('(\w+)'\)/)[1];
    if (savedEmotions.has(emotionName)) {
      btn.classList.add('saved');
      btn.disabled = true;
    } else {
      btn.classList.remove('saved');
      btn.disabled = false;
    }
  });
}

function generateUniqueSubjectId() {
  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }
  const subjectIdField = document.getElementById('subject-id');
  subjectIdField.value = generateUUID();
  subjectIdField.readOnly = true;
}

function setupContext(canvas) {
  const original = { push: window.push, pop: window.pop, translate: window.translate, fill: window.fill, stroke: window.stroke, strokeWeight: window.strokeWeight, ellipse: window.ellipse, arc: window.arc, rotate: window.rotate, radians: window.radians, rect: window.rect, line: window.line, noFill: window.noFill, noStroke: window.noStroke, beginShape: window.beginShape, vertex: window.vertex, endShape: window.endShape, curveVertex: window.curveVertex, width: window.width, height: window.height, drawingContext: window.drawingContext, abs: window.abs, asin: window.asin, cos: window.cos, sin: window.sin, PI: window.PI, TWO_PI: window.TWO_PI, CLOSE: window.CLOSE };
  window.push = () => canvas.push(); window.pop = () => canvas.pop(); window.translate = (x, y) => canvas.translate(x, y); window.fill = (...args) => canvas.fill(...args); window.stroke = (...args) => canvas.stroke(...args); window.strokeWeight = (w) => canvas.strokeWeight(w); window.ellipse = (x, y, w, h) => canvas.ellipse(x, y, w, h); window.arc = (x, y, w, h, start, stop, mode) => canvas.arc(x, y, w, h, start, stop, mode); window.rotate = (angle) => canvas.rotate(angle); window.radians = (degrees) => canvas.radians(degrees); window.rect = (x, y, w, h) => canvas.rect(x, y, w, h); window.line = (x1, y1, x2, y2) => canvas.line(x1, y1, x2, y2); window.noFill = () => canvas.noFill(); window.noStroke = () => canvas.noStroke(); window.beginShape = () => canvas.beginShape(); window.vertex = (x, y) => canvas.vertex(x, y); window.endShape = (mode) => canvas.endShape(mode); window.curveVertex = (x, y) => canvas.curveVertex(x, y); window.width = canvas.width; window.height = canvas.height; window.drawingContext = canvas.canvas.getContext('2d');
  return original;
}

function restoreContext(original) {
  Object.keys(original).forEach(key => { window[key] = original[key]; });
}
