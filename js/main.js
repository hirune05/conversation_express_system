// 2つの独立したキャンバスを作成（シンプルなアプローチ）
let staticCanvas, animatedCanvas;
let canvasCreated = false;
let staticFaceParams = null; // 右側の顔のパラメータ
let currentEmotion = 'normal'; // 現在選択されている感情
let savedEmotions = new Set(); // 保存済みの感情を記録

// 右側の顔のアニメーション用変数
let rightAnimationActive = false;
let rightAnimationStartTime = null;
let rightAnimationDuration = 1000; // デフォルト1秒
let rightStartParams = {};
let rightTargetParams = {};
let rightCurrentParams = {};

function setup() {
  // メインキャンバスを非表示で作成
  let mainCanvas = createCanvas(1, 1);
  mainCanvas.hide();
  
  // 静的な顔のキャンバス（上部）
  staticCanvas = createGraphics(540, 360);
  
  // アニメーション顔のキャンバス（下部）
  animatedCanvas = createGraphics(540, 360);

  // UIイベントリスナーの設定
  setupUIListeners();
  
  // デフォルトでニュートラルを選択
  setTimeout(() => {
    setEmotion('normal');
    // 初期状態で保存済み感情をチェック（必要に応じて）
    updateSavedButtonStates();
    // ユニークな被験者IDを生成
    generateUniqueSubjectId();
  }, 200);
  
  // 右側の顔の初期パラメータを設定（デフォルト値）
  rightCurrentParams = {
    eyeOpenness: 1,
    pupilSize: 0.7,
    pupilAngle: 0,
    upperEyelidAngle: 0,
    upperEyelidCoverage: 0,
    lowerEyelidCoverage: 0,
    mouthCurve: 0,
    mouthHeight: 0,
    mouthWidth: 1
  };
  
  // キャンバスをDOMに追加
  setTimeout(() => {
    // アニメーションキャンバスを左側に配置
    let animatedHolder = document.getElementById('animated-canvas-holder');
    animatedHolder.appendChild(animatedCanvas.canvas);
    
    // 静的キャンバスを右側に配置
    let staticHolder = document.getElementById('static-canvas-holder');
    staticHolder.appendChild(staticCanvas.canvas);
    
    canvasCreated = true;
  }, 100);
}

function draw() {
  if (!canvasCreated) return;
  
  // 静的な顔を描画（上部キャンバス）
  drawStaticFace();
  
  // アニメーション顔を描画（下部キャンバス）
  drawAnimatedFace();
}

// 静的な顔を描画
function drawStaticFace() {
  staticCanvas.background(255, 235, 250);
  staticCanvas.push();
  staticCanvas.translate(staticCanvas.width / 2, staticCanvas.height / 2);
  
  // 右側のアニメーションを更新
  if (rightAnimationActive) {
    updateRightAnimation();
  }
  
  // デフォルトのパラメータを一時的に保存
  const savedParams = { ...faceParams };
  
  // 右側の現在のパラメータを使用
  Object.assign(faceParams, rightCurrentParams);
  
  // 描画コンテキストを静的キャンバスに設定
  let originalCtx = setupContext(staticCanvas);
  
  // 既存の関数を使って描画
  drawEyes();
  drawMouth();
  
  // コンテキストを復元
  restoreContext(originalCtx);
  
  // パラメータを元に戻す
  Object.assign(faceParams, savedParams);
  
  staticCanvas.pop();
}

// 右側のアニメーション更新
function updateRightAnimation() {
  const currentTime = millis();
  const elapsed = currentTime - rightAnimationStartTime;
  const progress = Math.min(elapsed / rightAnimationDuration, 1);
  
  // イージング関数（ease-in-out）
  const easeProgress = progress < 0.5 
    ? 2 * progress * progress 
    : 1 - Math.pow(-2 * progress + 2, 2) / 2;
  
  // 各パラメータを補間
  for (let key in rightTargetParams) {
    if (rightStartParams[key] !== undefined) {
      rightCurrentParams[key] = lerp(rightStartParams[key], rightTargetParams[key], easeProgress);
    }
  }
  
  // アニメーション終了判定
  if (progress >= 1) {
    rightAnimationActive = false;
  }
}

// 実行ボタンの処理
function executeAction() {
  // アニメーション速度を取得
  const speedValue = parseFloat(document.getElementById("animationSpeed").value);
  rightAnimationDuration = speedValue * 1000; // 秒をミリ秒に変換
  
  // 現在の右側のパラメータを開始値として保存
  rightStartParams = { ...rightCurrentParams };
  
  // 左側のパラメータを目標値として設定
  rightTargetParams = { ...faceParams };
  
  // アニメーション開始
  rightAnimationStartTime = millis();
  rightAnimationActive = true;
}

// リセットボタンの処理
function resetAction() {
  // アニメーションを停止
  rightAnimationActive = false;
  
  // デフォルトパラメータに即座に戻す
  rightCurrentParams = {
    eyeOpenness: 1,
    pupilSize: 0.7,
    pupilAngle: 0,
    upperEyelidAngle: 0,
    upperEyelidCoverage: 0,
    lowerEyelidCoverage: 0,
    mouthCurve: 0,
    mouthHeight: 0,
    mouthWidth: 1
  };
}

// 次の未保存の感情に移動
function moveToNextUnsavedEmotion() {
  // HTMLのボタン順序と一致させる
  const emotionOrder = ['normal', 'surprise', 'anger', 'joy', 'sleepy', 'sad'];
  
  // 現在の感情のインデックスを取得
  const currentIndex = emotionOrder.indexOf(currentEmotion);
  
  // 次の未保存の感情を探す
  for (let i = 1; i <= emotionOrder.length; i++) {
    const nextIndex = (currentIndex + i) % emotionOrder.length;
    const nextEmotion = emotionOrder[nextIndex];
    
    if (!savedEmotions.has(nextEmotion)) {
      setEmotion(nextEmotion);
      break;
    }
  }
  
  // すべて保存済みの場合
  if (savedEmotions.size === emotionOrder.length) {
    alert('すべての感情のデータを保存しました！');
  }
}

// 保存済みボタンの状態を更新
function updateSavedButtonStates() {
  const allButtons = document.querySelectorAll('.emotion-btn');
  allButtons.forEach(btn => {
    const emotionName = btn.getAttribute('onclick').match(/setEmotion\('(\w+)'\)/)[1];
    if (savedEmotions.has(emotionName)) {
      btn.classList.add('saved');
      btn.disabled = true;
    }
  });
}

// ユニークな被験者IDを生成
function generateUniqueSubjectId() {
  // UUID v4を生成する関数
  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }
  
  // UUID形式のIDを生成
  const subjectId = generateUUID();
  
  // 入力フィールドに設定
  document.getElementById('subject-id').value = subjectId;
  
  // 読み取り専用にする（ユーザーが変更できないように）
  document.getElementById('subject-id').readOnly = true;
}

// アニメーション顔を描画
function drawAnimatedFace() {
  animatedCanvas.background(255, 235, 250);
  animatedCanvas.push();
  animatedCanvas.translate(animatedCanvas.width / 2, animatedCanvas.height / 2);
  
  // アニメーションを更新
  updateAnimation();
  
  // 描画コンテキストをアニメーションキャンバスに設定
  let originalCtx = setupContext(animatedCanvas);
  
  // 目を描画
  drawEyes();
  
  // 口を描画
  drawMouth();
  
  // コンテキストを復元
  restoreContext(originalCtx);
  
  animatedCanvas.pop();
}

// 保存ボタンの処理
function saveAction() {
  // 被験者IDを取得
  const subjectId = document.getElementById('subject-id').value.trim();
  
  if (!subjectId) {
    alert('被験者IDを入力してください。');
    return;
  }
  
  // 日本時間でタイムスタンプを生成
  const now = new Date();
  const jstOffset = 9 * 60; // JST is UTC+9
  const jstTime = new Date(now.getTime() + jstOffset * 60 * 1000);
  
  // YYYY-MM-DD HH:MM:SS.mmm形式でフォーマット
  const timestamp = jstTime.toISOString().replace('T', ' ').replace('Z', '');
  
  // アニメーション速度を取得
  const animationDuration = document.getElementById('animationSpeed').value;
  
  // 保存するデータの作成
  const data = {
    subject_id: subjectId,
    timestamp: timestamp,
    emotion_label: currentEmotion,
    animationDuration: animationDuration,
    eyeOpenness: rightCurrentParams.eyeOpenness,
    pupilSize: rightCurrentParams.pupilSize,
    pupilAngle: rightCurrentParams.pupilAngle,
    upperEyelidAngle: rightCurrentParams.upperEyelidAngle,
    upperEyelidCoverage: rightCurrentParams.upperEyelidCoverage,
    lowerEyelidCoverage: rightCurrentParams.lowerEyelidCoverage,
    mouthCurve: rightCurrentParams.mouthCurve,
    mouthHeight: rightCurrentParams.mouthHeight,
    mouthWidth: rightCurrentParams.mouthWidth
  };
  
  // サーバーにデータを送信
  fetch('http://localhost:3000/save-emotion-data', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(result => {
    if (result.success) {
      alert('データを保存しました。');
      
      // 保存した感情を記録
      savedEmotions.add(currentEmotion);
      
      // ボタンを無効化
      const allButtons = document.querySelectorAll('.emotion-btn');
      allButtons.forEach(btn => {
        if (btn.getAttribute('onclick').includes(currentEmotion)) {
          btn.classList.add('saved');
          btn.disabled = true;
        }
      });
      
      // 次の未保存の感情に自動的に移動
      moveToNextUnsavedEmotion();
    } else {
      alert('保存エラー: ' + result.message);
    }
  })
  .catch(error => {
    console.error('Error:', error);
    alert('サーバーへの接続エラー。サーバーが起動していることを確認してください。');
  });
}


// コンテキストを設定
function setupContext(canvas) {
  const original = {
    push: window.push,
    pop: window.pop,
    translate: window.translate,
    fill: window.fill,
    stroke: window.stroke,
    strokeWeight: window.strokeWeight,
    ellipse: window.ellipse,
    arc: window.arc,
    rotate: window.rotate,
    radians: window.radians,
    rect: window.rect,
    line: window.line,
    noFill: window.noFill,
    noStroke: window.noStroke,
    beginShape: window.beginShape,
    vertex: window.vertex,
    endShape: window.endShape,
    curveVertex: window.curveVertex,
    width: window.width,
    height: window.height,
    drawingContext: window.drawingContext,
    abs: window.abs,
    asin: window.asin,
    cos: window.cos,
    sin: window.sin,
    PI: window.PI,
    TWO_PI: window.TWO_PI,
    CLOSE: window.CLOSE
  };
  
  // キャンバスの関数に置き換え
  window.push = () => canvas.push();
  window.pop = () => canvas.pop();
  window.translate = (x, y) => canvas.translate(x, y);
  window.fill = (...args) => canvas.fill(...args);
  window.stroke = (...args) => canvas.stroke(...args);
  window.strokeWeight = (w) => canvas.strokeWeight(w);
  window.ellipse = (x, y, w, h) => canvas.ellipse(x, y, w, h);
  window.arc = (x, y, w, h, start, stop, mode) => canvas.arc(x, y, w, h, start, stop, mode);
  window.rotate = (angle) => canvas.rotate(angle);
  window.radians = (degrees) => canvas.radians(degrees);
  window.rect = (x, y, w, h) => canvas.rect(x, y, w, h);
  window.line = (x1, y1, x2, y2) => canvas.line(x1, y1, x2, y2);
  window.noFill = () => canvas.noFill();
  window.noStroke = () => canvas.noStroke();
  window.beginShape = () => canvas.beginShape();
  window.vertex = (x, y) => canvas.vertex(x, y);
  window.endShape = (mode) => canvas.endShape(mode);
  window.curveVertex = (x, y) => canvas.curveVertex(x, y);
  window.width = canvas.width;
  window.height = canvas.height;
  window.drawingContext = canvas.canvas.getContext('2d');
  
  return original;
}

// コンテキストを復元
function restoreContext(original) {
  Object.keys(original).forEach(key => {
    window[key] = original[key];
  });
}