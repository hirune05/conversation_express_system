// UI制御関数

// アニメーション用の変数
let animationStartTime = null;
let animationDuration = 1000; // 1秒（デフォルト）
let animationActive = false;
let startParams = {};
let targetParams = {};

function setEmotion(emotionName) {
  // 保存済みの感情は選択できないようにする
  if (window.savedEmotions && window.savedEmotions.has(emotionName)) {
    return;
  }
  
  if (emotions[emotionName]) {
    // 現在の感情を更新
    currentEmotion = emotionName;
    
    // パラメータを即座に変更（アニメーションなし）
    Object.assign(faceParams, emotions[emotionName]);
    
    // スライダーの値も更新
    for (let key in emotions[emotionName]) {
      if (document.getElementById(key)) {
        document.getElementById(key).value = emotions[emotionName][key];
        document.getElementById(key + "Value").textContent = emotions[emotionName][key].toFixed(2);
      }
    }
  }
  
  // 感情の説明を表示
  const descriptionDiv = document.getElementById('emotion-description');
  if (emotionDescriptions[emotionName]) {
    descriptionDiv.textContent = emotionDescriptions[emotionName];
  } else {
    descriptionDiv.textContent = "感情を選択してください";
  }
  
  // ボタンのアクティブ状態を更新
  const allButtons = document.querySelectorAll('.emotion-btn');
  allButtons.forEach(btn => btn.classList.remove('active'));
  
  // クリックされたボタンを探してアクティブにする
  allButtons.forEach(btn => {
    if (btn.getAttribute('onclick').includes(emotionName)) {
      btn.classList.add('active');
    }
  });
}

// パラメータとdurationを指定してアニメーションする関数
function animateToEmotion(params, duration = 0.5) {
  // 現在のパラメータを開始値として保存
  startParams = { ...faceParams };
  
  // 目標のパラメータを設定
  targetParams = params;
  
  // アニメーション時間を設定（秒をミリ秒に変換）
  animationDuration = duration * 1000;
  
  // アニメーション開始
  animationStartTime = millis();
  animationActive = true;
}

// window.animateToEmotionとして公開
window.animateToEmotion = animateToEmotion;

// アニメーション更新関数（draw関数から呼び出される）
function updateAnimation() {
  if (!animationActive) return;
  
  const currentTime = millis();
  const elapsed = currentTime - animationStartTime;
  const progress = Math.min(elapsed / animationDuration, 1);
  
  // イージング関数（ease-in-out）
  const easeProgress = progress < 0.5 
    ? 2 * progress * progress 
    : 1 - Math.pow(-2 * progress + 2, 2) / 2;
  
  // 各パラメータを補間
  for (let key in targetParams) {
    if (startParams[key] !== undefined) {
      // 線形補間
      faceParams[key] = lerp(startParams[key], targetParams[key], easeProgress);
      
      // スライダーの値も更新
      if (document.getElementById(key)) {
        document.getElementById(key).value = faceParams[key];
        document.getElementById(key + "Value").textContent = faceParams[key].toFixed(2);
      } else {
        // 古いパラメータ名から新しいパラメータ名へのマッピング
        const paramMapping = {
          'eyelidAngle': 'upperEyelidAngle',
          'eyelidStrength': 'upperEyelidCoverage',
          'lowerEyelidStrength': 'lowerEyelidCoverage'
        };
        const mappedKey = paramMapping[key];
        if (mappedKey && document.getElementById(mappedKey)) {
          faceParams[mappedKey] = lerp(startParams[key], targetParams[key], easeProgress);
          document.getElementById(mappedKey).value = faceParams[mappedKey];
          document.getElementById(mappedKey + "Value").textContent = faceParams[mappedKey].toFixed(2);
        }
      }
    }
  }
  
  // アニメーション終了判定
  if (progress >= 1) {
    animationActive = false;
  }
}

function updateParams() {
  // 目関連のパラメータ
  faceParams.eyeOpenness = parseFloat(
    document.getElementById("eyeOpenness").value
  );
  faceParams.pupilSize = parseFloat(document.getElementById("pupilSize").value);
  faceParams.pupilAngle = parseFloat(
    document.getElementById("pupilAngle").value
  );
  faceParams.upperEyelidAngle = parseFloat(
    document.getElementById("upperEyelidAngle").value
  );
  faceParams.upperEyelidCoverage = parseFloat(
    document.getElementById("upperEyelidCoverage").value
  );
  faceParams.lowerEyelidCoverage = parseFloat(
    document.getElementById("lowerEyelidCoverage").value
  );
  
  // 口関連のパラメータ
  faceParams.mouthCurve = parseFloat(
    document.getElementById("mouthCurve").value
  );
  faceParams.mouthHeight = parseFloat(
    document.getElementById("mouthHeight").value
  );
  faceParams.mouthWidth = parseFloat(
    document.getElementById("mouthWidth").value
  );

  // 値の表示を更新
  // 目関連のパラメータ
  document.getElementById("eyeOpennessValue").textContent =
    faceParams.eyeOpenness;
  document.getElementById("pupilSizeValue").textContent = faceParams.pupilSize;
  document.getElementById("pupilAngleValue").textContent =
    faceParams.pupilAngle;
  document.getElementById("upperEyelidAngleValue").textContent =
    faceParams.upperEyelidAngle;
  document.getElementById("upperEyelidCoverageValue").textContent =
    faceParams.upperEyelidCoverage;
  document.getElementById("lowerEyelidCoverageValue").textContent =
    faceParams.lowerEyelidCoverage;
  
  // 口関連のパラメータ
  document.getElementById("mouthCurveValue").textContent =
    faceParams.mouthCurve;
  document.getElementById("mouthHeightValue").textContent =
    faceParams.mouthHeight;
  document.getElementById("mouthWidthValue").textContent =
    faceParams.mouthWidth;
}

// アニメーション速度更新関数
function updateAnimationSpeed() {
  const speedValue = parseFloat(document.getElementById("animationSpeed").value);
  animationDuration = speedValue * 1000; // 秒をミリ秒に変換
  document.getElementById("animationSpeedValue").textContent = speedValue;
}

// イベントリスナーの設定
function setupUIListeners() {
  // アニメーション速度
  document
    .getElementById("animationSpeed")
    .addEventListener("input", updateAnimationSpeed);
    
  // 目関連のパラメータ
  document
    .getElementById("eyeOpenness")
    .addEventListener("input", updateParams);
  document.getElementById("pupilSize").addEventListener("input", updateParams);
  document.getElementById("pupilAngle").addEventListener("input", updateParams);
  document
    .getElementById("upperEyelidAngle")
    .addEventListener("input", updateParams);
  document
    .getElementById("upperEyelidCoverage")
    .addEventListener("input", updateParams);
  document
    .getElementById("lowerEyelidCoverage")
    .addEventListener("input", updateParams);
  
  // 口関連のパラメータ
  document.getElementById("mouthCurve").addEventListener("input", updateParams);
  document
    .getElementById("mouthHeight")
    .addEventListener("input", updateParams);
  document.getElementById("mouthWidth").addEventListener("input", updateParams);
}