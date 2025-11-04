// 顔のパラメータ定義
let faceParams = {
  // 目関連のパラメータ
  eyeOpenness: 1, // 目の開き具合（0=閉じる, 1=完全に開く）
  pupilSize: 0.7, // 瞳の大きさ（0.3-1.0）
  pupilAngle: 0, // 瞳の回転角度（-30〜30）
  upperEyelidAngle: 0, // 上瞼の傾き角度（-30〜30）
  upperEyelidCoverage: 0, // 上瞼の覆い具合（0-0.3）
  lowerEyelidCoverage: 0, // 下瞼の覆い具合（0-0.3）

  // 口関連のパラメータ
  mouthCurve: 0, // 口角の上下（-30〜60）
  mouthHeight: 0, // 口の開き具合（0-3）
  mouthWidth: 1, // 口の横幅倍率（0.5-4）
};

// 感情の説明
const emotionDescriptions = {
  normal: "ユーザーからの指示を待っている、ロボットの基本の顔。",
  surprise: "ユーザーから予想外の秘密を打ち明けられた、ロボットの驚いた顔。",
  anger: "ゲームでユーザーにズルをされた時の、ロボットの不満な顔。",
  joy: "ユーザーに「ありがとう」と褒められた時の、ロボットの嬉しい顔。",
  sleepy: "タスクが終わり、充電する時の「おやすみ」の顔。",
  sad: "一人で留守番することになった時の、ロボットの寂しそうな顔。",
};

// 感情パラメータの定義
const emotions = {
  normal: {
    eyeOpenness: 1,
    pupilSize: 0.7,
    pupilAngle: 0,
    upperEyelidAngle: 0,
    upperEyelidCoverage: 0,
    lowerEyelidCoverage: 0,
    mouthCurve: 0,
    mouthHeight: 0,
    mouthWidth: 1,
  },
  surprise: {
    eyeOpenness: 1,
    pupilSize: 0.65,
    pupilAngle: 0,
    upperEyelidAngle: -20,
    upperEyelidCoverage: 0,
    lowerEyelidCoverage: 0,
    mouthCurve: -5,
    mouthHeight: 1.5,
    mouthWidth: 1.2,
  },
  anger: {
    eyeOpenness: 0.7,
    pupilSize: 0.5,
    pupilAngle: 0,
    upperEyelidAngle: 25,
    upperEyelidCoverage: 0.2,
    lowerEyelidCoverage: 0.1,
    mouthCurve: -15,
    mouthHeight: 0.3,
    mouthWidth: 1.5,
  },
  joy: {
    eyeOpenness: 0.8,
    pupilSize: 0.8,
    pupilAngle: 0,
    upperEyelidAngle: -10,
    upperEyelidCoverage: 0,
    lowerEyelidCoverage: 0.1,
    mouthCurve: 34,
    mouthHeight: 0.5,
    mouthWidth: 1.8,
  },
  sleepy: {
    eyeOpenness: 0.3,
    pupilSize: 0.6,
    pupilAngle: 0,
    upperEyelidAngle: 10,
    upperEyelidCoverage: 0.25,
    lowerEyelidCoverage: 0.1,
    mouthCurve: -5,
    mouthHeight: 0.2,
    mouthWidth: 0.8,
  },
  sad: {
    eyeOpenness: 0.6,
    pupilSize: 0.7,
    pupilAngle: 0,
    upperEyelidAngle: -15,
    upperEyelidCoverage: 0.15,
    lowerEyelidCoverage: 0,
    mouthCurve: -18,
    mouthHeight: 0,
    mouthWidth: 0.9,
  },
};
