// パラメータマッピング定義
const parameterMappings = {
  // 目関連
  eyeOpenness: { min: 0, max: 1 }, // すでに0-1
  pupilSize: { min: 0.3, max: 1.0 },
  pupilAngle: { min: -30, max: 30 },
  upperEyelidAngle: { min: -30, max: 30 },
  upperEyelidCoverage: { min: 0, max: 0.3 },
  lowerEyelidCoverage: { min: 0, max: 0.3 },
  
  // 口関連
  mouthCurve: { min: -30, max: 60 },
  mouthHeight: { min: 0, max: 3 },
  mouthWidth: { min: 0.5, max: 4 }
};

// 正規化された値(0-1)を実際の値に変換
function mapNormalizedToActual(paramName, normalizedValue) {
  if (!parameterMappings[paramName]) {
    console.warn(`Unknown parameter: ${paramName}`);
    return normalizedValue;
  }
  
  const { min, max } = parameterMappings[paramName];
  return min + (max - min) * normalizedValue;
}

// 実際の値を正規化された値(0-1)に変換
function mapActualToNormalized(paramName, actualValue) {
  if (!parameterMappings[paramName]) {
    console.warn(`Unknown parameter: ${paramName}`);
    return actualValue;
  }
  
  const { min, max } = parameterMappings[paramName];
  return (actualValue - min) / (max - min);
}

// JSONフォーマットから実際のパラメータに変換
function convertJSONToParams(jsonData) {
  const actualParams = {};
  
  // 各パラメータを正規化された値から実際の値に変換
  for (const [key, normalizedValue] of Object.entries(jsonData.parameters)) {
    // キー名を現在のfaceParamsに合わせる
    const mappedKey = mapParameterKey(key);
    actualParams[mappedKey] = mapNormalizedToActual(mappedKey, normalizedValue);
  }
  
  return {
    emotion: jsonData.emotion,
    intensity: jsonData.intensity,
    params: actualParams,
    duration: jsonData.transition.duration
  };
}

// JSONのキー名を現在のシステムのキー名にマップ
function mapParameterKey(jsonKey) {
  const keyMapping = {
    'upper_eyelid_angle': 'upperEyelidAngle',
    'eye_openness': 'eyeOpenness',
    'pupil_size': 'pupilSize',
    'pupil_angle': 'pupilAngle',
    'mouth_curve': 'mouthCurve',
    'mouth_height': 'mouthHeight',
    'mouth_width': 'mouthWidth',
    'upper_eyelid_coverage': 'upperEyelidCoverage',
    'lower_eyelid_coverage': 'lowerEyelidCoverage'
  };
  
  return keyMapping[jsonKey] || jsonKey;
}

// 既存の感情プリセットを正規化された形式に変換
function convertEmotionToNormalizedJSON(emotionName, emotionData) {
  const normalizedParams = {};
  
  // 実装に必要なパラメータのみを変換
  normalizedParams.eye_openness = mapActualToNormalized('eyeOpenness', emotionData.eyeOpenness);
  normalizedParams.upper_eyelid_angle = mapActualToNormalized('upperEyelidAngle', emotionData.upperEyelidAngle);
  normalizedParams.pupil_size = mapActualToNormalized('pupilSize', emotionData.pupilSize);
  normalizedParams.pupil_angle = mapActualToNormalized('pupilAngle', emotionData.pupilAngle);
  normalizedParams.mouth_curve = mapActualToNormalized('mouthCurve', emotionData.mouthCurve);
  normalizedParams.mouth_height = mapActualToNormalized('mouthHeight', emotionData.mouthHeight);
  normalizedParams.mouth_width = mapActualToNormalized('mouthWidth', emotionData.mouthWidth);
  normalizedParams.upper_eyelid_coverage = mapActualToNormalized('upperEyelidCoverage', emotionData.upperEyelidCoverage);
  normalizedParams.lower_eyelid_coverage = mapActualToNormalized('lowerEyelidCoverage', emotionData.lowerEyelidCoverage);
  
  return {
    emotion: emotionName,
    intensity: 1.0,
    parameters: normalizedParams,
    transition: {
      duration: 0.5
    }
  };
}