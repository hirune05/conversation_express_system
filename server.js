const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// CSVファイルのパス
const CSV_FILE_PATH = path.join(__dirname, 'emotion_data.csv');

// CSVヘッダー
const CSV_HEADERS = 'subject_id,timestamp,emotion_label,animationDuration,eyeOpenness,pupilSize,pupilAngle,upperEyelidAngle,upperEyelidCoverage,lowerEyelidCoverage,mouthCurve,mouthHeight,mouthWidth\n';

// CSVファイルの初期化（存在しない場合）
if (!fs.existsSync(CSV_FILE_PATH)) {
  fs.writeFileSync(CSV_FILE_PATH, CSV_HEADERS);
}

// データ保存エンドポイント
app.post('/save-emotion-data', (req, res) => {
  try {
    const data = req.body;
    
    // CSV行を作成
    const csvLine = [
      data.subject_id,
      data.timestamp,
      data.emotion_label,
      data.animationDuration,
      data.eyeOpenness,
      data.pupilSize,
      data.pupilAngle,
      data.upperEyelidAngle,
      data.upperEyelidCoverage,
      data.lowerEyelidCoverage,
      data.mouthCurve,
      data.mouthHeight,
      data.mouthWidth
    ].join(',') + '\n';
    
    // ファイルに追記
    fs.appendFileSync(CSV_FILE_PATH, csvLine);
    
    res.json({ success: true, message: 'データを保存しました' });
  } catch (error) {
    console.error('Error saving data:', error);
    res.status(500).json({ success: false, message: 'データ保存エラー' });
  }
});

// サーバー起動
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`CSV file will be saved at: ${CSV_FILE_PATH}`);
});