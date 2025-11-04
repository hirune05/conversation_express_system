// 口の描画関数

function drawMouth() {
  push();
  // 目との相対位置を保ちながら配置
  let mouthY = height * 0.26; // 中心からの相対位置
  translate(0, mouthY);

  // 口を描画
  stroke(200, 0, 0);
  strokeWeight(4);
  noFill();

  // 口の基本的な幅と高さ
  let mouthWidth = 80 * faceParams.mouthWidth;
  let mouthHeight = 10;

  // 口角の上がり具合（-30〜30）
  let cornerLift = faceParams.mouthCurve;

  // 口の縦方向の開き具合を調整
  if (faceParams.mouthHeight > 0) {
    mouthHeight = mouthHeight + faceParams.mouthHeight * 40;
  }

  // ベジェ曲線で口を描画
  beginShape();

  // 左端
  let leftX = -mouthWidth / 2;
  let leftY = -cornerLift;

  // 右端
  let rightX = mouthWidth / 2;
  let rightY = -cornerLift;

  // 中央
  let centerY = cornerLift * 0.5;

  // 上唇
  vertex(leftX, leftY);
  bezierVertex(
    leftX + mouthWidth * 0.25,
    centerY - mouthHeight / 2,
    rightX - mouthWidth * 0.25,
    centerY - mouthHeight / 2,
    rightX,
    rightY
  );

  // 口が開いている場合は下唇も描く
  if (faceParams.mouthHeight > 0) {
    bezierVertex(
      rightX - mouthWidth * 0.25,
      centerY + mouthHeight / 2,
      leftX + mouthWidth * 0.25,
      centerY + mouthHeight / 2,
      leftX,
      leftY
    );
    endShape(CLOSE);

    // 口の中を塗りつぶす
    fill(200, 0, 0);
    noStroke();
    beginShape();
    vertex(leftX, leftY);
    bezierVertex(
      leftX + mouthWidth * 0.25,
      centerY - mouthHeight / 2,
      rightX - mouthWidth * 0.25,
      centerY - mouthHeight / 2,
      rightX,
      rightY
    );
    bezierVertex(
      rightX - mouthWidth * 0.25,
      centerY + mouthHeight / 2,
      leftX + mouthWidth * 0.25,
      centerY + mouthHeight / 2,
      leftX,
      leftY
    );
    endShape(CLOSE);
  } else {
    endShape();
  }

  pop();
}