// 目の描画関数

function drawEyes() {
  let eyeSpacing = width * 0.42; // 画面の横幅の40%
  let eyeSize = 120;

  // 目を画面の中央に配置
  let eyeY = 0; // 中心からの相対位置

  // 左目
  push();
  translate(-eyeSpacing / 2, eyeY);
  drawEye(eyeSize, true); // 左目
  pop();

  // 右目
  push();
  translate(eyeSpacing / 2, eyeY);
  drawEye(eyeSize, false); // 右目
  pop();
}

function drawEye(size, isLeft) {
  push();

  // 白目部分
  fill(240, 255, 240);
  stroke(0);
  strokeWeight(2);
  ellipse(0, 0, size, size);

  // 目の開き具合に応じて描画を変更
  if (faceParams.eyeOpenness <= 0.1) {
    // 完全に閉じた目（弧として描画）
    fill(255, 235, 250); // 背景と同じ色で上半分を塗りつぶす
    noStroke();
    arc(0, 0, size + 4, size + 4, PI, TWO_PI);

    // 閉じた目の線（弧）
    noFill();
    stroke(0);
    strokeWeight(3);
    arc(0, 0, size * 0.7, size * 0.4, 0, PI);
  } else {
    // 瞳（黒目）- eyeOpennessに応じて縦方向のサイズを調整
    push();
    let pupilAngle = isLeft ? faceParams.pupilAngle : -faceParams.pupilAngle;
    rotate(radians(pupilAngle));

    fill(0);
    noStroke();
    let pupilWidth = size * 0.7 * faceParams.pupilSize;
    let pupilHeight = pupilWidth * faceParams.eyeOpenness;
    ellipse(0, 0, pupilWidth, pupilHeight);

    // 光の反射（目が開いているときのみ）
    if (faceParams.eyeOpenness > 0.3) {
      fill(255);
      let highlightSize = pupilWidth * 0.2;
      ellipse(0, -pupilHeight * 0.2, highlightSize, highlightSize);
    }

    pop();
  }

  // 上瞼の効果（目の中に線を引いて上部を塗りつぶす）
  if (abs(faceParams.upperEyelidAngle) > 0 || faceParams.upperEyelidCoverage > 0) {
    // 上瞼の線の位置を計算（覆い具合のみで決定）
    let eyelidY = -size * 0.5 + size * 1.2 * faceParams.upperEyelidCoverage; // 覆い具合による位置

    // クリッピングマスクで目の円の中だけ描画
    push();
    drawingContext.save();

    // 円形のクリッピングパスを作成（少し内側にして輪郭を残す）
    drawingContext.beginPath();
    drawingContext.arc(0, 0, size / 2 - 1, 0, TWO_PI);
    drawingContext.clip();

    // 上瞼の角度に応じて傾いた上瞼を描画
    push();
    let lidAngle = isLeft ? faceParams.upperEyelidAngle : -faceParams.upperEyelidAngle;
    rotate(radians(lidAngle * 0.5)); // 上瞼の角度

    // 上瞼の線より上を白で塗りつぶす
    fill(240, 255, 240); // 白目と同じ色
    noStroke();
    rect(-size, -size, size * 2, size + eyelidY);

    // 上瞼の線を描く
    stroke(0);
    strokeWeight(2);
    line(-size, eyelidY, size, eyelidY);

    pop();

    drawingContext.restore();
    pop();
  }

  // 下瞼を描画
  if (faceParams.lowerEyelidCoverage > 0) {
    push();

    // 下瞼の位置をパラメータで調整
    let lowerEyelidY = size * 0.5 - size * faceParams.lowerEyelidCoverage;
    let arcHeight = size * 0.2 + size * faceParams.lowerEyelidCoverage * 0.3;

    // まず目の輪郭の下部分をピンクで上書き（下瞼の弧から下の部分）
    stroke(255, 235, 250); // 背景と同じピンク色
    strokeWeight(3);
    noFill();

    // 目の円の下部分を描画（下瞼の位置より少し上から下）
    let startAngle = asin((lowerEyelidY - 5) / (size / 2));
    if (!isNaN(startAngle)) {
      arc(0, 0, size, size, startAngle, PI - startAngle);
    }

    // クリッピングマスクを設定して目の中だけ描画
    drawingContext.save();
    drawingContext.beginPath();
    drawingContext.arc(0, 0, size / 2 - 1, 0, TWO_PI);
    drawingContext.clip();

    // 下瞼から下の部分をピンクで塗りつぶす
    fill(255, 235, 250); // 背景と同じピンク色
    noStroke();
    beginShape();
    // 弧の開始点（左端）
    vertex(-size * 0.35, lowerEyelidY);
    // 弧に沿って描画
    for (let angle = PI; angle <= TWO_PI; angle += 0.1) {
      let x = cos(angle) * size * 0.35;
      let y = lowerEyelidY + sin(angle) * arcHeight * 0.5;
      vertex(x, y);
    }
    // 下端まで塗りつぶす
    vertex(size * 0.35, size);
    vertex(-size * 0.35, size);
    endShape(CLOSE);

    // 下瞼の線を描画（上向きの弧）
    stroke(0);
    strokeWeight(2);
    noFill();
    arc(0, lowerEyelidY, size * 0.7, arcHeight, PI, TWO_PI);

    drawingContext.restore();
    pop();
  }

  pop();
}