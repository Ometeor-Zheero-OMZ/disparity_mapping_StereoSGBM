use opencv::{
    calib3d::{StereoMatcher, StereoSGBM},
    core::{absdiff, min_max_loc, normalize, Mat, Ptr, Rect, Vector, Size, BORDER_DEFAULT, CV_8U, NORM_MINMAX},
    imgcodecs, imgproc, prelude::*,
    ximgproc::create_disparity_wls_filter,
};

// SGBM (Semi-Global Block Matching) パラメータ設定
const MIN_DISPARITY: i32 = 0; // 最小視差
const NUM_DISPARITIES: i32 = 256; // 視差の範囲
const BLOCK_SIZE: i32 = 5; // ブロックサイズ
const P1: i32 = 16 * BLOCK_SIZE * BLOCK_SIZE; // P1 (平滑化パラメータ)
const P2: i32 = 64 * BLOCK_SIZE * BLOCK_SIZE; // P2 (平滑化パラメータ)
const DISP_12_MAX_DIFF: i32 = 1; // 視差の最大差
const PRE_FILTER_CAP: i32 = 31; // 前処理キャッピング
const UNIQUENESS_RATIO: i32 = 5; // 一意性比率
const SPECKLE_WINDOW_SIZE: i32 = 25; // スペックル検出ウィンドウサイズ
const SPECKLE_RANGE: i32 = 4; // スペックルの範囲
const MODE: i32 = opencv::calib3d::StereoSGBM_MODE_HH; // モード設定

// WLSフィルタのパラメータ
const LAMBDA: f64 = 500.0; // WLSフィルタのスムージングパラメータ
const FILTER_ALPHA: f64 = 1.0; // フィルタのスケール
const FILTER_BETA: f64 = 0.0; // フィルタのバイアス

// ブラー処理のパラメータ
const K_SIZE_HEIGHT: i32 = 3; // カーネル高さ
const K_SIZE_WIDTH: i32 = 3; // カーネル幅
const SIGMA_X: f64 = 0.0; // X方向のガウスぼかしの標準偏差
const SIGMA_Y: f64 = 0.0; // Y方向のガウスぼかしの標準偏差

// エッジ検出のパラメータ
const THRESHOLD_1: f64 = 100.0;
const THRESHOLD_2: f64 = 200.0;
const APERTURE_SIZE: i32 = 3;

fn main() -> opencv::Result<()> {
    // 入力画像の読み込み
    let left = imgcodecs::imread("./img/left.png", imgcodecs::IMREAD_GRAYSCALE)?;
    let right = imgcodecs::imread("./img/right.png", imgcodecs::IMREAD_GRAYSCALE)?;

    // StereoSGBMの初期化
    let mut sgbm = StereoSGBM::create(
        MIN_DISPARITY,
        NUM_DISPARITIES,
        BLOCK_SIZE,
        P1,
        P2,
        DISP_12_MAX_DIFF,
        PRE_FILTER_CAP,
        UNIQUENESS_RATIO,
        SPECKLE_WINDOW_SIZE,
        SPECKLE_RANGE,
        MODE,
    )?;

    // 視差マップの生成
    let mut disparity_sgbm = Mat::default();                   // 左画像からの視差マップ
    let mut disparity_right = Mat::default();                  // 右画像からの視差マップ
    sgbm.compute(&left, &right, &mut disparity_sgbm)?;              // 左から右の視差計算
    sgbm.compute(&right, &left, &mut disparity_right)?; // 右から左の視差計算

    // WLSフィルタの適用（視差の滑らかな補正）
    let sgbm_ptr: Ptr<StereoMatcher> = Ptr::from(sgbm);
    let mut wls_filter = create_disparity_wls_filter(sgbm_ptr)?;
    wls_filter.set_lambda(LAMBDA)?;

    let mut filtered_disparity = Mat::default(); // フィルタリングされた視差
    wls_filter.filter(
        &disparity_sgbm,
        &left,
        &mut filtered_disparity,
        &disparity_right,
        Rect::default(),
        &right,
    )?;

    // 浮動小数点型に変換
    let mut disparity_float = Mat::default();
    filtered_disparity.convert_to(&mut disparity_float, opencv::core::CV_32F, FILTER_ALPHA, FILTER_BETA)?;

    // 視差マップのぼかし処理（ガウシアンぼかし）
    let mut disparity_blurred = Mat::default();
    // imgproc::median_blur(&disparity_float, &mut disparity_blurred, K_SIZE)?; // 必要であればメディアンぼかしを使用
    imgproc::gaussian_blur(&disparity_float, &mut disparity_blurred, Size::new(K_SIZE_WIDTH, K_SIZE_HEIGHT), SIGMA_X, SIGMA_Y, BORDER_DEFAULT)?;

    // ゼロ行列作成
    let size = disparity_blurred.size()?; // 視差画像のサイズ取得
    let width = size.width;
    let height = size.height;
    
    // ゼロ行列（背景用）
    let mat_zeros = Mat::zeros(height, width, disparity_blurred.typ())?;
    
    // 視差マップとゼロ行列の差分を計算
    let disparity_blurred_clone = disparity_blurred.clone();
    absdiff(&disparity_blurred_clone, &mat_zeros, &mut disparity_blurred)?;

    // 視差範囲の確認
    let mut min_val = 0.0;
    let mut max_val = 0.0;
    min_max_loc(&disparity_blurred, Some(&mut min_val), Some(&mut max_val), None, None, &Mat::default())?;
    println!("Disparity range: min = {}, max = {}", min_val, max_val);

    // 正規化（視差画像を0~255の範囲にスケーリング）
    let mut disparity_visual = Mat::default();
    normalize(
        &disparity_blurred,
        &mut disparity_visual,
        min_val,
        max_val,
        NORM_MINMAX,
        CV_8U,
        &Mat::default(),
    )?;

    // Cannyエッジ検出（視差画像のエッジを抽出）
    let mut edges = Mat::default();
    imgproc::canny(&disparity_visual, &mut edges, THRESHOLD_1, THRESHOLD_2, APERTURE_SIZE, false)?;

    imgcodecs::imwrite("./out/disparity_map.jpg", &disparity_visual, &Vector::new())?;

    Ok(())
}
