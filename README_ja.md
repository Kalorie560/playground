# 🚀 宇宙船タイタニック ディープラーニングソリューション

Kaggle宇宙船タイタニックコンペティション向けの完全な機械学習ソリューションです。

## クイックスタート

### ステップ1: ClearMLの設定（オプション）
`config.yaml`を編集してClearML APIキーを追加します：

```yaml
clearml:
  api:
    access_key: "YOUR_ACCESS_KEY_HERE"
    secret_key: "YOUR_SECRET_KEY_HERE"
```

### ステップ2: データの準備
Kaggleコンペティションファイルをプロジェクトルートに配置します：
```
playground/
├── train.csv     ← 訓練データ
├── test.csv      ← テストデータ
└── ...
```

> ⚠️ **データファイルがない場合？** システムが自動的にテスト用のサンプルデータを生成します。

### ステップ3: モデルの学習
学習スクリプトを実行してモデルを生成します：

```bash
python train.py
```

これにより、訓練済みニューラルネットワークの`best_model.pth`が作成されます。

### ステップ4: ウェブアプリケーションの使用
インタラクティブなウェブインターフェースを起動します：

```bash
python run_app.py
```

ブラウザで http://localhost:8501 を開き、乗客の特徴量を入力してリアルタイム予測を取得します。

## データ前処理

前処理パイプラインは自動的に以下を実行します：

1. **欠損値の処理**
   - 数値特徴量：中央値で補完
   - カテゴリ特徴量：最頻値で補完
   - ブール特徴量：Falseで補完

2. **特徴エンジニアリング**
   - 客室文字列からデッキ、番号、サイドを抽出
   - PassengerIdから家族サイズとグループ特徴量を作成
   - 総支出額と支出比率を計算
   - 年齢グループと相互作用特徴量を生成

3. **エンコーディングとスケーリング**
   - カテゴリ変数をラベルエンコード
   - StandardScalerを使用して数値特徴量を標準化
   - ブール変数を整数に変換

## ニューラルネットワーク構造

モデルは多層フィードフォワードニューラルネットワークです：

```
入力層（特徴量数に依存）
    ↓
隠れ層1：512ニューロン + バッチ正規化 + Swish + ドロップアウト(0.2)
    ↓
隠れ層2：256ニューロン + バッチ正規化 + Swish + ドロップアウト(0.3)
    ↓
隠れ層3：128ニューロン + バッチ正規化 + Swish + ドロップアウト(0.4)
    ↓
隠れ層4：64ニューロン + バッチ正規化 + Swish + ドロップアウト(0.5)
    ↓
出力層：1ニューロン（二値分類）
```

**主な特徴：**
- **活性化関数**：Swish (x * sigmoid(x)) で滑らかな勾配を実現
- **正則化**：バッチ正規化と段階的ドロップアウト
- **残差接続**：勾配フローを改善するスキップ接続
- **重み初期化**：ReLU系活性化関数用のHe初期化

## ハイパーパラメータ

`config.yaml`の現在の最適化設定：

```yaml
model:
  hidden_sizes: [512, 256, 128, 64]
  dropout_rates: [0.2, 0.3, 0.4, 0.5]
  activation: "swish"
  use_batch_norm: true
  use_residual: true

training:
  batch_size: 32
  learning_rate: 0.0005
  epochs: 150
  optimizer: "adamw"
  weight_decay: 0.0005
  
  loss:
    type: "label_smoothing"
    smoothing: 0.1
  
  scheduler:
    type: "cosine"
  
  early_stopping:
    patience: 20
    min_delta: 0.00005
```

**学習戦略：**
- **オプティマイザ**：正則化のためのweight decayを持つAdamW
- **損失関数**：過信頼を防ぐラベルスムージングBCE
- **学習率スケジューラ**：滑らかな収束のためのコサインアニーリング
- **早期停止**：patience ベースの監視で過学習を防止

## インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# オプション：実験追跡のためのClearMLセットアップ
python setup_clearml.py
```

## 使用例

### カスタム設定での学習
```bash
# 設定をコピーして変更
cp config.yaml my_config.yaml
# my_config.yamlを設定に合わせて編集
python train.py --config my_config.yaml
```

### 予測の実行
```bash
# テストセットで予測を生成
python predict.py
```

### ウェブアプリケーション
```bash
# インタラクティブインターフェースを起動
python run_app.py
# またはシェルスクリプトを使用
./run_app.sh
```

## ファイル概要

- `config.yaml` - すべてのハイパーパラメータと設定
- `train.py` - ClearML統合を含むメイン学習パイプライン
- `model.py` - ニューラルネットワーク構造
- `data_preprocessing.py` - データ読み込みと特徴エンジニアリング
- `predict.py` - テスト予測の生成
- `app.py` - Streamlitウェブアプリケーション
- `run_app.py` - ウェブアプリ起動スクリプト

## 要件

- Python 3.7+
- PyTorch 2.0+
- Streamlit 1.28+
- scikit-learn 1.3+
- ClearML 1.14+（オプション）

完全なリストは`requirements.txt`を参照してください。