# Spaceship Titanic ディープラーニングソリューション

Kaggle Spaceship Titanicコンペティション向けの包括的なディープラーニングソリューション。モジュラーアーキテクチャ、ClearML実験トラッキング、設定可能なハイパーパラメータ、インタラクティブなウェブアプリケーションを特徴としています。

## ✨ 機能

- **モジュラー設計**: データ処理、モデルアーキテクチャ、訓練、予測のための分離されたモジュールを持つ整理されたコード構造
- **ディープラーニング**: 設定可能なアーキテクチャを持つPyTorchベースのニューラルネットワーク
- **インタラクティブウェブアプリ**: リアルタイム予測のためのStreamlitベースウェブインターフェース
- **実験トラッキング**: 包括的な実験ログのためのClearML統合
- **設定管理**: 簡単なハイパーパラメータ調整のためのYAMLベース設定システム
- **特徴エンジニアリング**: キャビン情報抽出、家族サイズ特徴、支出パターンを含む高度な前処理
- **訓練パイプライン**: 早期停止、学習率スケジューリング、包括的メトリクスを含む完全な訓練ループ
- **再現性**: 再現可能な結果のためのシード管理と確定的な訓練

## 📁 プロジェクト構造

```
├── config.yaml                 # すべてのハイパーパラメータの設定ファイル
├── data_preprocessing.py        # データ読み込み、クリーニング、特徴エンジニアリング
├── model.py                    # ニューラルネットワークアーキテクチャとユーティリティ
├── train.py                    # ClearML統合を含む訓練パイプライン
├── predict.py                  # テストセット予測と提出ファイル生成
├── app.py                      # インタラクティブ予測のためのStreamlitウェブアプリケーション
├── run_app.py                  # ウェブアプリケーション起動用Pythonスクリプト
├── run_app.sh                  # ウェブアプリケーション起動用バッシュスクリプト
├── setup_clearml.py            # インタラクティブClearMLセットアップヘルパー
├── requirements.txt            # Python依存関係
├── README.md                   # メインドキュメント（英語）
├── README_ja.md                # 日本語ドキュメント
└── WEB_APP_README.md           # 詳細なウェブアプリケーションドキュメント
```

## 🚀 クイックスタート

### 1. インストール

```bash
pip install -r requirements.txt
```

### 2. データ準備

Kaggleコンペティションのデータファイルをプロジェクトディレクトリに配置してください：
- `train.csv` - 訓練データセット
- `test.csv` - テストデータセット

**注意**: データファイルが存在しない場合、ソリューションはデモンストレーション目的でサンプルデータを作成します。

### 3. 使用オプション

#### オプションA: インタラクティブウェブアプリケーション（推奨）

簡単な予測のためのウェブインターフェースを起動：

```bash
python run_app.py
```

またはバッシュスクリプトを使用：
```bash
./run_app.sh
```

またはStreamlitで直接実行：
```bash
streamlit run app.py
```

ウェブアプリは以下で利用可能： http://localhost:8501

#### オプションB: コマンドライン訓練と予測

**訓練：**
```bash
python train.py
```

**予測：**
```bash
python predict.py
```

## 🌐 ウェブアプリケーション

### 機能
- **ユーザーフレンドリーインターフェース**: 日本語サポートを備えた直感的なフォームベース入力
- **リアルタイム予測**: 入力特徴量に基づく即座の予測
- **詳細なガイダンス**: 各特徴量のヘルプテキストと説明
- **可視化された結果**: 詳細メトリクスを含む確率表示
- **入力検証**: 自動データ検証と前処理

### 入力特徴量
- **個人情報**: ホームプラネット、年齢、VIPステータス、冷凍睡眠
- **客室詳細**: デッキレベル、客室番号、ポート/スターボード側
- **目的地**: 最終目的地惑星
- **サービス支出**: ルームサービス、フードコート、ショッピングモール、スパ、VRデッキ

### 出力
- **輸送確率**: 輸送予測におけるモデルの信頼度
- **詳細メトリクス**: 予測信頼度の内訳
- **総支出**: すべてのサービス支出の要約
- **入力要約**: 入力されたすべてのデータのレビュー

## ⚙️ 設定

すべてのハイパーパラメータと設定は`config.yaml`で管理されています。主要なセクションは以下の通りです：

### データ設定
- 訓練/テストデータのファイルパス
- 検証分割比
- 再現性のためのランダムシード

### モデルアーキテクチャ
- 隠れ層サイズ: `[256, 128, 64]`
- ドロップアウト率: `[0.3, 0.4, 0.5]`
- 活性化関数: `relu`
- バッチ正規化: `true`

### 訓練パラメータ
- バッチサイズ: `64`
- 学習率: `0.001`
- エポック数: `100`
- オプティマイザ: `adam`
- 重み減衰: `0.0001`

### 早期停止とスケジューリング
- 早期停止の忍耐度: `15`
- 学習率スケジューラ: `step`
- ステップサイズ: `20`、ガンマ: `0.5`

### ClearML統合
- プロジェクト名: `Spaceship_Titanic`
- タスク名: `Neural_Network_Classifier`
- 自動フレームワーク接続

## 🔧 データ前処理機能

### 欠損値処理
- 数値特徴: 中央値で埋める
- カテゴリ特徴: 最頻値または'Unknown'で埋める
- ブール特徴: `False`で埋める

### 特徴エンジニアリング
- **キャビン情報**: キャビン文字列からデッキ、キャビン番号、サイドを抽出
- **家族特徴**: グループサイズ、一人旅行者、小/大グループ指標
- **支出パターン**: 総支出、すべての設備での支出指標
- **年齢グループ**: 子供、ティーン、若い成人、成人、高齢者にカテゴライズ
- **カテゴリエンコーディング**: カテゴリ変数のラベルエンコーディング
- **特徴スケーリング**: 数値特徴のStandardScaler

### 生成された特徴
- `Cabin`から`Deck`、`Cabin_num`、`Side`
- `PassengerId`から`Group_size`、`Is_solo`、`Is_small_group`、`Is_large_group`
- 設備支出から`Total_spending`、`Has_spending`
- `Age`から`Age_group`

## 🧠 モデルアーキテクチャ

### ニューラルネットワーク設計
- **入力層**: 前処理された特徴に基づいて自動でサイズ調整
- **隠れ層**: デフォルト`[256, 128, 64]`で設定可能なサイズ
- **正則化**: 設定可能な率のドロップアウト層
- **正規化**: オプションのバッチ正規化
- **活性化**: 設定可能な活性化関数（ReLU、LeakyReLU、ELU、GELU）
- **出力層**: 二値分類のための単一ニューロン

### 訓練機能
- **損失関数**: 安定した訓練のためのBCEWithLogitsLoss
- **オプティマイザ**: 設定可能なパラメータを持つAdam、SGD、RMSprop
- **学習率スケジューリング**: Step、Cosine、Exponential、Plateauスケジューラ
- **早期停止**: 設定可能な忍耐度でのオーバーフィッティング防止
- **モデルチェックポイント**: 最良モデルの自動保存

## 📊 ClearML実験トラッキング

ソリューションは包括的な実験トラッキングのためにClearMLと統合されています：

### クイックセットアップ

ClearML実験トラッキングをセットアップするには、インタラクティブセットアップヘルパーを実行してください：

```bash
python setup_clearml.py
```

このスクリプトは設定プロセスをガイドし、接続をテストします。

### 手動セットアップオプション

#### オプション1: clearml-initを使用（推奨）
1. [https://app.clear.ml/](https://app.clear.ml/)で無料アカウントを作成
2. Settings > Workspace Configurationに移動
3. 設定をコピー
4. `clearml-init`を実行して設定を貼り付け

#### オプション2: 環境変数
シェルプロファイル（`.bashrc`、`.zshrc`など）に追加：
```bash
export CLEARML_WEB_HOST=https://app.clear.ml
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=your_access_key
export CLEARML_API_SECRET_KEY=your_secret_key
```

#### オプション3: 設定ファイル
認証情報で`config.yaml`を更新：
```yaml
clearml:
  api:
    web_server: "https://app.clear.ml"
    api_server: "https://api.clear.ml"
    files_server: "https://files.clear.ml"
    access_key: "your_access_key"
    secret_key: "your_secret_key"
```

### ログされるメトリクス
- エポックごとの訓練・検証損失
- 検証精度、適合率、再現率、F1スコア、AUC
- 学習率の進行
- モデルアーキテクチャパラメータ
- データ統計（特徴数、データセットサイズ）
- 最終モデル性能メトリクス

### アーティファクト
- モデルチェックポイント
- 設定ファイル
- 訓練履歴

## 📋 使用例

### カスタム設定

カスタム設定ファイルを作成：

```yaml
# custom_config.yaml
model:
  hidden_sizes: [512, 256, 128]
  dropout_rates: [0.2, 0.3, 0.4]
  activation: "leaky_relu"

training:
  batch_size: 128
  learning_rate: 0.0005
  epochs: 150
```

カスタム設定で訓練を実行：

```bash
python train.py --config custom_config.yaml
```

### カスタムモデルでの予測

```bash
python predict.py --model custom_model.pth --output custom_submission.csv
```

### データ前処理のみ

```python
from data_preprocessing import SpaceshipDataProcessor

processor = SpaceshipDataProcessor()
train_df, test_df = processor.load_data('train.csv', 'test.csv')
processed_data = processor.preprocess(train_df, test_df)
```

### モデル作成と訓練

```python
from model import create_model
from train import SpaceshipTrainer

# カスタム訓練
trainer = SpaceshipTrainer('config.yaml')
history, metrics = trainer.train()
```

### ウェブアプリケーション統合

```python
from app import SpaceshipWebPredictor

# 予測器インスタンスを作成
predictor = SpaceshipWebPredictor()

# ユーザー入力から予測を作成
user_inputs = {
    'home_planet': 'Earth',
    'age': 25,
    'vip': False,
    # ... その他の特徴量
}
probability = predictor.predict(user_inputs)
```

## 📈 性能監視

### 検証メトリクス
- **精度**: 全体の分類精度
- **適合率**: 輸送された乗客の真陽性率
- **再現率**: 輸送された乗客の感度
- **F1スコア**: 適合率と再現率の調和平均
- **AUC**: ROC曲線下面積

### 訓練監視
- リアルタイム損失トラッキング
- 学習率スケジューリング
- 検証損失に基づく早期停止
- 最良モデルの自動保存

## 🔧 高度な機能

### 再現性
- 設定可能なランダムシード
- 確定的な訓練オプション
- 一貫したデータ分割

### ハードウェアサポート
- 自動デバイス検出（CPU/CUDA/MPS）
- データロードのワーカー数設定可能
- GPU転送の高速化のためのメモリピニング

### エラーハンドリング
- 欠損データファイルの適切な処理
- ClearML接続のフォールバック
- 包括的なログとエラーメッセージ

## 📁 出力ファイル

### 生成されるファイル
- `best_model.pth`: メタデータ付きの訓練済みモデルチェックポイント
- `submission.csv`: Kaggle提出用ファイル
- `detailed_predictions.csv`: 確率付きの詳細予測（オプション）

### 提出フォーマット
```csv
PassengerId,Transported
0001_01,False
0002_01,True
...
```

## 💡 最適化のためのヒント

### ハイパーパラメータ調整
1. 学習率を調整（0.001から開始、0.0001-0.01を試す）
2. ネットワークアーキテクチャを変更（層のサイズと深さ）
3. ドロップアウト率を調整（通常0.1-0.5）
4. 異なるオプティマイザを実験
5. 利用可能メモリに基づいてバッチサイズを調整

### 特徴エンジニアリング
1. コンペティションの洞察に基づいてドメイン固有の特徴を作成
2. 異なるエンコーディング戦略を実験
3. 特徴選択技術を検討
4. 多項式や相互作用特徴を試す

### 訓練戦略
1. より堅牢な評価のためのクロスバリデーションを使用
2. アンサンブル手法を実装
3. 異なる損失関数を試す
4. 学習率スケジューリングを実験

## 🐛 トラブルシューティング

### 一般的な問題

**CUDA/メモリエラー**:
- config.yamlでバッチサイズを削減
- ハードウェア設定でデバイスを'cpu'に設定

**ClearML接続問題**:
- インタラクティブセットアップ支援のために`python setup_clearml.py`を実行
- ClearMLサーバー設定を確認
- 適切なAPI認証情報を確保（上記のClearMLセットアップセクション参照）
- 接続に失敗してもClearMLなしで訓練は継続されます

**データロードエラー**:
- train.csvとtest.csvファイルパスを確認
- ファイル形式と列名を確認
- ファイルが欠損している場合、サンプルデータが生成されます

**ウェブアプリケーション問題**:
- すべての依存関係がインストールされていることを確認: `pip install -r requirements.txt`
- ポート8501が利用可能かどうか確認
- 代替ポートを試す: `streamlit run app.py --server.port 8502`
- 詳細なエラー情報についてアプリケーションログを確認

## 📋 必要要件

- Python 3.7+
- PyTorch 2.0+
- Streamlit 1.28+
- pandas 1.5+
- scikit-learn 1.3+
- ClearML 1.14+（オプション）
- PyYAML 6.0+

## 📄 ライセンス

このソリューションは教育およびコンペティション目的で提供されています。

## 🙏 謝辞

- Kaggle Spaceship Titanic Competition
- PyTorchチーム
- Streamlitコミュニティ
- ClearMLチーム
- オープンソース機械学習コミュニティ

---

**🚀 楽しい予測を！**