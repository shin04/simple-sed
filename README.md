# About

卒業研究の下流タスク用のリポジトリ。
オーディオデータ、またはHuBERT特徴量を利用した学習が可能。

# Setup

1. 環境変数の設定

    `.env`ファイルに以下の変数を設定

    - `WORKDIR` : 作業ディレクトリ
    - docker関連（dockerを利用しない場合は不要）
        - `IMAGE_NAME` : docker imageの名前
        - `CONTAINER_NAME` : docker containerの名前
        - `VERSION` : docker imageのバージョン
        - `MOUNT_PATH` : コンテナ内の作業ディレクトリ
    - データ関連
        - `AUDIO_PATH` : オーディオファイルの保存先
        - `META_PATH` : オーディオファイルのメタデータの保存先
        - `FEAT_PATH` : HUBERT特徴量の保存先
    - `MODEL_PATH` : 学習したモデルの保存先
    - `RESULT_PATH` : 生成物の保存先

2. `init.sh`の実行

    ```bash
    $ chmod 700 init.sh
    $ ./init.sh
    ```

    シンボリックリンクの生成、mlflowの設定、venv環境の生成を行います。

3. docker imageのbuild（dockerを利用しない場合は不要）

    必要なdocker imageをbuild

    ```bash
    $ make build # for downstream task
    $ make build-finetune # for finetuning task
    $ make build-cuml # for analyze
    ```

4. docker containerの起動（dockerを利用しない場合は不要）

    ```bash
    $ make run # for downstream task
    $ make run-finetune # for finetuning task
    $ make feature_visualize # for analyze
    $ make pretrained_feature_vis # for analyze
    ```

# Examples

## Training

## Inference

## Visualize
