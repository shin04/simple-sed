# About

卒業研究の下流タスク用のリポジトリ。
オーディオデータ、またはHuBERT特徴量を利用した学習が可能。

## ディレクトリ構成

```
simple-sed
├─ config # 各パラメータの設定ファイル
│   ├─ baseline.yaml
│   ├─ default.yaml
│   ├─ finetuning.yaml
│   └─ hubert.yaml
├─ dataset # データセット（init.shで生成）
├─ log # ログファイル
├─ meta # メタファイル（init.shで生成）
├─ models # モデルの保存先（init.shで生成）
├─ results # 生成物の保存先（init.shで生成）
├─ src # ソースディレクトリ
├─ .gitignore
├─ Dockerfile
├─ Dockerfile.cuml
├─ Dockerfile.finetune
├─ init.sh
├─ Makefile
├─ README.md
├─ requirements.txt
├─ requirements.finetune.txt
└─ .venv # init.shで生成
```

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

5. setting config file

    `config`ファイル内のyamlファイルに各パラメータを設定

# Examples

## Training

**そのまま**
```bash
$ source .venv/bin/activate
$ cd src
$ python run.py
```

**dockerを利用**
```bash
$ make train
```

## Inference

```bash
$ source .venv/bin/activate
$ cd src
$ python test_process.py $(model type) $(path/to/model) $(path/to/hubert/feature) -a -g
```
