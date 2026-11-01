# 四脚ロボットの指示歩行と外乱・不整地への挑戦

Physical AI講座2026応用編 最終課題 / OmniCampus ID: 86574

Genesisの公式Go2歩行例をもとに、PPOで速度変更・停止・左右旋回を学習し、左右の外力と滑らかな不整地への追加学習を試したコードです。実演データ・実機・画像入力は使用していません。

## 実行するNotebook

1. [`01_baseline.ipynb`](notebooks/01_baseline.ipynb)：ColabでGPUを選び、依存導入→公式ソース取得・ハッシュ検証→前進モデルの学習・評価・録画。Driveを使う場合は自分のアカウントでマウントします。
2. [`02_commands.ipynb`](notebooks/02_commands.ipynb)：冒頭の`BASELINE_CHECKPOINT`へ1で作成した`checkpoint.pt`の絶対パスを記入。速度・停止から旋回へ段階的に学習し、30条件で評価します。checkpointと隣接する`config.json`が必要です。
3. [`03_extensions.ipynb`](notebooks/03_extensions.ipynb)：同じランタイムで`CHECKPOINT`を2の結果へ変更し、外乱または不整地の節を実行。既存結果を上書きしない新しい保存先を作ります。**全セル一括実行せず、必要な実験を選んでください。** 不整地再挑戦は初回不整地の準備セルに依存します。

Notebookは出力・実行回数を除いた配布用です。実験に使ったソースを収録し、個別workerと学習・評価の動作はColab T4で確認しました。公開用に整理したNotebook全体の新規ランタイムからの再実行は未検証です。GPU割当・所要時間・同じ成功数は保証しません。数十分以上のGPU処理とDriveへの書き込みが発生します。中断する場合は実行中の子プロセスも確認してください。

## 構成

- `src/command_*`, `commands.py`：指示の生成・段階学習・評価
- `src/push_*`, `pushes.py`：外力・回復判定・追加学習
- `src/terrain_*`, `prepare_terrain_retry.py`：地形・高さ報酬・再挑戦
- `src/calibrate_*`, `probe_push_env.py`, `probe_terrain_policy.py`：物理と環境の検査
- `src/audit_*`, `record_*`, `finish_*`, `export_learning_curves.py`：CSV監査・録画・曲線・保存
- `*-config.json`, `terrain_retry_spec.json`：指示分布・外力・学習上限・評価条件

45次元の状態から12関節の目標角度差を出し、PD制御します。制御50 Hz、行動の倍率0.25 rad、傾き10度で終了。報酬は速度追従、旋回追従、上下動、地表からの胴体高さ、行動変化、基準関節角との差で構成します。追加の姿勢報酬は不整地の再挑戦の一候補だけに使用します。詳細はworkerと設定に記録されます。

## 実験結果

- 指示歩行：前進モデル0/30 → 指示学習28/30。
- 外乱：追加学習前後60/60。強い外乱の回復時間中央値0.86→0.72秒、最大横ずれ中央値0.128→0.201 m。全体的な改善とは主張しません。
- 不整地初回：弱・中・強で旧20/20・20/20・12/20、追加学習後20/20・20/20・4/20。強で悪化しました。
- 再挑戦：低学習率候補は調整用17/18・平地25/30、姿勢報酬を加えた候補は18/18・26/30。平地27/30の採用条件を満たさず両方不採用。
- 維持した指示モデルの別の未知地形評価：20/20・20/20・8/20。平地29/30。再学習改善の結果ではありません。

地形は高さ±1・2・4 cmの滑らかな凹凸のみ。階段・岩・実機は未検証。外乱と不整地は指示モデルから別々に追加学習したもので、全能力を統合したモデルではありません。各runに設定・依存版・乱数・CSV・checkpoint・ログを保存します。重みや実行ログはこの公開リポジトリに同梱していません。

## 依存と出典

Colab Linux / Python 3.10–3.13 / NVIDIA GPU。実験はT4・Python 3.13で実施。
Genesis 1.4.5、rsl-rl-lib 5.5.2、PyTorch 2.11.0+cu128。Notebookで独立venvへ導入し、推移依存は`pip freeze`で保存します。完全なlockではありません。

公式ソースは [Genesis locomotion](https://github.com/Genesis-Embodied-AI/Genesis/tree/1ac23c39b32e567cc76e20a467fee99aaad257b7/examples/locomotion) の固定commitからNotebook実行時に取得し、`src/upstream.lock.json`のSHA-256で検証します。GenesisはApache-2.0、rsl_rlはBSD-3-Clauseです。ロボット資産は各配布元の条件も参照してください。講義資料は含みません。
