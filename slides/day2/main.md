---
marp: true
theme: mixi
paginate: true
title: 新卒AI研修(２日目)
description: LLM・生成AIに関する基礎とagentについて
---

![mixi-logo w:400px](https://webtan.impress.co.jp/sites/default/files/images/news2022/0914_mixi.png)
<!-- _class: title -->
<!-- _paginate: false -->

# 2026 新卒 AI 研修資料(2日目)

生成AI・LLM・Agentの基礎と実践編

---

# メタ情報

タイトルに「📄」が付いているスライドは、過去の資料を使用していることを示します。

Notion: [LINK](https://www.notion.so/familyalbum/AI-2-TBD)
2025AI研修資料：[LINK](https://docs.google.com/presentation/d/1d-DrS9T4X9nsGQcaTBcT_PVuRRh7QTsojA-fV0N5gJQ/edit)

---
<!-- _class: section -->
<!-- _paginate: false -->

## Introduction

---

# 自己紹介：牧野 舜

**牧野　舜　/　まきの　しゅん**

職種：MLエンジニア

### 経歴

- TBD

---

# ML / DL / 生成AI の関係整理

### 人工知能 (AI)

- 人間のように「考える・判断する」コンピュータの技術全般

### 機械学習 (ML、パターン認識)

- データからパターンを学習して予測する仕組み
- **AIを実現する中核技術**

### 深層学習 (Deep Learning)

- 大量のデータから複雑なパターンを学習できるML手法
- ニューラルネットワークを多層化したもの

### 生成AI (Generative AI)

- テキスト、画像、音声などのコンテンツを生成するAI
- LLM（GPT、Claude）、画像生成（Stable Diffusion、Nano Banana）など

---

# 1日目の復習：教師あり学習

### day1で学んだこと

- **入力 → モデル → 出力** の基本構造
- **分類と回帰**: 離散値か連続値かの違い
- **学習**: データからパターンを学習（loss最小化）
- **推論**: 学習済みモデルで予測
- **評価**: metric（Accuracy、F1、mAP等）で性能測定

### 教師あり学習の特徴

- ラベル付きデータが必要
- 入力 $x$ に対して正解 $y$ を予測する

---

# 2日目のゴール

### 学ぶこと

1. **LLM・生成AIの基礎を理解する**
   - Transformerアーキテクチャ、Attention、LLMの構造

2. **生成モデルを理解する**
   - 拡散モデルの仕組みと画像生成

3. **マルチモーダルを理解する**
   - テキスト・画像・音声を統合する技術

4. **LLMアプリケーションを体験する**
   - RAG、プロンプトエンジニアリング

5. **AI Agentを設計・実装する**
   - Agentの概念、Planning、Tool使用

---

# 目次（今日やること）

1. LLM・生成AIについて
2. LLMの基礎
3. 生成モデルについて
4. マルチモーダルの実現
5. LLMアプリケーション
6. AI Agent
7. Agent ハンズオン
8. クロージング

---
<!-- _class: section -->
<!-- _paginate: false -->

## 1. LLM・生成AIについて

---

# LLM・生成AIについて整理

### LLM（Large Language Model）

- **大規模言語モデル**
- 大量のテキストデータで事前学習された大量のパラメタを持つ言語モデル
  - 参考: GPT-3は1750億パラメータ、GPT-4は推定1兆以上のパラメータ

### マルチモーダルモデル

- 画像や音声もtokenとして扱うことができる(Vision Language Modelなど)
  - tokenについては後述
  - モーダル: テキスト、画像、音声などの扱うデータの形式
- テキストに加えて、画像や音声も入力や出力に使用することができる

### 生成AI（Generative AI）

- **コンテンツを生成するAI**の総称
- テキスト、画像、音声、動画など多様なモダリティに対応
- LLMが中核技術の１つ

---

# 代表的なLLM・生成AIプロダクト（2026年現在）

### LLM（テキスト生成）

- **OpenAI GPT**: GPT-4、GPT-4 Turbo、o1（推論特化）
- **Anthropic Claude**: Claude 3.5/4 (Opus, Sonnet, Haiku)
- **Google Gemini**: Gemini 1.5/2.0 Pro、Ultra
- **Meta Llama**: Llama 3.1/3.2（オープンソース、400B+）
- **DeepSeek**: 中国発の高性能オープンソースLLM

### 画像生成

- **Stable Diffusion 3**（オープンソース）
- **DALL-E 3**（OpenAI）
- **Midjourney v6/v7**
- **Imagen 3**（Google）

### マルチモーダル

- **GPT-4o**: 音声・画像・テキストをネイティブ統合
- **Gemini**: テキスト・画像・音声・動画を統合
- **Claude 3.5 Sonnet**: 画像理解 + テキスト生成

---

# LLM・生成AIの簡単な歴史的発展

### 2017: Transformer登場

- "Attention is All You Need"論文
- TransformerではAttentionを並列で効率的に計算
- 大規模な言語モデルがアーキテクチャとして可能になる

### 2018-2019: BERT、GPT-2

- **BERT**: 双方向エンコーダで文脈理解
- **GPT-2**: 大規模テキスト生成で注目

### 2020: GPT-3

- 1750億パラメータ、Few-shot学習で多様なタスクに対応

### 2022: ChatGPT

- GPT-3.5 + RLHF（人間フィードバック強化学習）
- 対話型インターフェースで爆発的普及

### 2023-2024: GPT-4、Claude 3、Gemini

- マルチモーダル対応、高精度、長いコンテキスト

### 現在 (2025-)

- さらなる精度向上、推論能力の強化
- Agentによる自律的なタスク遂行
- VLA（Vision-Language-Action）モデル：ロボット制御など物理世界への適用

### 20xx

- ???

---
<!-- _class: section -->
<!-- _paginate: false -->

## 2. LLMの基礎

---

# 言語モデルとは

### 定義

- **次の単語を予測する確率モデル**
- 文章の自然さを確率で表現

### 数式表現

$$P(w_t | w_1, w_2, ..., w_{t-1})$$

- $w_t$: 次の単語
- $w_1, ..., w_{t-1}$: これまでの単語列

### 例

「今日の天気は」→ 次に来る確率が高い単語は？

- 「晴れ」: 0.4
- 「曇り」: 0.3
- 「雨」: 0.2
- 「猫」: 0.001

### 分散表現（Word Embedding）

- **単語をベクトル（数値の列）で表現**
- 意味が似た単語は近いベクトルになる

**例**:

- 「猫」: [0.8, 0.1, 0.3, ...]
- 「犬」: [0.7, 0.2, 0.4, ...] ← 猫に近い
- 「車」: [0.1, 0.9, 0.1, ...] ← 猫から遠い

**メリット**:

- 意味の類似性を計算可能
- 機械学習モデルの入力として使用

**代表的な手法**:

- Word2Vec、GloVe → Transformer（現在の主流）

### Decoding戦略

生成時に次の単語をどう選ぶか：

- **Greedy Decoding**: 常に確率最大の単語を選択（決定論的）
- **Beam Search**: 複数候補を保持して探索
- **Top-k Sampling**: 確率上位k個からランダムサンプリング
- **Temperature**: 確率分布の鋭さを調整（高い→多様、低い→保守的）

---

# Transformerの登場

### それまでの問題

- **RNN/LSTM**: 単語の系列を順番に処理
  - 並列化できない
    - 大規模化が難しい
  - 長距離依存が難しい

### Transformerの特徴

1. **Self-Attention + 並列化**:
   - Attentionの概念自体は以前から存在（RNN + Attention）
   - Transformerは**全てをAttentionで処理**し、完全並列化を実現
   - これにより大規模化が可能に
2. **Multi-Head Attention**: 複数の視点で同時に注目
3. **位置エンコーディング**: 単語の順序情報を保持

### 構成要素

- **Encoder**: 入力をベクトル表現に変換（BERT等）
- **Decoder**: ベクトル表現から出力を生成（GPT等）
- **Encoder-Decoder**: 両方を使用（T5等）

---

# Attentionメカニズム

### 仕組み

- **Query（Q）**: 「何を探すか」
- **Key（K）**: 「どこにあるか」
- **Value（V）**: 「実際の値」

### 計算式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 直感的説明

1. QueryとKeyの類似度を計算
2. softmaxで確率分布に変換（重要度）
3. Valueを重み付け合計

### Self-Attention

- 入力系列内の単語同士の関係を学習
- 例：「彼は本を読んだ」→「彼」と「読んだ」の関係を捉える

---

# Multi-Head Attention

### 複数のAttentionを並列実行

- 異なる視点（head）で同時に情報を抽出
- 例：8 heads → 8種類の注目パターンを学習

### メリット

- 多様な関係性を捉えられる
- 性能向上

### 構造

```
Input → [Head1, Head2, ..., Head8] → Concat → Linear → Output
```

---

# TransformerからLLMへ

### GPTの進化

1. **GPT-1** (2018): Transformer Decoder + 事前学習
2. **GPT-2** (2019): スケールアップ（15億パラメータ）
3. **GPT-3** (2020): さらにスケールアップ（1750億パラメータ）
4. **GPT-4** (2023): マルチモーダル、高精度

### スケーリング則（Scaling Laws）

- **モデルサイズ、データ量、計算量**を増やすと性能向上
- ただし、効率とのトレードオフ

### 事前学習とファインチューニング

- **事前学習**: 大量のテキストで汎用的な言語理解を学習
- **ファインチューニング**: タスク特化データで微調整

---

# 代表的なLLMの学習方法

**注意**: 本資料で述べる手法は代表的なものであり、各開発ベンダーは非公開の独自手法や、複数のプロセスを複雑に組み合わせた最適化を行っています。

---

### 1. 事前学習（Pre-training）

- **目的**: 汎用的な言語理解、文脈把握、広範な世界知識の獲得
- **データ**: 数兆トークン規模の公開データ（Web、書籍、論文、ソースコード等）
- **手法**: **次単語予測（Next Token Prediction / Causal Language Modeling）**
- **成果物**: ベースモデル。単語の連続性を確率的に予測する能力は高いが、対話や指示への追従性は限定的。

---

### 2. 事後学習（Post-training）

ベースモデルに対し、特定の用途（対話、推論、指示追従）に適応させるための一連の工程です。

#### 2.1. 指示調整（Supervised Fine-tuning: SFT）

- **目的**: ユーザーの指示（Prompt）に対して適切な形式で回答する「型」を学習させる
- **データ**: 「問い」と「模範解答」のペアからなる高品質なインストラクションデータ
- **手法**: 教師あり学習。特定のドメイン（医療、法務、コード等）への専門特化にも利用される

#### 2.2. 優先度最適化（Preference Optimization）

複数の回答候補から、安全性や有用性の観点で「より好ましいもの」を選択する能力を磨きます。

| 手法 | 概要 | 特徴 |
| :--- | :--- | :--- |
| **RLHF (PPO)** | 人間による評価から報酬モデルを構築し、強化学習を行う。 | 人間の感性に近い調整が可能だが、計算コストが高い。 |
| **DPO** | 報酬モデルを介さず、比較データから直接ポリシーを更新する。 | 学習が安定しており、現在の主流手法の一つ。 |
| **GRPO** | グループ内での相対的なスコアを報酬として利用する。 | DeepSeek-R1で採用。計算リソース（VRAM）を劇的に節約可能。 |

---

### 3. 推論能力の強化（Reasoning RL）

数学、論理パズル、プログラミングなど、客観的な正解（Ground Truth）が定義可能なタスクに対するアプローチです。

- **手法**: ルールベース報酬を用いた強化学習
- **内容**: モデルに「思考の連鎖（Chain-of-Thought）」を生成させ、最終的な正答率や論理的整合性に対して報酬を与える
- **効果**: モデルが自己修正（Self-correction）を行いながら複雑な問題を解く能力を獲得する

---

### 4. 自己改善と合成データ（Self-Evolution）

学習データの限界を突破し、モデルの性能をさらに引き上げるための手法です。

- **棄却サンプリング（Rejection Sampling）**: モデルが生成した大量の回答から、報酬モデルやテストコード等で検証済みの高品質なものだけを抽出し、再度SFTを行う
- **RLAIF (AI Feedback)**: 人間の代わりに、より高度なモデル（または特定のルールセット）がフィードバックを行い、学習をスケーリングさせる

---

# LLMの評価指標

### 言語モデル評価

- **Perplexity（PPL）**: 予測の自信度（低いほど良い）

### タスク評価

- **BLEU**: 機械翻訳の精度
- **ROUGE**: 要約の精度
- **Exact Match / F1**: 質問応答の精度

### ベンチマーク

- **GLUE / SuperGLUE**: 自然言語理解タスク
- **MMLU**: 多分野知識理解
- **HumanEval**: コード生成精度

### 人間評価

- **Helpfulness**: 有用性
- **Harmlessness**: 無害性
- **Honesty**: 正直さ

---
<!-- _class: section -->
<!-- _paginate: false -->

## 3. 生成モデルについて

---

# 生成モデルとは

### 教師あり学習との違い

| | 教師あり学習 | 生成モデル |
|---|---|---|
| **目的** | 入力 → 正解ラベルを予測 | データの分布を学習 → 新規データ生成 |
| **出力** | 分類ラベル、数値 | 画像、テキスト、音声 |
| **例** | 犬/猫分類、売上予測 | 画像生成、文章生成 |

### 生成モデルの種類

1. **GAN（Generative Adversarial Network）**
2. **VAE（Variational Autoencoder）**
3. **拡散モデル（Diffusion Models）** ← 今回の焦点

---

# 拡散モデルの概要

### 基本アイデア

1. **Forward Process（拡散過程）**: 画像にノイズを徐々に加えて破壊
2. **Reverse Process（逆拡散過程）**: ノイズから元の画像を復元

### 学習

- **ノイズ除去を学習**する
- 各ステップでどのノイズを除去すれば良いかを予測

### 生成

- ランダムノイズから開始
- 学習したノイズ除去を繰り返し適用
- 最終的に自然な画像を生成

---

# 拡散モデルの仕組み

### Forward Process

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

- $x_0$: 元画像
- $x_t$: ステップ $t$ でのノイズ付き画像
- $\beta_t$: ノイズスケジュール

### Reverse Process

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

- モデル $\theta$ がノイズを予測して除去

---

# Stable Diffusion

### 特徴

- **Latent Diffusion Model**: 画素空間ではなく潜在空間で拡散
- **高速・高品質**: 計算効率が良い
- **オープンソース**: 商用利用可能

### 構成要素

1. **VAE Encoder**: 画像 → 潜在表現
2. **U-Net**: ノイズ除去（Attention付き）
3. **VAE Decoder**: 潜在表現 → 画像
4. **Text Encoder（CLIP）**: プロンプト → 条件付け

### プロンプト

- テキストで生成内容を制御
- 例：「a beautiful sunset over mountains, photorealistic」

---

# ハンズオン：Stable Diffusionで画像生成

### やること

```bash
cd day2/03-diffusion-model
uv run python src/stable_diffusion_demo.py
```

- プロンプトから画像生成
- Negative promptで不要な要素を除外
- Seed値で再現性確保

### 観察ポイント

- プロンプトの変化で出力がどう変わるか
- ステップ数と品質のトレードオフ
- Guidance scaleの影響

---
<!-- _class: section -->
<!-- _paginate: false -->

## 4. マルチモーダルの実現

---

# マルチモーダルとは

### 定義

- **複数のモダリティ（テキスト、画像、音声など）を統合**して処理

### なぜ必要か

- 人間は複数の感覚で世界を理解している
- テキストだけでは表現できない情報がある（画像の内容、音声のニュアンス等）

### 応用例

- **画像キャプション生成**: 画像 → テキスト説明
- **Visual Question Answering**: 画像 + 質問 → 回答
- **Text-to-Image**: テキスト → 画像生成

---

# CLIP（Contrastive Language-Image Pre-training）

### 概要

- **画像とテキストを同じ空間に埋め込む**
- OpenAIが2021年に発表

### 学習方法

1. 大量の（画像、テキスト）ペアを収集
2. **Contrastive Learning**: 対応ペアは近く、非対応は遠く

### 構造

- **Image Encoder**: Vision Transformer（ViT）
- **Text Encoder**: Transformer
- **共通埋め込み空間**: 両方を同じ次元に射影

### 応用

- Zero-shot画像分類
- Stable Diffusionのテキスト条件付け

---

# Vision Transformer（ViT）

### CNNからTransformerへ

- **従来**: CNN（畳み込み）で画像処理
- **ViT**: 画像をパッチ分割 → Transformerで処理

### 仕組み

1. 画像を16×16パッチに分割
2. 各パッチを線形変換してembedding
3. 位置エンコーディング追加
4. Transformer Encoderで処理

### メリット

- **長距離依存**を捉えやすい
- **スケーラビリティ**が高い

---

# マルチモーダルLLM

### 代表例

- **GPT-4V**: 画像理解 + テキスト生成
- **Gemini**: テキスト・画像・音声・動画を統合
- **LLaVA**: オープンソースのVision-LLM

### 仕組み

1. **Vision Encoder**: 画像をembeddingに変換（CLIP、ViT）
2. **Projection Layer**: Vision embeddingをLLMの入力形式に変換
3. **LLM**: テキストとVision embeddingを統合して処理

### 応用

- 画像の詳細説明
- 図表の解釈
- 視覚的推論

---

# Optional: Speech Encoder + LLM

### 音声とLLMの統合

- **Whisper**: OpenAIの音声認識モデル
- **音声 → テキスト → LLM**: 従来の方法
- **音声 → Embedding → LLM**: 直接統合（研究中）

### メリット

- 音声のニュアンス（感情、イントネーション）を保持
- 多言語対応

---
<!-- _class: section -->
<!-- _paginate: false -->

## お昼休憩

---
<!-- _class: section -->
<!-- _paginate: false -->

## 5. LLMアプリケーション

---

# LLM APIの使い方

### 代表的なAPI

- **OpenAI API**: GPT-4、GPT-3.5
- **Anthropic API**: Claude 3
- **Google Gemini API**
- **Azure OpenAI Service**

### 基本的な使い方

```python
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

---

# プロンプトエンジニアリング

### プロンプトとは

- LLMへの指示文
- 出力を制御する重要な要素

### 基本テクニック

1. **明確な指示**: 「〜してください」と具体的に
2. **Few-shot**: 例を示す
3. **Chain of Thought**: 「ステップバイステップで考えて」
4. **ロール設定**: 「あなたは〜の専門家です」

### 例

```
あなたはPythonの専門家です。
以下のコードをレビューして、改善点を3つ挙げてください。

[コード]
```

---

# RAG（Retrieval-Augmented Generation）

### 背景

- LLMは**学習時点の知識のみ**を持つ
- 最新情報や社内情報には対応できない

### RAGの仕組み

1. **Retrieval（検索）**: 質問に関連する文書を検索
2. **Augmented（拡張）**: 検索結果をプロンプトに追加
3. **Generation（生成）**: LLMが回答生成

### メリット

- 最新情報に対応
- 社内文書など特定ドメインに対応
- ハルシネーション（幻覚）を減らす

---

# RAGのアーキテクチャ

### 構成要素

1. **Document Store**: ベクトルDB（Pinecone、Weaviate、Chroma等）
2. **Embedding Model**: テキスト → ベクトル変換
3. **Retriever**: 類似文書検索
4. **LLM**: 回答生成

### フロー

```
質問 → Embedding → Vector検索 → 関連文書取得 →
プロンプト構築（質問 + 文書） → LLM → 回答
```

---

# Vector Database

### なぜ必要か

- **類似度検索**を高速に行うため
- 従来のRDBでは大規模ベクトル検索が遅い

### 代表的なVector DB

- **Pinecone**: マネージドサービス
- **Weaviate**: オープンソース
- **Chroma**: 軽量・ローカル開発向け
- **Qdrant**: Rust製、高速

### 検索アルゴリズム

- **HNSW**: 近似最近傍探索
- **IVF**: クラスタリングベース

---

# LangChain入門

### LangChainとは

- LLMアプリケーション開発のフレームワーク
- プロンプト管理、Chain構築、Agent実装を簡素化

### 主要コンポーネント

1. **Prompts**: プロンプトテンプレート
2. **Chains**: 複数処理の連鎖
3. **Agents**: 動的にToolを使用
4. **Memory**: 会話履歴管理
5. **Retrieval**: RAG実装

---

# ハンズオン：RAGシステム構築

### やること

```bash
cd day2/05-llm-application
uv run python src/rag_demo.py
```

- ドキュメントをEmbedding化してVector DBに格納
- 質問に対して関連文書を検索
- LLMで回答生成

### 観察ポイント

- Embedding modelの選択（OpenAI、Sentence-BERT等）
- Retrieval件数と精度のトレードオフ
- プロンプト設計の影響

---
<!-- _class: section -->
<!-- _paginate: false -->

## 6. AI Agent

---

# AI Agentとは

### 定義

- **自律的に目標達成のために行動するシステム**
- LLMを「頭脳」として、Tools（外部ツール）を使って複雑なタスクを実行

### 従来のLLMアプリとの違い

| | 従来のLLMアプリ | AI Agent |
|---|---|---|
| **処理** | 1回のプロンプト → 1回の応答 | 複数ステップを自律実行 |
| **外部ツール** | なし or 固定 | 動的にツールを選択・使用 |
| **適応性** | 低い | 高い（状況に応じて計画変更） |

---

# AI Agentの構成要素

### 1. Planning（計画）

- タスクを分解して実行計画を立てる
- 例：「データ分析レポート作成」 → [データ取得、分析、グラフ作成、レポート執筆]

### 2. Tools（ツール）

- 外部API、DB、検索エンジン、コード実行環境など
- 例：Google検索、Python実行、SQL Query

### 3. Memory（記憶）

- 過去の対話・実行結果を保持
- Short-term memory（会話履歴）、Long-term memory（知識ベース）

### 4. Execution（実行）

- 計画に基づいてToolを呼び出し
- 結果を評価して次のアクションを決定

---

# Function Calling

### 概要

- LLMが**構造化された関数呼び出し**を返す機能
- OpenAI、Anthropic、Gemini等が対応

### 仕組み

1. 関数定義（名前、引数、説明）をLLMに渡す
2. LLMが適切な関数とパラメータを選択
3. アプリ側で関数を実行
4. 結果をLLMに返して続きを生成

### 例

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {"location": "string"}
    }
]
# LLM → {"name": "get_weather", "arguments": {"location": "Tokyo"}}
```

---

# ReAct（Reasoning + Acting）

### コンセプト

- **推論（Reasoning）と行動（Acting）を交互に実行**
- Thought → Action → Observation のループ

### フロー

1. **Thought**: 「次に何をすべきか考える」
2. **Action**: ツールを実行
3. **Observation**: 結果を観察
4. 1に戻る（ゴール達成まで繰り返し）

### メリット

- 透明性が高い（思考過程が見える）
- 動的にツールを選択できる

---

# AI Agentの実装フレームワーク

### LangGraph

- LangChainのAgent実装に特化
- **グラフ構造**でワークフローを定義
- State管理、条件分岐、ループをサポート

### その他のフレームワーク

- **AutoGPT**: 自律的タスク実行
- **BabyAGI**: タスク管理Agent
- **CrewAI**: マルチAgent協調

---

# AI Agentの開発フロー

### 1. 要件定義

- 何を達成したいか明確化
- 必要なToolsをリストアップ

### 2. 設計

- Agentのワークフロー設計（Planning → Execution）
- Toolsの定義（入出力スキーマ）

### 3. 実装

- LangGraph等でAgent実装
- Toolsの統合

### 4. 評価

- タスク成功率、実行時間、コスト
- エラーハンドリング

### 5. 運用

- ログ・モニタリング
- 継続的改善

---

# AI Agentの評価

### 成功率

- タスク完了率
- 正しい結果を返した割合

### 効率性

- 実行時間
- API呼び出し回数（コスト）

### 安定性

- エラー率
- 予期しない動作の頻度

### 人間評価

- 出力の有用性
- エラー時のハンドリング

---

# AI Agentの課題

### 1. コスト

- LLM APIコールが多い
- 長いcontext → 高コスト

### 2. レイテンシ

- 複数ステップ → 遅い

### 3. 信頼性

- Tool実行の失敗
- LLMの誤判断

### 4. セキュリティ

- Tool実行の権限管理
- Prompt Injection攻撃

---
<!-- _class: section -->
<!-- _paginate: false -->

## 7. Agent ハンズオン

---

# ハンズオン：LangGraphでCoding Agent作成

### タスク

- ユーザーの要求からPythonコードを生成・実行するAgent

### 構成

1. **Planning Node**: タスク理解・計画立案
2. **Code Generation Node**: コード生成
3. **Code Execution Node**: コード実行
4. **Evaluation Node**: 結果評価・修正判断

### Tools

- Python実行環境（サンドボックス）
- ファイル読み書き
- Web検索（必要に応じて）

---

# ハンズオン手順

### 1. セットアップ

```bash
cd day2/07-agent-hands-on
uv sync
```

### 2. Agent実装

- `src/coding_agent.py` を実装
- LangGraphでワークフロー定義

### 3. 実行

```bash
uv run python src/coding_agent.py
```

### 4. 観察・改善

- 成功率を測定
- エラーハンドリング追加
- プロンプト改善

---

# 拡張課題

### 1. エラーリトライ機能

- コード実行失敗時に自動修正

### 2. テスト生成

- 生成コードのテストケースを自動生成

### 3. マルチステップタスク

- 複数ファイルにまたがる実装

### 4. 評価ダッシュボード

- 実行結果を可視化

---
<!-- _class: section -->
<!-- _paginate: false -->

## 8. クロージング

---

# 今日のまとめ

### 学んだこと

1. **LLM・生成AIの基礎**: Transformer、Attention、事前学習
2. **生成モデル**: 拡散モデルで画像生成
3. **マルチモーダル**: CLIP、ViTで画像とテキストを統合
4. **LLMアプリケーション**: RAG、プロンプトエンジニアリング
5. **AI Agent**: Planning、Tools、Function Calling

### 体験したこと

- Stable Diffusionで画像生成
- RAGシステム構築
- LangGraphでCoding Agent作成

---

# day1とday2のつながり

### day1: ML基礎

- 教師あり学習、推論、学習、評価、運用

### day2: LLM・生成AI

- 生成モデル、自己教師あり学習、事前学習、RLHF

### 共通点

- **データ → モデル → 評価 → 運用**のサイクル
- **KPI・metric・loss**の設計
- **監視・更新プロセス**の重要性

### 違い

- day1: タスク固有のモデル学習
- day2: 汎用モデル + プロンプト・RAGで適応

---

# これからの学び

### 実務での活用

- LLMをプロダクトにどう組み込むか
- コスト・レイテンシ・精度のバランス
- 評価・モニタリング設計

### 継続的学習

- 最新論文・技術のキャッチアップ
- ハンズオンで実装力向上
- コミュニティ参加（勉強会、OSS貢献）

---

# おつかれさまでした

### 質問・フィードバック

- 疑問点や気になったことがあれば遠慮なく質問してください
- 今日の内容で特に役立ったこと、難しかったことを共有してください

### 次のステップ

- ハンズオンの復習・拡張課題に取り組む
- 自分のプロダクト・業務で応用できそうなポイントを考える
- AI技術を使ってプロダクトを改善していきましょう！
