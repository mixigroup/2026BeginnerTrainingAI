# Slides Project

Marpを使用してMarkdownからプレゼンテーションスライドを作成する

## 📋 概要

このプロジェクトでは、Markdownファイルから美しいプレゼンテーションスライドを生成できます。HTML、PDF、PowerPoint形式での出力をサポートしています。

## 🚀 セットアップ

### 1. miseのインストール

```bash
curl https://mise.run | sh
```

### 2. プロジェクトのツールをインストール

```bash
cd slides/
mise install  # .mise.tomlに基づいてNode.js 20とMarp CLIをインストール
```

## 🔨 ビルド方法

### mise taskを使用（推奨）

#### HTML出力

```bash
# day1スライドをビルド
mise run build:day1

# day2スライドをビルド
mise run build:day2

# 全スライドをビルド
mise run build:all
```

#### PDF/PowerPoint出力

```bash
# day1スライドをPDFに出力
mise run build:day1:pdf

# day1スライドをPowerPointに出力
mise run build:day1:pptx

# day2スライドをPDFに出力
mise run build:day2:pdf

# day2スライドをPowerPointに出力
mise run build:day2:pptx
```

#### すべてのフォーマットを一括出力

```bash
# day1スライドを全フォーマット（HTML, PDF, PPTX）で出力
mise run export:day1

# day2スライドを全フォーマットで出力
mise run export:day2

# 全スライドを全フォーマットで出力
mise run export:all
```

#### プレビューサーバー

```bash
# プレビューサーバーを起動（localhost:8080）
mise run serve
```

### 直接Marp CLIを使用

```bash
# HTMLファイルとして出力
mise exec -- marp slides/day1-slide/main.md --html --allow-local-files -o public/day1.html

# PDFファイルとして出力
mise exec -- marp slides/day1-slide/main.md --pdf --allow-local-files -o public/day1.pdf

# PowerPoint（PPTX）ファイルとして出力
mise exec -- marp slides/day1-slide/main.md --pptx --allow-local-files -o public/day1.pptx
```

**注意**: `--allow-local-files` フラグはローカル画像やアセットを使用する場合に必要です。

## 📝 スライド作成ガイド

### 基本構文

```markdown
---
marp: true
theme: default
paginate: true
---

# スライドタイトル

スライドの内容

---

## 次のスライド

- 箇条書き項目1
- 箇条書き項目2
```

### スライドの区切り

- `---` でスライドを区切る
- YAML frontmatterでスライド設定を行う

### テーマとスタイリング

```markdown
---
marp: true
theme: default  # default, gaia, uncover
paginate: true  # ページ番号表示
backgroundColor: white
color: black
---
```

## 📁 プロジェクト構造

```
.
├── slides/              # Markdownスライドファイル
│   ├── day1-slide/     # day1スライド
│   │   ├── main.md
│   │   └── imgs/
│   ├── day2-slide/     # day2スライド
│   │   ├── main.md
│   │   └── imgs/
│   └── theme-debug/    # テーマデバッグ用
├── public/             # 生成されたファイル（HTML, PDF等）
├── themes/             # カスタムテーマCSS
├── .mise.toml          # mise設定ファイル
├── README.md           # このファイル
└── CLAUDE.md           # Claude Code用ガイド
```

## 🎨 カスタマイゼーション

### カスタムテーマ

`themes/`ディレクトリにCSSファイルを作成して独自テーマを適用できます：

```markdown
---
marp: true
theme: mixi
---
```

### 画像の使用

```markdown
![画像の説明](slides/day1-slide/imgs/example.png)

<!-- サイズ指定 -->
![w:500](slides/day1-slide/imgs/example.png)
```

## 🎯 Deckを使用したGoogle Slidesへの適用

### セットアップ

#### 1. deckのインストール

```bash
# プロジェクトルートまたはslidesディレクトリで実行
mise install  # Go言語とdeck CLIがインストールされます
```

#### 2. Google OAuth認証の設定（初回のみ）

deckを使用するには、Google Cloud ConsoleでOAuth認証を設定する必要があります：

**手順**:

1. **Google Cloud Consoleでプロジェクトを作成**
   - https://console.cloud.google.com

2. **APIを有効化**
   - [Google Slides API](https://console.cloud.google.com/apis/library/slides.googleapis.com)
   - [Google Drive API](https://console.cloud.google.com/apis/library/drive.googleapis.com)

3. **OAuth 2.0 Client IDを作成**
   - [認証情報ページ](https://console.cloud.google.com/apis/credentials)
   - 「認証情報を作成」→「OAuth クライアント ID」
   - アプリケーションの種類: **デスクトップアプリ**
   - JSONファイルをダウンロード

4. **credentials.jsonを配置**
   ```bash
   mkdir -p ~/.local/share/deck
   cp ~/Downloads/client_secret_xxx.json ~/.local/share/deck/credentials.json
   ```

5. **初回認証**
   ```bash
   mise exec -- deck ls
   # ブラウザが開くのでGoogleアカウントで認証
   ```

詳細は[deck公式ドキュメント](https://github.com/k1LoW/deck#get-and-set-your-oauth-client-credentials)を参照してください。

#### 3. Google Slidesプレゼンテーションの準備

**方法A: 手動でGoogle Slidesを作成（推奨）**

1. Google Slidesで新規プレゼンテーションを作成
2. テーマ/レイアウトを設定（表示 > テーマを編集）
   - レイアウト名を設定（例: `title`, `section`）
3. URLからプレゼンテーションIDをコピー

**方法B: deckコマンドで作成**

```bash
# 新規プレゼンテーション作成
mise exec -- deck new -t "AI研修 Day2"

# 既存のテーマをベースに作成
mise exec -- deck new -t "AI研修 Day2" -b {ベースプレゼンテーションID}
```

#### 4. 環境変数の設定

`.env`ファイルを作成して、Google SlidesのプレゼンテーションIDを設定します：

```bash
cp .env.sample .env
# .envファイルを編集してPRESENTATION_ID_DAY1とPRESENTATION_ID_DAY2を設定
```

Google SlidesのプレゼンテーションIDは、URLから取得できます：
```
https://docs.google.com/presentation/d/{PRESENTATION_ID}/edit
```

#### 5. レイアウトの確認（オプション）

Google Slides側で利用可能なレイアウトを確認：

```bash
cd slides/
source .env
mise exec -- deck ls-layouts -i $PRESENTATION_ID_DAY2
```

### Deckの使用方法

```bash
# day1スライドをGoogle Slidesに適用
mise run deck:day1

# day2スライドをGoogle Slidesに適用
mise run deck:day2

# 全スライドを適用
mise run deck:all

# 特定のページのみを適用（例：3ページ目）
mise run deck:page:day1 page=3
mise run deck:page:day2 page=3
```

### Deck形式について

DeckはMarkdownファイルからGoogle Slidesを直接更新できるツールです。スライドのレイアウトはHTMLコメントで指定します：

```markdown
<!-- {"layout": "title"} -->

# タイトルスライド

---

<!-- {"layout": "section"} -->

## セクション

---

<!-- {"layout": "title-and-body-2col"} -->

# 通常のスライド

コンテンツ
```

詳細は[deck公式リポジトリ](https://github.com/k1LoW/deck)を参照してください。

## 📚 参考資料

- [Marp公式ドキュメント](https://marpit.marp.app/)
- [Marp CLI](https://github.com/marp-team/marp-cli)
- [テーマギャラリー](https://github.com/marp-team/marp-core/tree/main/themes)
- [deck公式リポジトリ](https://github.com/k1LoW/deck)
