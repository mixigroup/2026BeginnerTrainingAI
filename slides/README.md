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

## 📚 参考資料

- [Marp公式ドキュメント](https://marpit.marp.app/)
- [Marp CLI](https://github.com/marp-team/marp-cli)
- [テーマギャラリー](https://github.com/marp-team/marp-core/tree/main/themes)
