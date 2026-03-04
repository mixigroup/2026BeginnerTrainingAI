# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このプロジェクトはMarp（Markdown Presentation Ecosystem）を使用してMarkdownからスライドを生成するプロジェクトです。

## 環境について

このプロジェクトではmiseを使用してMarp CLIをローカルに管理しています。
`.mise.toml`でNode.js 20とMarp CLIのバージョンを固定しています。

## セットアップ

### 1. miseのインストール

```bash
curl https://mise.run | sh
```

### 2. プロジェクトのツールをインストール

```bash
cd slides/
mise install  # .mise.tomlに基づいてNode.js 20とMarp CLIをインストール
```

### 3. ディレクトリ構造

```bash
# 出力ディレクトリの作成
mkdir -p public
```

## よく使用するコマンド

### mise task（推奨）

```bash
mise run build:day1     # day1スライドをビルド
mise run build:day2     # day2スライドをビルド
mise run build:all      # 全スライドをビルド
mise run serve          # プレビューサーバーを起動（localhost:8080）
```

### 直接的なMarp CLIコマンド

```bash
# miseでインストールされたコマンドを使用
mise exec -- marp slides/day1-slide/main.md --html --allow-local-files -o public/day1.html
mise exec -- marp slides/day1-slide/main.md --pdf --allow-local-files -o public/day1.pdf
mise exec -- marp slides/day1-slide/main.md --pptx --allow-local-files -o public/day1.pptx
```

## プロジェクト構造

```
.
├── slides/           # Markdownスライドファイル
├── dist/            # 生成されたファイル（HTML, PDF等）
├── assets/          # 画像やCSSファイル
├── package.json
└── CLAUDE.md
```

## スライド作成のガイドライン

### 基本的なMarkdown構文
- `---` でスライドを区切る
- `# タイトル` でスライドタイトル
- YAML frontmatterでテーマやスタイルを設定

### 典型的なスライドファイルの例
```markdown
---
marp: true
theme: default
paginate: true
---

# タイトルスライド

---

## 内容スライド

- 項目1
- 項目2
```

## Playwright MCPを使った開発ワークフロー

このプロジェクトではPlaywright MCPを使ってブラウザでスライドをプレビューしながら開発することが推奨されます。

### 基本的なワークフロー

1. **Marpサーバーの起動**
   ```bash
   marp -s slides/ -p 8080
   ```
   または
   ```bash
   npm run serve
   ```

2. **Playwright MCPでブラウザを開く**
   - `browser_navigate` で `http://localhost:8080` にアクセス
   - `browser_snapshot` でページの内容を確認
   - `browser_take_screenshot` でスクリーンショットを取得

3. **開発サイクル**
   - Markdownファイルを編集
   - Marpが自動的に変更を検知して再ビルド
   - ブラウザで `browser_navigate_back` や再読み込みで最新の状態を確認
   - `browser_snapshot` で最新のスライドの状態を確認

### 実用的な使用例

```bash
# 1. サーバー起動（バックグラウンドで）
marp -s slides/ -p 8080 &

# 2. Playwright MCPでブラウザを開いてプレビュー
# Claude Codeで以下のツールを使用：
# - browser_navigate: http://localhost:8080
# - browser_snapshot: スライドの内容を確認
# - browser_click: スライドのナビゲーション
# - browser_press_key: 矢印キーでスライド移動

# 3. スライドを編集しながらリアルタイムで確認
```

### デバッグのポイント

- **レイアウト確認**: `browser_snapshot`でスライドの構造を確認
- **視覚的確認**: `browser_take_screenshot`でスクリーンショットを保存
- **ナビゲーション**: `browser_press_key`で矢印キーを使ってスライド間を移動
- **リロード**: ファイル変更後は自動リロードされるが、手動で確認したい場合は`browser_navigate`で再アクセス

### よくある確認項目

- スライドのフォントサイズや配置
- 画像の表示確認
- コードブロックのシンタックスハイライト
- ページネーションの表示
- テーマの適用状況

## 開発時の注意事項

- Markdownファイルは`slides/`ディレクトリに配置
- 生成されたファイルは`dist/`ディレクトリに出力
- テーマやカスタムCSSが必要な場合は`assets/`ディレクトリを使用
- miseでMarp CLIが管理されているため、グローバルで`marp`コマンドが利用可能
- Playwright MCPを使った開発では、サーバーをバックグラウンドで起動しておくと便利