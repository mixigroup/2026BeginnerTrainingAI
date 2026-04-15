# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このプロジェクトはMarp（Markdown Presentation Ecosystem）とdeck（Google Slides連携ツール）を使用してMarkdownからスライドを生成するプロジェクトです。

- **Marp**: HTML/PDF/PowerPoint形式でスライドを出力
- **deck**: Google Slidesに直接Markdownを適用

## 環境について

このプロジェクトではmiseを使用してMarp CLIとdeckをローカルに管理しています。
`.mise.toml`でNode.js 20、Marp CLI、deckのバージョンを固定しています。

## セットアップ

### 1. miseのインストール

```bash
curl https://mise.run | sh
```

### 2. プロジェクトのツールをインストール

```bash
cd slides/
mise install  # .mise.tomlに基づいてNode.js 20、Marp CLI、deckをインストール
```

### 3. deck用のOAuth認証設定（Google Slidesを使用する場合、初回のみ）

```bash
# 1. Google Cloud ConsoleでOAuth 2.0 Client IDを作成
# 2. credentials.jsonをダウンロード
mkdir -p ~/.local/share/deck
cp ~/Downloads/client_secret_xxx.json ~/.local/share/deck/credentials.json

# 3. 初回認証
mise exec -- deck ls  # ブラウザが開いてGoogle認証
```

詳細: https://github.com/k1LoW/deck#get-and-set-your-oauth-client-credentials

### 4. Google Slidesプレゼンテーションの準備

**方法A: 手動作成**
- Google Slidesで新規作成
- レイアウトを設定（表示 > テーマを編集）
- URLからプレゼンテーションIDを取得

**方法B: deckコマンドで作成**
```bash
mise exec -- deck new -t "タイトル" -b {ベースプレゼンテーションID}
```

### 5. 環境変数設定

```bash
cp .env.sample .env
# .envファイルを編集してPRESENTATION_ID_DAY1とPRESENTATION_ID_DAY2を設定
```

### 4. ディレクトリ構造

```bash
# 出力ディレクトリの作成
mkdir -p public
```

## よく使用するコマンド

### Marp（HTML/PDF/PowerPoint出力）

```bash
mise run build:day1     # day1スライドをビルド
mise run build:day2     # day2スライドをビルド
mise run build:all      # 全スライドをビルド
mise run serve          # プレビューサーバーを起動（localhost:8080）
```

### deck（Google Slides連携）

```bash
mise run deck:day1           # day1スライドをGoogle Slidesに適用
mise run deck:day2           # day2スライドをGoogle Slidesに適用
mise run deck:all            # 全スライドを適用
mise run deck:page:day1 page=3  # day1の特定ページのみ適用
mise run deck:page:day2 page=5  # day2の特定ページのみ適用
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
├── day1/                # day1スライド
│   ├── main.md
│   └── imgs/
├── day2/                # day2スライド
│   ├── main.md
│   └── imgs/
├── themes/              # カスタムテーマCSS
├── public/              # 生成されたファイル（HTML, PDF等）
├── mise.toml            # miseタスク定義とツール管理
├── .env.sample          # 環境変数サンプル
├── .env                 # 環境変数（PRESENTATION_ID等）
├── README.md
└── CLAUDE.md
```

## スライド作成のガイドライン

### 基本的なMarkdown構文
- `---` でスライドを区切る
- `# タイトル` でスライドタイトル

### Marp形式（HTML/PDF/PowerPoint出力用）

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

### deck形式（Google Slides連携用）

deck形式ではYAML frontmatterを使用せず、HTMLコメントでレイアウトを指定します：

```markdown
<!-- {"layout": "title"} -->

# タイトルスライド

## サブタイトル

---

<!-- {"layout": "section"} -->

## セクション

---

# 通常のスライド

コンテンツ
```

**利用可能なレイアウト**:
- `title`: タイトルスライド
- `section`: セクション区切り
- `title-and-body-2col`: タイトルと2カラムのボディ
- レイアウト指定なし: デフォルトレイアウト

詳細は[deck公式リポジトリ](https://github.com/k1LoW/deck)を参照してください。

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

- Markdownファイルは各day配下（`day1/`, `day2/`）に配置
- 生成されたファイルは`public/`ディレクトリに出力
- テーマやカスタムCSSが必要な場合は`themes/`ディレクトリを使用
- miseでMarp CLIとdeckが管理されているため、グローバルで`marp`と`deck`コマンドが利用可能
- Playwright MCPを使った開発では、サーバーをバックグラウンドで起動しておくと便利
- deck形式では、Marpの`<!-- _class: xxx -->`ではなく、`<!-- {"layout": "xxx"} -->`形式を使用する
- Google Slidesに適用する際は、事前に`.env`ファイルでPRESENTATION_IDを設定する必要がある