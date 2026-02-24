# Slides Project

Marpを使用してMarkdownからプレゼンテーションスライドを作成する

## 📋 概要

このプロジェクトでは、Markdownファイルから美しいプレゼンテーションスライドを生成できます。HTML、PDF、PowerPoint形式での出力をサポートしています。

### Marp CLIの基本的な使用方法

```bash
# サーバを立てて、スライドを確認
marp -s ./slides --allow-local-files
```

### エクスポート方法

#### PDFファイルとして出力

```bash
# 特定のディレクトリをPDFとして出力
marp --input-dir ./slides/sample/ --pdf --allow-local-files

# 単一ファイルをPDFとして出力
marp ./slides/sample.md --pdf --allow-local-files
```

#### PowerPoint（PPTX）ファイルとして出力

```bash
# 特定のディレクトリをPowerPointとして出力
marp --input-dir ./slides/sample/ --pptx --allow-local-files

# 単一ファイルをPowerPointとして出力
marp ./slides/sample.md --pptx --allow-local-files
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
│   ├── sample.md       # サンプルスライド
│   └── ...             # その他のスライド
├── dist/               # 生成されたファイル（HTML, PDF等）
├── assets/             # 画像やカスタムCSSファイル
├── package.json        # プロジェクト設定
├── README.md          # このファイル
└── CLAUDE.md          # Claude Code用ガイド
```

## 🎨 カスタマイゼーション

### カスタムテーマ

`assets/`ディレクトリにCSSファイルを作成して独自テーマを適用できます：

```markdown
---
marp: true
theme: assets/custom-theme.css
---
```

### 画像の使用

```markdown
![画像の説明](assets/images/my-image.png)

<!-- サイズ指定 -->
![w:500](assets/images/my-image.png)
```

## 📚 参考資料

- [Marp公式ドキュメント](https://marpit.marp.app/)
- [Marp CLI](https://github.com/marp-team/marp-cli)
- [テーマギャラリー](https://github.com/marp-team/marp-core/tree/main/themes)
