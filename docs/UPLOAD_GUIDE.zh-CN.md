> 发布状态（2026-09-20）：三个配套文件已上传 Google Drive，ZIP 匿名下载成功且 SHA-256 与本地发布包一致。以下上传步骤供后续维护使用。

# GitHub 与 Google Drive 上传维护说明

## 内容分工

- GitHub：代码、环境、配置、测试、汇总数据、图表、许可证和复现说明。
- Google Drive：派生特征、逐条预测、划分身份及其他较大的复现文件。
- 原始数据：使用原作者官方链接，不重复上传。
- 手稿全文、作者信息原件、预审材料和临时文件不在本仓库中。

## 发布 Google Drive 数据包

1. 上传交付目录中的 `AI_for_edu_data_v1.0.0.zip`、同名 `.sha256` 文件和 `DATA_MANIFEST.json`。
2. 把文件的常规访问权限设为“知道链接的任何人”，角色为“查看者”。
3. 用未登录账号的浏览器验证链接能够下载文件。
4. 将实际 ZIP 文件共享链接填入 `docs/data_download.json` 的 `url` 字段，将 `status` 改为 `available`。不要修改已核对的 SHA-256。
5. 同步更新 `docs/DATA_ACCESS.md` 和 README 中的下载状态，再提交到 GitHub。

## 固定投稿版本

数据链接验证成功后，为论文最终使用的 Git 提交创建版本标签，例如 `v1.0.0`，并可在 GitHub Releases 发布。论文 Code Availability 引用仓库及该标签/提交；Data Availability 引用原始来源和 Drive 数据包链接。不需要为本项目上传 Zenodo。

后续数据有变化时创建新文件名和新校验值，不要悄悄覆盖论文所引用的数据包。代码更新正常创建新提交；本次不强制推送、不覆盖远端历史。

## 日常更新

```bash
git status
git add README.md docs/ src/ scripts/ tests/ configs/
git diff --cached
git commit -m "Describe the concrete update"
git push origin main
```

提交前检查暂存文件。不要使用整个原研究工作目录替换本公开仓库。`.gitignore` 已排除主要逐条数据、压缩包、环境与临时文件。
