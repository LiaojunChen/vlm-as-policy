# VLM as Policy v72 · RoboTwin results

网站：[VLM as Policy](https://liaojunchen.github.io/vlm-as-policy/) · 仓库：[LiaojunChen/vlm-as-policy](https://github.com/LiaojunChen/vlm-as-policy)

VLM as Policy 的 v72 源码、真实评测结果与交互式证据展示。

评测源码位于 [`harness/`](harness/)，同步范围、外部依赖与运行说明见 [HARNESS.md](HARNESS.md)，逐文件校验清单见 [harness-source-manifest.json](harness-source-manifest.json)。

主批次：`repeat500_official_sampling_maxsteps30_20260922`，50 个任务 × 10 个不同接受 seed，233/500 成功，成功率 **46.6%**，0 基础设施错误。2026-09-23 03:48:09 UTC 完成。
模型为 `phyRSI/ZDTaichu5.0-9B`；`demo_clean`、30 决策预算、1200 秒墙钟预算。按 RoboTwin 官方方式从 seed 100000 顺序扫描，专家验证通过后取每任务前 10 个 seed（官方常规为每任务 100 次）。与官网 C2R 不同评测口径，不构成同榜排名。历史开发批次 36/50（72%）单独展示，不能作为同 seed 配对比较。

## 网站功能

- 50×10 可点击结果矩阵；50 个任务筛选、排序、逐轮入口。
- 首页对比本次 46.6% 与用户提供的 GPT-6 zero-shot 53.2%，标明差值 6.6 个百分点与评测口径限制。
- 全部 500 段原始录像可从结果矩阵、任务表及失败案例直接在新标签页打开；独立逐轮播放器已移除，原始证据保留。
- 6 个可追溯失败案例，区分事实、解释、待验证改进；同任务成功 seed 回放对照。
- 终止原因、动作反馈频次、成功/失败决策次数分布、耗时散点和任务成功次数分布。
- 500 行 episode CSV、50 行任务 CSV、JSON、源 result/trace/seed/video SHA-256 清单。

耗时口径：52 次超时的 `elapsed_s` 未保存，345.5 秒仅为其余 **448 个有记录样本**的均值，不是完整 500 次均值。缺失值保留为空，页面标为超时而非填零。超时无最终观测时显示“最后保存的动作前观测”。成功率分母始终为 500。

## 操作过程回放与 harness 详尽过程

`#calls` 新增逐调用引导回放，直接读取同两份 workflow 数据。按观测 → 请求 → 原始返回 → 拒绝反馈（如有）→ harness 执行 → 环境反馈展开全部记录。模型请求计数只在 request 增加，已完成决策计数只在 feedback 增加；显示真实请求累计耗时。支持播放/暂停、阅读节奏、事件滑块、逐决策跳转、拒绝与零调用章节、URL 深链接和刷新恢复。左侧只在模型调用事件展示该请求实际附图，纯文本请求明确留空；执行阶段展示保存的环境观测，不推测视频同步。播放时长为阅读节奏，不是推理时间。新增交互验证：`node scripts/browser_calls.cjs`。

网页 `#workflow` 收录两个完整样本：成功的 `stack_blocks_three__00`（8 决策 / 10 调用）及失败的 `scan_object__00`（30 决策 / 37 调用）。可逐步查看三路 RGB、RGB-D 派生高度图、末端/接触状态、实际 prompt 和图片、VLM 原始输出、修复反馈、最终 harness action、全部运动子步骤与终局判定。包含原始 request/response、trace/result、114 份 RGB-D NPZ 及 SHA-256 下载。

调用归属按原始 request 与 observation 图像的文件保存时间恢复（精度为秒），并用 `action.raw` / `grounding_raw` 的 JSON 语义核验；没有冒充原生 call→step ID 或视频同步。初始 mission 的纯文本请求与不需要新模型调用的动作分别明确标注。高度图是 `world_z - table_z`（显示 0–0.4 m），并非额外输入给 VLM 的图像。

重建：在有 numpy / Pillow 的评测环境运行 `python3 scripts/build_workflow.py`；`python3 scripts/validate_site.py` 会一起检查这两份完整流程及其原始证据哈希。

## 发布流程

仓库 Settings → Pages → Source 设为 **GitHub Actions**。
推送到 `main` 后，`.github/workflows/pages.yml` 校验并发布 `site/`。
无后端、无需 API key 或 GitHub Secrets，图表库随网站托管。

本地预览：

```bash
python3 -m http.server 4175 --directory site
```

## 更新数据

源数据由本地评测工作区管理，不在 Actions 中运行机器人评测。
在有原始数据的机器上执行以下命令，检查后将 `site/` 的变化提交并推送：

```bash
python3 scripts/build_official500.py \
  --results ../phyRSI/robodawn_robotwin_harness_v72/results
python3 scripts/validate_site.py
```

构建环境需要 Pillow（`pip install Pillow`）。`export_site.py` 也转发到本轮构建器；旧 `build_data.py` 仅保留历史导出逻辑及任务族归类函数，不用于当前发布。

导出从每轮 result 与 trace 重建，核验总数、成功数、10 个不同 seed、6 个案例反馈计数。验证器核对全部 500 段视频哈希、21,453 个引用图像、CSV/JSON 一致性和 Pages 体积限制。旧版媒体可从 Git 历史恢复，新页面只引用 `media/official500/`。

浏览器验证（需安装 Playwright 与 Chromium）：

```bash
python3 -m http.server 4187 --directory site
# 在另一终端
node scripts/browser_check.cjs
```

`PLAYWRIGHT_MODULE` 可指定已有安装路径，`SITE_URL` 可指向其他本地预览地址。浏览器检查覆盖首页标题与对比、筛选、录像链接和实际播放、失败案例图像，以及 5 种屏幕宽度的溢出检查。媒体更新时设置 `CHECK_ALL_VIDEOS=1` 额外检查全部 500 段 MP4 元数据解码。保留的两个回放功能分别用 `scripts/browser_calls.cjs` 和 `scripts/browser_workflow.cjs` 检查。

发布页为静态快照，不连接本地评测服务器。
录像与动作没有逐步视频时间戳，页面提供独立播放与动作观测浏览，不推测同步。

图表使用 Apache-2.0 许可的 Apache ECharts 5.6.0，许可文本见 `site/vendor/ECHARTS-LICENSE.txt`。
