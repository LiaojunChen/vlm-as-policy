# VLM as Policy v72 · RoboTwin results

网站：[VLM as Policy](https://liaojunchen.github.io/vlm-as-policy/) · 仓库：[LiaojunChen/vlm-as-policy](https://github.com/LiaojunChen/vlm-as-policy)

参考 [robodawn.top](https://robodawn.top) 的内容组织与分析方式，展示本地 v72 的真实评测结果。

主批次：`repeat500_official_sampling_maxsteps30_20260922`，50 个任务 × 10 个不同接受 seed，233/500 成功，成功率 **46.6%**，0 基础设施错误。2026-09-23 03:48:09 UTC 完成。
模型为 `phyRSI/ZDTaichu5.0-9B`；`demo_clean`、30 决策预算、1200 秒墙钟预算。按 RoboTwin 官方方式从 seed 100000 顺序扫描，专家验证通过后取每任务前 10 个 seed（官方常规为每任务 100 次）。与官网 C2R 不同评测口径，不构成同榜排名。历史开发批次 36/50（72%）单独展示，不能作为同 seed 配对比较。

## 网站功能

- 50×10 可点击结果矩阵；50 个任务筛选、排序、逐轮入口。
- 全部 500 段原始录像、倍速、episode 深链接；按需加载动作轨迹和三路相机无损 WebP 观测。
- 6 个可追溯失败案例，区分事实、解释、待验证改进；同任务成功 seed 回放对照。
- 终止原因、动作反馈频次、成功/失败决策次数分布、耗时散点和任务成功次数分布。
- 500 行 episode CSV、50 行任务 CSV、JSON、源 result/trace/seed/video SHA-256 清单。

耗时口径：52 次超时的 `elapsed_s` 未保存，345.5 秒仅为其余 **448 个有记录样本**的均值，不是完整 500 次均值。缺失值保留为空，页面标为超时而非填零。超时无最终观测时显示“最后保存的动作前观测”。成功率分母始终为 500。

## 发布

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

`PLAYWRIGHT_MODULE` 可指定已有安装路径，`SITE_URL` 可指向其他本地预览地址。浏览器检查覆盖筛选、轮次切换、深链接、失败案例图像、视频实际播放、500 段 MP4 元数据解码和移动端溢出。

发布页为静态快照，不连接本地评测服务器。
录像与动作没有逐步视频时间戳，页面提供独立播放与动作观测浏览，不推测同步。

图表使用 Apache-2.0 许可的 Apache ECharts 5.6.0，许可文本见 `site/vendor/ECHARTS-LICENSE.txt`。
