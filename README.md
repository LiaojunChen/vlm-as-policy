# VLM as Policy v72 · RoboTwin results

参考 [robodawn.top](https://robodawn.top) 的内容组织与分析方式，展示本地 v72 的真实评测结果。

主批次：`full50_completion_review_restored_protocol_standard_20260922`，50 个任务，36 次成功，成功率 72%，0 运行错误。
模型为 `phyRSI/ZDTaichu5.0-9B`；`demo_clean`、固定开发 seed、每任务一次、30 决策预算、1200 秒墙钟预算。与官网 C2R 不同评测口径，不构成同榜排名。

## 网站功能

- 全部任务筛选、排序、官方结果与耗时。
- 50 段原始录像、逐步动作反馈、头部与左右腕相机观测。
- 失败终止原因、动作级失败反馈、恢复与耗时分析。
- 500 次同 seed 重复实验的发布快照；该实验使用 20 决策预算，独立展示。
- CSV、JSON 与原始 result/trace 文件 SHA-256 清单下载。

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
python3 scripts/export_site.py \
  --source ../robodawn-v72-observatory \
  --results ../phyRSI/robodawn_robotwin_harness_v72/results
python3 scripts/validate_site.py
```

导出保留主批次已核验结果，并重新读取重复实验汇总。发布页刷新按钮只加载最近发布的 JSON，不连接本地评测服务器。
录像与动作没有逐步视频时间戳，页面提供独立播放与动作观测浏览，不推测同步。

图表使用 Apache-2.0 许可的 Apache ECharts 5.6.0，许可文本见 `site/vendor/ECHARTS-LICENSE.txt`。
