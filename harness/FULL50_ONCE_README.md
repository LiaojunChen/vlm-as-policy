# ZDTaichu + RoboTwin full50 单次评测

一键启动或复用两路本地 ZDTaichu 服务，并用 RoboDawn v72 对 50 个已注册
RoboTwin 任务各执行一次。

## 运行

```bash
cd /mnt/vepfs01/output/watson.chen/phyRSI/robodawn_robotwin_harness_v72
./run_full50_once.sh
```

默认结果目录：

```text
results/full50_once_<UTC timestamp>/
```

指定结果目录并在评测结束后保留由脚本启动的模型服务：

```bash
./run_full50_once.sh \
  --output results/my_full50_once \
  --keep-models
```

如果进程中断，可使用完全相同的 `--output` 重新运行。已完成任务不会被覆盖。

## 固定配置

- 50 个任务，每个任务 1 个 episode
- 每个任务使用 v72 `seeds/<task>/result.json` 中的专家验证 seed
- 最大策略决策次数：30
- 单 episode 超时：1200 秒
- GPU/服务：GPU 1 → 18052，GPU 0 → 18053
- 两个评测 worker
- 320×240、37° head camera
- JSON Schema 开启，`repair_only`

可以使用 `--max-steps` 和 `--timeout` 显式调整预算。若要保持正式 v72
单次评测协议，请保留默认值。

## 产物

- `summary.json`：聚合结果
- `results.csv`：每任务结果表
- `REPORT.md`：可读报告和证据链接
- `robodawn/<task>/result.json`：单任务结果
- `robodawn/<task>/trace.jsonl`：策略轨迹
- `robodawn/<task>/continuous.mp4`：执行视频
- `launcher/`：模型服务 PID、日志和本次启动配置

脚本不会终止或覆盖已存在的健康模型服务。若指定端口被不匹配的服务占用，
脚本会直接报错退出。
