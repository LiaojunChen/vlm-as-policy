# v72 代码快照

`harness/` 同步自 `phyRSI/robodawn_robotwin_harness_v72`，包括根目录工具、策略包、Show-Harness 上游代码与许可、RoboTwin 环境及 cuRobo 源码、配置、50 个专家验证 seed、测试和启动脚本。逐文件 SHA-256 见 `harness-source-manifest.json`。

源代码与配置按原始字节同步，原有包名、导入路径、项目参数及第三方归属保留，避免更改已评测实现。网页使用 VLM as Policy 展示名称；原始证据下载保持原样。

## 外部资源

仓库同步的是代码快照，以下资源单独准备：

- RoboTwin 对象、背景、机器人 mesh 资产（DAE / STL / OBJ / GLB 等）；其中两个本地 DAE 文件各约 205 MiB，超过 GitHub 单文件限制。已有 URDF 与相关 YAML 配置保留，需将对应 mesh 恢复到原相对位置。
- 源目录通过软链接共享的 `RoboTwin/assets/{background_texture,files,objects}`，以及运行时 ffmpeg。
- 模型权重、CUDA/仿真 Python 环境、编译生成的 `.so`；cuRobo 源码需按运行环境重新安装编译。
- 原始 `results/`、缓存和 PID；网站公开的评测证据继续保留在 `site/`。

完整排除路径及原因见清单中的 `excluded`。上游安装入口包括 `harness/RoboTwin/scripts/_install.sh`、`_download_assets.sh` 和 `requirements.txt`；先阅读并配置实际环境与资源路径，再执行仿真。

## 运行入口

在装好依赖的环境中：

```bash
cd harness
python -m robodawn --provider mock
python -m pytest tests test_fine_control.py -q
```

`run_harness_v3.py`、`run_suite_v3.py`、`run_cohort_v3.py` 是 episode/suite/cohort 入口。`run_full50_once.sh` 是原工作区的单次 50 任务启动器，默认查找相邻的模型和仿真环境；外部机器需按脚本配置相应路径。详见 `harness/README.md`、`harness/FULL50_ONCE_README.md` 与 `harness/runtime/env.sh`。

这是当前 v72 源目录的快照，不声明与历史每个结果批次的代码哈希相同。复核具体实验时，以对应批次的 `code_hashes.json` 为准。

## 重新同步

```bash
python3 scripts/sync_harness.py --source ../phyRSI/robodawn_robotwin_harness_v72
```

脚本复制原始文件，生成清单，不修改评测源目录。发布流程仅上传 `site/`，代码不会增加 Pages 部署体积。
