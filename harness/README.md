# RoboDawn / Show-Harness RoboTwin evaluation

两套策略分别维护在 `robodawn/` 和 `showharness/`；共用的仿真执行、感知、动作校验和评测调度保留在根目录。

```text
robodawn/
  core.py                 # see-think-act 循环、协议和离线 baseline
  main.py, __main__.py     # 离线 demo / 模型调用 demo
  planner_v1.py           # 原始 primitive-action planner
  planner_v2.py           # grounded planner
  planner_v3.py           # v3 visual skill planner
  reviewed.py             # reviewed diagnostic planner
showharness/
  planner_v3.py           # v3 native planner/controller 适配
  policy_v1.py             # 原始双臂策略与 fine-control prompts
  policy_v2.py             # grounded stage policy
  reviewed.py              # reviewed diagnostic policy
  runtime.py               # 上游模块和 prompt 的统一路径定位
  upstream/                # 原 Show-Harness/，内容保持不变
robotwin_harness_v3.py      # 共用 v3 Bridge、动作契约和视觉定位
robotwin_bridge.py          # 共用 v1 Bridge
robotwin_grounding_v2.py     # 共用 v2 grounding / executor
reviewed_interface.py       # 共用 reviewed 动作契约
benchmark_clients.py        # 共用模型调用和审计，复用 upstream VLMClient
run_harness_v3.py           # 单 episode 入口
run_suite_v3.py             # suite 调度
run_cohort_v3.py            # cohort 调度
source_layout.py           # 源码冻结与交付清单
RoboTwin/, seeds/, results/ # 共用仿真、种子和历史结果
```

新代码直接从各自的包导入，例如：

```python
from robodawn.planner_v3 import RobodawnPlanner
from showharness.planner_v3 import ShowPlanner
from robotwin_harness_v3 import Bridge
```

两套策略可以独立导入，无需先导入 `benchmark_clients` 设置 Show-Harness 路径。共享客户端继续复用上游 transport；策略拆分不改变动作、模型或评分逻辑。根目录的旧策略模块仅保留兼容导出，混合模块中的旧类名也继续可用。

在本目录运行，原有参数和项目标识保持一致：

```bash
python -m robodawn --provider mock
python run_harness_v3.py --project robodawn --task click_bell --output results/new_run/robodawn/click_bell
python run_harness_v3.py --project show_harness --task click_bell --output results/new_run/show_harness/click_bell
python -m pytest tests test_fine_control.py -q
```

仿真评测需要安装 RoboTwin 依赖并配置模型服务。工作区现有 Python 环境为 `../robodawn_robotwin/.venv-robotwin310/bin/python`，可替换上述 `python`。

源码冻结会覆盖两个策略包、上游 Python/config/prompt 文件及共享模块。拆分后的源码应使用新的评测输出目录；历史结果和原有哈希清单保持原样，不与新代码混用。
