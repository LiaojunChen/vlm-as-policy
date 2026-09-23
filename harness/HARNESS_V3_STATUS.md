# RoboTwin harness 优化与评测入口

仅使用 **Qwen3-VL-Plus**，Codex 保持停止。代码在独立拷贝中修改。2026-09-16 17:03 UTC 快照：原相机固定版本 v3p 完成 **25/100**，成功 **10**；双方均完成的 24 个同种子 episode 中，v2 成功 4 个，v3p 成功 9 个。全量尚未完成，不能将局部比例当成最终成功率。

## 固定全量评测

- [实时对照报告](../robodawn_robotwin_harness_v3p_eval/results/full_50/COMPARISON.md)
- [交付分类后的完整 CSV](../robodawn_robotwin_harness_v3p_eval/results/full_50/delivery_results.csv)
- [交付分类后的完整 JSON](../robodawn_robotwin_harness_v3p_eval/results/full_50/delivery_summary.json)
- [固定任务清单](../robodawn_robotwin_harness_v3p_eval/results/full_50/manifest.json)

50 官方任务 × 两项目 × 每任务 1 episode，demo_clean，30 次策略决策、1200 秒。原头部相机 320×240、37°。使用与 v2 相同的已验证开发种子，**不是留出种子的泛化成绩**。所有失败保留；策略失败、格式错误和预算耗尽不会择优重跑，只允许基础设施错误重试。

每个 episode 保留结果、逐步轨迹、模型请求/响应、RGB/RGB-D、连续视频与日志。后验物体位姿诊断只在策略结束后保存，不进入策略输入。

## 已落实的设计

- 根据 RGB-D 估计抓取中心、尺寸与方向；短物体优先顶部抓取，搜索可达朝向。
- 对开合夹爪进行物理收敛等待，以实际双指接触确认持物；转移中失去物体会停止放置阶段。
- 预检并复用可达轨迹，保留持物手臂；利用初始可见垫子的中心和长轴提高放置精度。
- Show-Harness 保留原生任务分解与双手控制器，明确语义动作枚举，阶段只在执行确认后推进；分离任务规划 JSON 和动作参数 JSON。
- 新增把手抓持、铰链圆弧和桌面推移；使用公共任务完成条件提示，不使用专家动作或运行时物体真实位姿。
- 使用机器人自身 link ID 过滤深度中的机械臂，只提供匿名接触关系，不提供物体分割身份。

最新 v3w 进一步修正三项有轨迹证据的问题：定位模型不得改写规划器指定的手臂；叠放时从可见方块上表面估计高度，不能因模型误报 table 而投影到桌面；夹爪打开必须持续下发目标，配合 200 个物理步，使官方接口的每次 10% 开度限制逐步收敛。

执行效率优化：静止手臂直接保持实测关节，避免重复 IK；本体状态读取不再触发三相机渲染；跳过随后会被重置清空的图规划预热，保留按需图搜索；同一 worker 复用静态规划器。

## 独立实验结果

不同版本不可拼接成一个分数。

| 实验 | 完成 / 计划 | 成功 | 说明 |
|---|---:|---:|---|
| v3p 标准相机全量 | 25 / 100 | 10 | 运行中，最新数据见实时报告 |
| dev_r 接口增强 | 4 / 4 | 3 | 鼠标放垫、开笔记本、面包入篮成功；叠方块失败；多个变化一起测试，不能归因于单一因素 |
| v3s 推移初测 | 4 / 4 | 1 | 嵌套 target 导致动作格式错误 |
| v3t 修正格式 | 4 / 4 | 2 | 两项目按铃通过，移杯超时；杯子大部分在原相机视野外 |
| v3u 宽视野实验 | 4 / 4 | 3 | 两项目按铃通过；Robodawn 移杯 4 次决策成功，Show-Harness 移杯超时 |
| v3v 执行器回归 | 进行中 | 暂不计分 | 抓取后暴露 NumPy 整数序列化缺陷，保留错误结果，未进入全量 |
| v3w 最新冻结版 | 待验证 | 待验证 | 新执行器修正；960×720、60°增强感知，保持原相机位姿 |

v3u、v3v、v3w 更改了头部相机视野，属于**增强感知实验**，不能宣称与原相机设置完全等价。仍使用官方物理环境和成功判定。

## 最新版自动评测安排

[冻结源码与配置](../robodawn_robotwin_harness_v3w_eval/FROZEN_VERSION.json)；[队列状态](../robodawn_robotwin_harness_v3w_eval/EVALUATION_QUEUE.json)。

v3w 修复完整观测中 NumPy 夹爪标量的序列化错误，等待上一回归队列释放资源后，先测叠方块、按铃 × 两项目，共 4 个 episode；没有基础设施错误且至少 2 个成功后，等待原相机全量队列结束，再运行新版 50×2 全量。先导结果全部保留，新版全量另设目录、每组合只取一次，不拼接历史最佳结果。未通过启动条件则保留报告、等待进一步审查。

## 验证及剩余问题

36 项接口/回归检查通过（包含完整观测 JSON 边界测试）。真实仿真验证了机器人自遮挡过滤、本体状态与官方观察一致、静止手臂保持、夹爪闭合后重新打开到约 43.6 mm：见 [物理验证](results/robot_self_mask_probe/validation.json)。

[完整性审计](EVALUATION_INTEGRITY.json)：已记录的原项目 41 个源码文件未变；50 个官方成功判定未变；v3p 冻结的 42 个源码文件未变。

尚未解决全部复杂操作：双臂同步抬升、可靠交接、遮挡下的精细装配与定向放置仍有限制。宽视野解决了部分不可见目标，但不能保证抓取、接触运动和放置成功。v3q 的抓后 ICP 曾因遮挡错误拟合造成退化，默认禁用，失败证据保留。

## 成功视频示例

- [标准相机双瓶举起](../robodawn_robotwin_harness_v3p_eval/results/full_50/robodawn/pick_dual_bottles/continuous.mp4)
- [开笔记本](results/dev_r/robodawn/open_laptop/continuous.mp4)
- [面包入篮](results/dev_r/robodawn/place_bread_basket/continuous.mp4)
- [宽视野移杯](../robodawn_robotwin_harness_v3u_eval/results/wide_view_smoke/robodawn/place_empty_cup/continuous.mp4)
