# 数学几何推理 + Pointing 对比实验

这个目录用于做一个清晰的对比实验：

- `baseline_visual_only`：只让大模型凭视觉直接判断桌面四角
- `geometry_reasoning`：显式加入平行四边形、对角线中心、三点推第四点等数学几何推理

目标是回答一个核心问题：

在同一个模型、同一张图、同一套输出格式下，加入几何推理后，桌角定位是否更稳定、更准确、更符合结构约束？

## 文件说明

- [run_pointing_comparison.py](</D:/大学/计算机/浙大wwg组面试/Gemini Robotics-ER 1.5/A：具体的数学几何推理 + Pointing 的重点 Case/run_pointing_comparison.py>)：批量跑实验，输出 `json/csv` 结果，并生成可视化 overlay 图
- [pointing_experiment_app.py](</D:/大学/计算机/浙大wwg组面试/Gemini Robotics-ER 1.5/A：具体的数学几何推理 + Pointing 的重点 Case/pointing_experiment_app.py>)：Streamlit 可视化页面，适合演示
- `results/`：实验结果输出目录

## 当前使用的数据

当前真值来自项目根目录的 [corner_dataset.json](</D:/大学/计算机/浙大wwg组面试/Gemini Robotics-ER 1.5/corner_dataset.json>)。

现在至少包含：

- `table1.png`

如果你后续补充更多标注，这套实验会自动扩展到更多图片。

## 指标设计

实验里默认输出两个核心指标：

- `mean_corner_error`
  4 个角点预测与真值之间的平均欧氏距离，越小越好
- `parallelogram_residual`
  检查 `corner_0 + corner_2` 与 `corner_1 + corner_3` 的差距，越小越符合几何结构

这两个指标正好对应两类能力：

- 准确定位能力
- 几何一致性能力

## 运行前准备

在 PowerShell 中设置：

```powershell
$env:GEMINI_API_KEY="你的key"
```

如果你使用中转接口，还需要：

```powershell
$env:GOOGLE_GEMINI_BASE_URL="你的中转地址"
```

可选模型设置：

```powershell
$env:GEMINI_MODEL="gemini-3.1-pro-preview"
```

## 运行方式

批量跑：

```powershell
python ".\A：具体的数学几何推理 + Pointing 的重点 Case\run_pointing_comparison.py"
```

只跑单张图：

```powershell
python ".\A：具体的数学几何推理 + Pointing 的重点 Case\run_pointing_comparison.py" --image table1.png
```

打开可视化页面：

```powershell
streamlit run ".\A：具体的数学几何推理 + Pointing 的重点 Case\pointing_experiment_app.py"
```

## 结果如何解释

如果 `geometry_reasoning` 同时表现出：

- 更低的 `mean_corner_error`
- 更低的 `parallelogram_residual`

那么就可以较强地说明：

数学几何推理不仅提升了定位精度，也提升了结构一致性，这正是“几何推理对大模型视觉识别定位有帮助”的证据。

## 关于你截图里的 403 错误

`403 Forbidden` 不是代码逻辑报错，而是接口权限被拒绝。常见原因：

- `GEMINI_API_KEY` 不正确
- 中转站不支持这个模型或这类请求
- 中转地址不是 Gemini / `google-genai` 兼容接口
- 额度、白名单或鉴权限制

如果需要，我下一步可以继续帮你补：

- 多图片汇总统计表
- 自动生成实验报告 markdown
- 更强的 bad case 标注与截图导出
