# Gemini Robotics-ER 1.5 使用说明（识别 / 对比 / 图像生成）

这个项目目前包含三类核心系统：

- 图像识别与标注（通用物体 Pointing）
- 桌面四角点检测与几何推理对比（Pointing Benchmark）
- 提示词驱动的三维场景图像生成

本文档统一说明它们的安装与使用方法，便于快速复现。

## 1. 环境准备

### 1.1 Python 与依赖

建议使用 Python 3.10+，在项目根目录执行：

```powershell
pip install -r requirements.txt
```

如果要使用 `数据集/prompt_to_3d_scene_app.py`，还需要 Streamlit（通常已装；若未安装可补装）：

```powershell
pip install streamlit
```

### 1.2 环境变量（必须）

在 PowerShell 中设置：

```powershell
$env:GEMINI_API_KEY="你的密钥"
```

可选（中转接口）：

```powershell
$env:GOOGLE_GEMINI_BASE_URL="你的中转地址"
```

可选（指定模型）：

```powershell
$env:GEMINI_MODEL="gemini-3.1-pro-preview"
```

可选（图像生成模型）：

```powershell
$env:GEMINI_IMAGE_MODEL="gemini-2.5-flash-image"
```

---

## 2. 图像识别与标注系统

### 2.1 Streamlit 可视化版（推荐）

入口文件：`gemini_vision_ai.py`

作用：

- 对 `10_items.png` 做最多 10 个物体的 Pointing 检测
- 解析模型返回 JSON 并绘制标签点
- 输出标注图 `10_items_marked.png`

启动命令：

```powershell
streamlit run .\gemini_vision_ai.py
```

使用步骤：

1. 打开页面后点击“运行 Gemini 检测并标记”
2. 右侧会显示标注结果与 JSON
3. 结果图片会保存到项目根目录：`10_items_marked.png`

### 2.2 命令行快速版

入口文件：`gemini_robotics_er.py`

作用：

- 直接调用模型输出 Pointing JSON 文本
- 适合快速验证接口是否通

启动命令：

```powershell
python .\gemini_robotics_er.py
```

---

## 3. 桌角检测与几何推理系统（含对比）

这一部分有两类入口：单图交互检测 + 基准对比实验。

### 3.1 单图检测（交互式）

入口文件：`gemini_table_corner_vision_ai.py`

作用：

- 在 `数据集/` 中选择 `table1~table5` 等图像
- 检测桌面四角点（归一化 `[y, x]`）
- 绘制红点与红色四边形并保存结果图

启动命令：

```powershell
streamlit run .\gemini_table_corner_vision_ai.py
```

页面内可选项（关键）：

- 检测方式
  - `仅 Gemini`
  - `分割初定位 + Gemini 精修（推荐，需 opencv-python）`
- 角点推理策略（用于对比）
  - `含数学几何推理`
  - `仅大模型`
- 可勾选参考图 `table.png` 作为辅助

输出文件：

- 几何模式：`*_table_marked_geom.png`
- 仅大模型模式：`*_table_marked_llm.png`

默认保存在对应原图同目录（通常是 `数据集/`）。

### 3.2 基准对比实验（批量 + 指标）

目录：`A：具体的数学几何推理 + Pointing 的重点 Case`

#### 批量脚本

入口文件：`A：具体的数学几何推理 + Pointing 的重点 Case/run_pointing_comparison.py`

作用：

- 读取 `corner_dataset.json` 作为真值
- 对每张图分别跑：
  - `baseline_visual_only`
  - `geometry_reasoning`
- 输出核心指标：
  - `mean_corner_error`（越小越好）
  - `parallelogram_residual`（越小越好）

运行命令：

```powershell
python ".\A：具体的数学几何推理 + Pointing 的重点 Case\run_pointing_comparison.py"
```

只跑单图：

```powershell
python ".\A：具体的数学几何推理 + Pointing 的重点 Case\run_pointing_comparison.py" --image table1.png
```

结果输出到：

- `A：具体的数学几何推理 + Pointing 的重点 Case/results/comparison_results_*.json`
- `A：具体的数学几何推理 + Pointing 的重点 Case/results/comparison_results_*.csv`
- `A：具体的数学几何推理 + Pointing 的重点 Case/results/*_overlay.png`

#### 可视化对比页面

入口文件：`A：具体的数学几何推理 + Pointing 的重点 Case/pointing_experiment_app.py`

启动命令：

```powershell
streamlit run ".\A：具体的数学几何推理 + Pointing 的重点 Case\pointing_experiment_app.py"
```

页面会并排展示：

- Baseline（纯视觉）
- Geometry（数学几何推理）

并显示两组指标与原始模型输出，方便演示与分析。

---

## 4. 三维场景图像生成系统

入口文件：`数据集/prompt_to_3d_scene_app.py`

作用：

- 文本提示词生成三维空间场景图
- 支持上传参考图进行二次生成
- 支持风格与光照预设

启动命令：

```powershell
streamlit run .\数据集\prompt_to_3d_scene_app.py
```

页面流程：

1. 选择图像模型（如 `gemini-2.5-flash-image`）
2. 输入场景提示词与补充要求
3. 可上传参考图（可选）
4. 点击“生成三维空间图片”
5. 预览后点击“保存当前图片”

保存目录：

- `数据集/generated_3d_scenes/generated_3d_scene_时间戳.png`

---

## 5. 常见问题排查

### 5.1 403 / 权限错误

常见原因：

- `GEMINI_API_KEY` 错误或过期
- 中转站不支持当前模型
- `GOOGLE_GEMINI_BASE_URL` 不是 Gemini / google-genai 兼容地址
- 账户额度或白名单限制

### 5.2 OpenCV 不可用

如果桌角检测页面提示分割不可用，执行：

```powershell
pip install opencv-python numpy
```

### 5.3 未生成图片文件

在三维生成系统中，点击“生成”只会预览；需要再点击“保存当前图片”才会写入 `generated_3d_scenes`。

---

## 6. 最小复现命令清单

```powershell
# 1) 识别标注
streamlit run .\gemini_vision_ai.py

# 2) 桌角检测（单图）
streamlit run .\gemini_table_corner_vision_ai.py

# 3) 几何推理对比（批量）
python ".\A：具体的数学几何推理 + Pointing 的重点 Case\run_pointing_comparison.py"

# 4) 几何推理对比（可视化）
streamlit run ".\A：具体的数学几何推理 + Pointing 的重点 Case\pointing_experiment_app.py"

# 5) 三维场景图生成
streamlit run .\数据集\prompt_to_3d_scene_app.py
```
