# BlenderProc 数据生成说明

## 1. 这个文件夹的用途

本目录用于使用 BlenderProc 批量生成几何场景数据，并自动导出与图片一一对应的 GT 标注（JSON）。

当前管线特点：
- 可批量生成多目标几何场景（立方体/圆柱/圆锥/球体）
- 含随机相机视角、随机材质、随机光照、遮挡体
- 每张图自动导出对应 GT（相机参数 + 几何投影结果）
- 输出目录按运行时间隔离，方便追踪与复现

---

## 2. 核心原理（简述）

BlenderProc 在渲染时已知完整 3D 场景与相机参数，因此可以直接计算 GT，而不是后处理猜测：
- RGB：渲染图像
- Depth：每像素深度
- Segmentation：像素级分割（不同版本支持情况略有差异）
- Projection GT：将 3D 目标点投影到 2D 像素坐标

这套流程适合你在问题 B 中的自动标注/3D 投影标注方向。

---

## 3. 快速开始

在本目录运行：

```powershell
blenderproc run generate.py
```

可配置参数（环境变量）：

```powershell
$env:BPROC_NUM_SAMPLES=120      # 生成样本数，默认 24
$env:BPROC_SEED=42              # 随机种子，默认 42
$env:BPROC_WIDTH=768            # 图像宽，默认 768
$env:BPROC_HEIGHT=768           # 图像高，默认 768
$env:BPROC_RUN_NAME=my_run_name # 可选：指定输出目录名
blenderproc run generate.py
```

---

## 4. 输出目录结构

每次运行会在 `output/` 下创建一个数据集目录：

```text
output/
  geom_dataset_YYYYMMDD_HHMMSS/
    images/
      sample_00000.png
      sample_00001.png
      ...
    gt_json/
      sample_00000.json
      sample_00001.json
      ...
    hdf5/
      0.hdf5
      1.hdf5
      ...
    manifest.json
```

命名约定：
- `images/sample_xxxxx.png`
- `gt_json/sample_xxxxx.json`

二者一一对应。

---

## 5. GT JSON 字段说明（重点）

每个 `sample_xxxxx.json` 常见字段如下：

- `sample_id`：样本编号
- `image_file` / `gt_file`：图片与标注文件名
- `image_size_wh`：图像尺寸
- `cam2world`：相机外参（4x4）
- `k_matrix`：相机内参（3x3）
- `instances`：当前样本的目标列表（可直接用于 Pointing）
  - `target_id`
  - `shape_type`
  - `centroid_uv`：2D 像素中心点
  - `bbox_xyxy`：2D 包围框
  - `in_image`：是否在图像范围内
  - `visible_by_depth`：基于深度的可见性判断
- `projected_targets`：更完整的投影结果（包含 3D 点与 2D 投影）

说明：
- 某些 BlenderProc 版本下 segmentation 可能为空，本脚本已内置回退逻辑，优先保证 `instances/projected_targets` 始终可用。

---

## 6. 常见问题

### Q1: 图片发灰/看不到目标？
- 典型原因是相机朝向或场景采样问题。
- 当前脚本已修复相机朝向逻辑；若仍异常，建议：
  - 减小相机随机范围
  - 提高目标尺寸下限
  - 检查 `BPROC_SEED` 是否导致极端样本

### Q2: JSON 只有很少字段（如 12 行）？
- 那是早期空壳样本或旧脚本产物。
- 请使用当前脚本重新生成，或查看最新 `geom_dataset_*` 目录。

### Q3: 如何验证图片与 GT 对齐？
- 检查同名文件：`sample_00000.png` 对应 `sample_00000.json`
- 在 JSON 中查看 `centroid_uv`、`bbox_xyxy` 并可视化叠加验证

---

## 7. 建议的下一步

- 增加 `quality_report.json`（空 GT 率、越界率、可见目标比例）
- 增加关系级 GT（相对方位、距离、角度）用于几何推理 benchmark
- 增加语言指令模板（把几何 GT 转成问答/pointing 指令）

