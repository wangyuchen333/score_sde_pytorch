大多数模型现在也可以在 🧨 Diffusers 中使用，并通过 [ScoreSdeVE 管道](https://huggingface.co/docs/diffusers/api/pipelines/score_sde_ve) 访问。

然后您可以用几行代码尝试这些模型：

```python
from diffusers import DiffusionPipeline

model_id = "google/ncsnpp-ffhq-1024"

# 加载模型和调度器
sde_ve = DiffusionPipeline.from_pretrained(model_id)

# 在推理中运行管道（采样随机噪声并去噪）
image = sde_ve().images[0]

# 保存图像
image[0].save("sde_ve_generated_image.png")
```

更多模型可以直接在 [Hub](https://huggingface.co/models?library=diffusers&pipeline_tag=unconditional-image-generation&sort=downloads&search=ncsnpp) 上找到。

## 如何运行代码

请查看 [#5](https://github.com/yang-song/score_sde/pull/5) 以了解如何为新数据集计算此统计文件。

### 用法

通过 `main.py` 训练和评估我们的模型。

```sh
main.py:
  --config: 训练配置。
    （默认值: 'None'）
  --eval_folder: 存储评估结果的文件夹名称
    （默认值: 'eval'）
  --mode: <train|eval>: 运行模式：训练或评估
  --workdir: 工作目录
```

* `config` 是配置文件的路径。我们提供的配置文件位于 `configs/` 中，格式符合 [`ml_collections`](https://github.com/google/ml_collections)，应当相当易于理解。

  **配置文件的命名约定**：配置文件的路径是以下维度组合的结果：
  * 数据集：`cifar10`、`celeba`、`celebahq`、`celebahq_256`、`ffhq_256`、`celebahq`、`ffhq` 之一。
  * 模型：`ncsn`、`ncsnv2`、`ncsnpp`、`ddpm`、`ddpmpp` 之一。
  * 连续：使用连续采样的时间步训练模型。

* `workdir` 是存储一个实验的所有工件的路径，如检查点、样本和评估结果。

* `eval_folder` 是 `workdir` 中一个子文件夹的名称，存储评估过程的所有工件，如防止预占的元检查点、图像样本和定量结果的 numpy 转储。

* `mode` 是 "train" 或 "eval"。当设置为 "train" 时，它开始训练一个新模型，或在其元检查点存在的情况下恢复旧模型的训练。当设置为 "eval" 时，可以进行任意组合的以下操作：

  * 在测试/验证数据集上评估损失函数。

  * 生成固定数量的样本并计算其 Inception 分数、FID 或 KID。在评估之前，必须已下载/计算并存储统计文件到 `assets/stats`。

  * 计算训练或测试数据集上的对数似然。

  这些功能可以通过配置文件进行配置，或者通过 `ml_collections` 包的命令行支持更方便地进行配置。例如，要生成样本并评估样本质量，提供 `--config.eval.enable_sampling` 标志；要计算对数似然，提供 `--config.eval.enable_bpd` 标志，并指定 `--config.eval.dataset=train/test` 以指示是否在训练或测试数据集上计算对数似然。

## 如何扩展代码
* **新 SDEs**：继承 `sde_lib.SDE` 抽象类并实现所有抽象方法。`discretize()` 方法是可选的，默认是 Euler-Maruyama 离散化。现有的采样方法和似然计算将自动适用于此新 SDE。
* **新预测器**：继承 `sampling.Predictor` 抽象类，实现 `update_fn` 抽象方法，并用 `@register_predictor` 注册其名称。新的预测器可以直接用于 `sampling.get_pc_sampler` 进行预测-校正采样，以及所有其他在 `controllable_generation.py` 中的可控生成方法。
* **新校正器**：继承 `sampling.Corrector` 抽象类，实现 `update_fn` 抽象方法，并用 `@register_corrector` 注册其名称。新的校正器可以直接用于 `sampling.get_pc_sampler`，以及所有其他在 `controllable_generation.py` 中的可控生成方法。

## 预训练检查点
所有检查点都提供在此 [Google Drive](https://drive.google.com/drive/folders/1tFmF_uh57O6lx9ggtZT_5LdonVK2cV-e?usp=sharing)。

**说明**：对于某些模型，您可能会发现两个检查点。第一个检查点（编号较小）是我们在论文的表 3 中报告 FID 分数的检查点（也对应于下表中的 FID 和 IS 列）。第二个检查点（编号较大）是我们在论文的表 2 中报告黑盒 ODE 采样器的似然值和 FID 的检查点（也对应于下表中的 FID(ODE) 和 NNL (bits/dim) 列）。前者对应于训练过程中每 50k 次迭代时的最小 FID。后者是训练过程中的最后一个检查点。

根据 Google 的政策，我们无法发布我们的原始 CelebA 和 CelebA-HQ 检查点。也就是说，我用个人资源重新训练了在 FFHQ 1024px、FFHQ 256px 和 CelebA-HQ 256px 上的模型，并取得了与我们内部检查点相似的性能。

以下是检查点及其在论文中报告的结果的详细列表。**FID (ODE)** 对应于应用于概率流 ODE 的黑盒 ODE 解算器的样本质量。

| 检查点路径 | FID | IS | FID (ODE) | NNL (bits/dim) |
|:----------|:-------:|:----------:|:----------:|:----------:|
| [`ve/cifar10_ncsnpp/`](https://drive.google.com/drive/folders/1sP4GwvrYiI-sDPTp7sKYzsxJLGVamVMZ?usp=sharing) |  2.45 | 9.73 | - | - |
| [`ve/cifar10_ncsnpp_continuous/`](https://drive.google.com/drive/folders/1b0gy_LLgO_DaQBgoWXwlVnL_rcAUgREh?usp=sharing) | 2.38 | 9.83 | - | - |
| [`ve/cifar10_ncsnpp_deep_continuous/`](https://drive.google.com/drive/folders/11s6A_xM7qiztdj8AHQWqaIAUSC3I7uX2?usp=sharing) | **2.20** | **9.89** | - | - |
| [`vp/cifar10_ddpm/`](https://drive.google.com/drive/folders/1zDKcy3xbsN3F4AfyB_DfY_1oho89iKcf?usp=sharing) | 3.24 | - | 3.37 | 3.28 |
| [`vp/cifar10_ddpm_continuous`](https://drive.google.com/drive/folders/1RHNxW1qY-mTr0JMAE5t4V181Hi_aVWXK?usp=sharing) | - | - | 3.69| 3.21 |
| [`vp/cifar10_ddpmpp`](https://drive.google.com/drive/folders/1zOVj03ZBcq339p5QEKJPh2bBrxR_HOCM?usp=sharing) | 2.78 | 9.64 | - | - |
| [`vp/cifar10_ddpmpp_continuous`](https://drive.google.com/drive/folders/1xYjVMx10N9ivQQBIsEoXEeu9nvSGTBrC?usp=sharing) | 2.55 | 9.58 | 3.93 | 

3.16 |
| [`vp/cifar10_ddpmpp_deep_continuous`](https://drive.google.com/drive/folders/1ZMLBiu9j7-rpdTQu8M2LlHAEQq4xRYrj?usp=sharing) | 2.41 | 9.68 | 3.08 | 3.13 |
| [`subvp/cifar10_ddpm_continuous`](https://drive.google.com/drive/folders/1DeebpmBkCxlZx89t3z45Te37T7BPOzd2?usp=sharing) | - | - | 3.56 | 3.05 |
| [`subvp/cifar10_ddpmpp_continuous`](https://drive.google.com/drive/folders/1bLgmnEAZnysRZfWt8qN3omGfijJ_B884?usp=sharing) | 2.61 | 9.56 | 3.16 | 3.02 |
| [`subvp/cifar10_ddpmpp_deep_continuous`](https://drive.google.com/drive/folders/16QGkviGcizSbIPRk37-YksUhlNIna4Ys?usp=sharing) | 2.41 | 9.57 | **2.92** | **2.99** |

| 检查点路径 | 样本 |
|:-----|:------:|
| [`ve/bedroom_ncsnpp_continuous`](https://drive.google.com/drive/folders/18GmxDvfGR8se9uFucc9uweeVrX_GzuUG?usp=sharing) | ![bedroom_samples](assets/bedroom.jpeg) |
| [`ve/church_ncsnpp_continuous`](https://drive.google.com/drive/folders/1zVChA0HrnJU66Jkt4P6KOnlREhBMc4Yh?usp=sharing) | ![church_samples](assets/church.jpeg) |
| [`ve/ffhq_1024_ncsnpp_continuous`](https://drive.google.com/drive/folders/1ZqLNr_kH0o9DxvwSlrQPMmkrhEnXhBm2?usp=sharing) |![ffhq_1024](assets/ffhq_1024.jpeg)|
| [`ve/ffhq_256_ncsnpp_continuous`](https://drive.google.com/drive/folders/1KG72ZKUCUa8dDcA03hOf1BsnK8kBcdPD?usp=sharing) |![ffhq_256_samples](assets/ffhq_256.jpg)|
| [`ve/celebahq_256_ncsnpp_continuous`](https://drive.google.com/drive/folders/19VJ7UZTE-ytGX6z5rl-tumW9c0Ps3itk?usp=sharing) |![celebahq_256_samples](assets/celebahq_256.jpg)|


## 演示和教程
| 链接 | 描述 |
|:----:|:-----|
|[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17lTrPLTt_0EDXa4hkbHmbAFQEkpRDZnh?usp=sharing) | 加载我们的预训练检查点并玩转采样、似然计算和可控合成（PyTorch） |
|[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)| PyTorch 中基于分数的生成模型教程 |


## 小贴士
* `LangevinCorrector` 的 `snr`（信噪比）参数在某种程度上表现得像温度参数。较大的 `snr` 通常会导致更平滑的样本，而较小的 `snr` 则会产生更多样但质量较低的样本。`snr` 的典型值为 `0.05 - 0.2`，需要调优以找到最佳平衡点。
* 对于 VE SDE，我们建议选择 `config.model.sigma_max` 为训练数据集中数据样本之间的最大成对距离。
