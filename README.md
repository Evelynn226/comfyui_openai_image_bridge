# ComfyUI OpenAI Image Bridge

一个通过 **OpenAI 兼容 API** 调用图片模型的 ComfyUI 自定义节点，适用于 GPT Image、Nano Banana，以及其他提供兼容图片接口的服务。

## 节点

### 1. OpenAI 兼容 API 配置

- `base_url`：服务商提供的 API 根地址，通常填写到 `/v1`，例如 `https://api.openai.com/v1`
- `api_key`：服务商 API Key
- `timeout_seconds`：请求超时
- `verify_ssl`：是否校验 TLS 证书
- `extra_headers_json`：额外请求头 JSON 对象

配置节点输出 `api_config`，可连接到多个图片生成节点复用。

### 2. GPT Image / Nano Banana 图片生成

主要设置：

- `model`：模型 ID，默认 `gpt-image-2.6-sunburst`。可填写服务商提供的 GPT Image 或 Nano Banana 模型 ID，例如 `gpt-image-2.6-flare`、`gemini-2.5-flash-image` 等。模型是否可用取决于 API 服务商。
- `api_mode`：
  - `Images API`：请求 `/v1/images/generations`；连接 `input_image` 后改用 `/v1/images/edits`
  - `Chat Completions`：请求 `/v1/chat/completions`，适合通过聊天接口封装 Nano Banana 的服务商
- `output_format`：`png / jpeg / webp / gif / svg`
- `resolution_tier`：标准分辨率档位 `Auto / 1K / 1.5K / 2K / 4K`，以长边作为目标像素长度。
- `aspect_ratio`：常用比例预设：`1:1`、`3:2`、`2:3`、`4:3`、`3:4`、`16:9`、`9:16`、`5:4`、`4:5`、`21:9`。
- 选择具体档位时，插件会按档位和比例自动计算目标宽高，短边按 8 像素对齐；选择 `Auto` 时由模型自行决定输出尺寸。例如 1K 16:9 为 `1024x576`，2K 9:16 为 `1152x2048`，4K 1:1 为 `4096x4096`。
- `final_resolution` 输出会同时显示目标分辨率和 API 实际返回分辨率；可连接 ComfyUI 的 Show Text / 文本显示节点查看。
- `quality`：`auto / low / medium / high / xhigh / max`
- `background`：`auto / opaque / transparent`
- `n`：生成数量，1–8
- `output_compression`：JPEG / WebP 压缩质量
- `frame_count`、`frame_duration_ms`：GIF 帧数与每帧时长
- `complexity`：图像规划复杂度
- `svg_model`、`svg_reasoning`：服务商开放 SVG 模型选择时使用
- `send_modalities`：Chat Completions 模式是否发送 `modalities: ["text", "image"]`
- `send_image_options`：Chat Completions 模式是否额外发送 `output_format / size / quality / image_config`（默认关闭，避免严格网关拒绝非标准字段）。开启后会发送 `image_config.aspect_ratio` 与 `image_config.image_size`，适用于采用该参数约定的 Nano Banana 网关。
- `extra_parameters_json`：额外请求体参数，可覆盖或扩展默认字段
- `input_image`：可选参考图；Images API 模式下将调用编辑端点，Chat Completions 模式下以 data URL 形式传入
- `mask`：可选编辑蒙版，仅用于 Images API 编辑模式

### GPT Image 2.6 的格式支持

当前 GPT Image 图片接口定义了 PNG、JPEG、WebP、GIF、SVG 输出选项。GIF 可配置 2–16 帧及帧间隔；SVG 可通过 `svg_model` 和 `svg_reasoning` 指定 SVG 作者模型与推理强度（前提是服务商已开放相应能力）。这些选项是否实际可用仍取决于所连接的网关、模型权限和模型本身。

**注意：** ComfyUI 的标准 `IMAGE` 数据类型是静态栅格图像。因此：

- PNG/JPEG/WebP：保存原图，并输出可继续连接工作流的图像。
- GIF：完整动画 GIF 保存到输出目录；`IMAGE` 插口输出 GIF 第一帧预览。
- SVG：原始 SVG 保存到输出目录；`IMAGE` 插口输出栅格化预览。SVG 预览需要 CairoSVG。

## 安装

将整个 `comfyui_openai_image_bridge` 文件夹放入：

```text
ComfyUI/custom_nodes/comfyui_openai_image_bridge/
```

在 **ComfyUI 正在使用的 Python 环境** 安装依赖：

```bash
pip install -r requirements.txt
```

如需在 ComfyUI 画布上预览 SVG，再安装可选依赖：

```bash
pip install cairosvg
```

重启 ComfyUI，在节点菜单 `AI 图片生成/OpenAI 兼容` 中找到节点。

### Windows 便携版

在 ComfyUI 根目录运行，按实际路径调整：

```bat
.\python_embeded\python.exe -m pip install -r .\custom_nodes\comfyui_openai_image_bridge\requirements.txt
.\python_embeded\python.exe -m pip install cairosvg
```

## 配置示例

### GPT Image 2.6

- Base URL：`https://api.openai.com/v1`（或你所用服务商提供的兼容地址）
- API Mode：`Images API`
- Model：`gpt-image-2.6-sunburst` 或 `gpt-image-2.6-flare`
- Output Format：`png` / `gif` / `svg`

### Nano Banana（Chat Completions 网关示例）

- Base URL：填写服务商的 OpenAI 兼容根地址
- API Mode：`Chat Completions`
- Model：填写服务商给出的 Nano Banana 模型 ID
- `send_modalities`：开启
- `send_image_options`：若网关支持 `image_config` 参数约定则开启；插件会自动填入所选比例和 Auto/1K/1.5K/2K/4K 档位。若网关参数结构不同，可关闭此项并通过 `extra_parameters_json` 自行设置，例如：

```json
{
  "image_config": {
    "aspect_ratio": "16:9",
    "image_size": "2K"
  }
}
```

上面只是部分网关的参数格式示例，不是所有 OpenAI 兼容接口都支持它。

## 安全提醒

- API Key 如果直接填在节点里，会随 ComfyUI 工作流 JSON 一起保存/分享。分享工作流前请先清除密钥。
- 更安全的方式是将 API Key 放入环境变量：`OPENAI_COMPAT_API_KEY`（或 `OPENAI_API_KEY`），并将节点 `api_key` 留空。
- 插件只会向你填写的 `base_url` 发起 API 请求；对 API 响应中的远程图片 URL 下载时，不会转发 Authorization 请求头。
- `extra_parameters_json` 适合服务商私有扩展参数；请确认参数名与服务商文档一致。

## 输出

节点输出：

1. `images`：ComfyUI IMAGE batch
2. `masks`：透明区域对应的 ComfyUI MASK batch
3. `saved_files`：完整原始生成文件路径，每个文件一行
4. `final_resolution`：目标分辨率及实际返回分辨率文本

文件保存位置：`ComfyUI/output/openai_image_bridge/`

## 常见问题

**接口返回 400 / 404**  
检查 Base URL 是否填写正确、API Mode 与服务商支持的端点是否匹配，以及模型 ID 是否有权限。

**Chat Completions 没有输出图片**  
确认服务商的聊天接口确实返回图片，尝试开启 `send_modalities`，并按服务商文档配置 `extra_parameters_json`。

**GIF / SVG 请求失败**  
格式参数虽然已支持，但服务商必须将对应能力开放给所选模型和 API Key。部分网关只支持 PNG/JPEG/WebP。

**SVG 无法在画布预览**  
使用 ComfyUI 的 Python 环境安装 `cairosvg`。原始 SVG 文件会以 `.svg` 扩展名单独保存。


## v1.0.3 分辨率预览与自定义尺寸

- 节点内新增实时画幅预览：随分辨率档位、比例和自定义宽高变化。
- 预览显示画幅比例、目标宽×高像素、MP 总像素量。
- 新增「预设档位与比例 / 手动自定义尺寸」模式。
- 自定义宽高范围为 64–8192，步进 8；需为 8 的倍数。
- ComfyUI 画布上的 IMAGE 仍显示生成结果；预览框仅用于设置时查看目标画幅。


## v1.0.4 分辨率档位补充

- 新增 `1.5K` 档位，以长边 1536 px 为目标尺寸。
- 新增独立 `Auto` 尺寸档位：Images API 使用 `size: "auto"`，由模型决定实际输出尺寸。
- Chat Completions 开启 `send_image_options` 时，会在 `image_config.image_size` 中传递 `AUTO`，并保留所选比例作为网关提示。
- 实时预览在 Auto 模式下显示「模型自动决定尺寸」，比例仅作为参考，不再伪造具体像素数。
