# audio.cpp WebUI 启动脚本说明

仓库根目录下的一组 `.bat` 脚本，覆盖 audio.cpp 的三种本地运行方式：命令行单句合成、
HTTP API 服务、图形界面。所有脚本都可以**双击运行**，也可以在命令行/PowerShell
里带参数调用。

| 脚本 | 作用 | 典型命令 |
|---|---|---|
| `run_cli_tts.bat` | 单句/单次命令行 TTS | `run_cli_tts.bat qwen3-tts "你好世界"` |
| `run_server.bat` | OpenAI 兼容 HTTP API 服务 | `run_server.bat qwen3-tts 8080` |
| `run_server_asr.bat` | Qwen3-ASR 服务（`run_server.bat` 的 ASR 预设） | `run_server_asr.bat`（默认 :8081） |
| `run_webui.bat` | Gradio 网页界面（按需起服务） | `run_webui.bat` |
| `_env.bat` | 共享环境探测（**不直接运行**） | 被其它脚本 `call` |

## Linux / macOS

本文档其余部分描述 Windows 的 `.bat` 脚本。在 Linux / macOS 上用仓库根目录的
`run_webui.sh`（`run_webui.bat` 的 POSIX 版本）：

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./run_webui.sh                      # UI -> http://127.0.0.1:7860
```

- Python 解释器依次探测 `$AUDIOCPP_PYTHON`、`venv/bin/python`、`.venv/bin/python`、
  `python3`、`python`。
- 后端（cuda/cpu）自动探测：Windows 看 `nvcuda.dll`，其它平台看 `nvidia-smi`，
  再确认对应的 server 构建存在；用 `AUDIOCPP_BACKEND=gpu|cpu` 覆盖。
- 二进制既支持 portable 包的 `gpu/`、`cpu/` 目录，也支持从源码构建的
  `build/<os>-<backend>-<type>/bin`（如 `build/linux-cuda-release/bin`）。
  直接 `cmake -B build` 产生的 `build/bin` 也能识别——目录名不含后端信息时，
  从 `CMakeCache.txt` 的 `GGML_CUDA` 判断。
- `requirements.txt` 里 Windows 专用的包（pywin32 及 SpeakType 用到的
  pythonnet/pywebview）带 `sys_platform` 标记，因此在 Linux 上也能直接安装。

## 界面语言 / UI language

界面支持 **English / 中文 / 中文繁體**，默认英文，右上角的语言下拉可随时切换。
选择会保存到 `webui/configs/ui_language.json`，下次启动沿用。

环境变量 `AUDIOCPP_LANG`（`en` | `zh` | `zh-Hant`，也接受 `zh_TW`、`zh-CN` 等写法）
只在**还没有保存过选择时**决定默认语言——否则它会盖掉用户在界面里的明确选择。
想改回环境变量控制，删掉 `webui/configs/ui_language.json` 即可。

> 繁體中文由 OpenCC 从简体字面转换（`opencc-python-reimplemented`），属于**字形**转换，
> 不做台湾/香港的用词替换（例如「软件」→「軟件」而非「軟體」）。

> 长文本合成不再需要单独脚本（原 `run_tts_long.bat` 已移除）：WebUI 的 TTS 标签页会自动
> 把长文本分段（VibeVoice 600 字/段，其它模型 1000 字/段），逐段合成后拼接成一个 wav。
> 命令行等价物是 `audiocpp_cli` 的 `--batch-text-file <txt> --batch-merge-audio concat`。
>
> 反过来，**VibeVoice 对过短文本（约 <40 个汉字）会整段胡言乱语**——模型特性，与音色/
> 参数/seed 无关，WebUI 会直接拦截并提示加长文本或改用其它模型；分段后的过短尾段也会
> 自动并回前一段。短句测试请用 `qwen3-tts` / `voxcpm2` / `pocket-tts`。

---

## 通用约定

- **模型用 catalog id 指定。** 两个合成脚本（cli / server）都用 `configs\models_catalog.json`
  里的 **id** 来指定模型，脚本会自动查出它的 `family` / `task` / 绝对路径，你不用再手写这些。
  当前已安装的 id：`qwen3-tts`、`qwen3-asr`、`vibevoice`、`omnivoice`、`pocket-tts`。
  未安装的 id 会提示 “not installed”，可在 WebUI 里下载，或用
  `python tools/model_manager.py install <download_id>` 安装（见 `models_catalog.json`）。
- **后端自动选择：** 检测到 CUDA（NVIDIA 驱动）就用 GPU，否则回退 CPU。
  想强制某个后端，设环境变量 `AUDIOCPP_BACKEND=gpu`（=cuda）或 `AUDIOCPP_BACKEND=cpu`。
  CLI、server、WebUI 都遵循这一检测（无 N 卡的机器自动落到 CPU 版，速度较慢、部分大模型不实用）。
- **路径基准：** 脚本内相对路径（如 `voice\demo_01_man.wav`、`output\xxx.wav`）都相对 `webui\` 目录。
- **可执行文件来源：** 自动定位整合包 `..\audiocpp-portable`（内含 `cpu\ gpu\ models\`），
  脚本被拷进整合包时也能自识别。

---

## `_env.bat`（内部共享，不要直接运行）

被其它脚本 `call`，负责一次性设置好公共变量（故意不用 `setlocal`，这样变量能带回调用方）：

- `BUNDLE` — 整合包根目录（含 `cpu\ gpu\ models\`）
- `HAS_CUDA` — 是否检测到 CUDA（`nvcuda.dll` 或 `nvidia-smi`）
- `BACKEND` / `CLI_EXE` — 选定的后端（`cuda`/`cpu`）与对应的 `audiocpp_cli.exe`
- `SERVER_EXE` — 按 `BACKEND` 选 `gpu\` 或 `cpu\` 的 `audiocpp_server.exe`（cpu 版缺失时回退 gpu 版）
- `PY` — 带依赖的 Python（供 `run_webui.bat` 用）

改动探测逻辑只需改这一个文件。

---

## 1. `run_cli_tts.bat` — 命令行单次 TTS

一次加载模型、合成一句、输出一个 wav。适合快速测试或脚本化单次生成。

```
用法: run_cli_tts.bat [model_id] ["合成文本"] [voice_ref] [ref_text]
```

| 位置参数 | 含义 | 默认 |
|---|---|---|
| 1 `model_id` | catalog 里的模型 id | `qwen3-tts` |
| 2 `"文本"` | 要合成的文本（含空格务必加引号） | 一句英文示例 |
| 3 `voice_ref` | 参考音色 wav（声音克隆用） | `voice\demo_01_man.wav` |
| 4 `ref_text` | 参考音频对应的文本 | 示例台词 |

- 输出固定到 `output\out_cli.wav`（可在脚本顶部改 `OUT`）。
- 语言、`max-tokens`、`seed` 等也在脚本顶部可改。
- 非声音克隆的模型可把 `VOICE_REF` 留空（脚本会自动不带 `--voice-ref`）。

**示例**

```bat
run_cli_tts.bat qwen3-tts "Hello, this is audio dot cpp."
run_cli_tts.bat qwen3-tts "换个音色" voice\demo_02_woman.wav "her reference line."
set AUDIOCPP_BACKEND=cpu & run_cli_tts.bat qwen3-tts "强制用 CPU 跑"
```

---

## 2. `run_server.bat` — HTTP API 服务

启动一个 OpenAI 兼容的 HTTP 服务，供**其它应用**调用。后端自动检测：有 CUDA 用 GPU，
否则用 CPU 版 server（CPU 下自动把 ggml 线程数设为核数-1；速度较慢，部分大模型不实用）。

```
用法: run_server.bat <model_id> [port] [device]
```

| 位置参数 | 含义 | 默认 |
|---|---|---|
| 1 `model_id` | catalog 里的模型 id | （必填） |
| 2 `port` | 监听端口 | `8080` |
| 3 `device` | GPU 设备号 | `0` |

- 脚本会用该 id 生成一份**单模型、绝对路径**的临时配置
  `%TEMP%\audiocpp_server_<port>.json`（按端口命名，两个实例互不冲突），再启动 server。
- **同时起两个服务**：在两个窗口分别运行不同 id + 不同端口，例如一个做 TTS、一个做 ASR：

  ```bat
  run_server.bat qwen3-tts 8080     :: 窗口 A：TTS
  run_server_asr.bat                :: 窗口 B：ASR（= run_server.bat qwen3-asr 8081）
  ```

  ⚠️ 两个模型要同时装进显存（8GB 下 0.6B + 0.6B 没问题；两个 1.7B 装不下）。
- **`run_server_asr.bat`**：`run_server.bat` 的 ASR 预设包装，双击即用。
  参数为 `[port] [device] [model_id]`，默认 `8081` / `0` / `qwen3-asr`。
- **局域网访问**：设 `AUDIOCPP_HOST=0.0.0.0` 让其它机器能连（**无鉴权**，仅在可信内网使用）。

### API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 就绪状态 + 已配置模型数 |
| GET | `/v1/models` | 列出该实例加载的模型 |
| POST | `/v1/audio/speech` | 文本转语音，默认返回 `audio/wav` |
| POST | `/v1/audio/transcriptions` | 语音转文本（ASR） |
| POST | `/v1/tasks/run` | 通用任务入口（字段同 CLI 请求格式） |

> **参考音色是每次请求带的**（server 端不预存音色）。声音克隆 TTS 每个请求要带
> `voice_ref` + `reference_text`。请求里的 `voice_ref` / `audio` 路径是**服务器本机路径**，
> 相对路径以 server 的工作目录（用本脚本启动时即 `webui\`）为基准，也可用绝对路径。

### 调用示例

TTS（用现成模板 `configs\req_speech.json`，其中含 `input`/`voice_ref`/`reference_text`）：

```bat
curl http://127.0.0.1:8080/v1/audio/speech -H "Content-Type: application/json" -o output\out_server.wav -d @configs\req_speech.json
```

ASR（音频用服务器本机路径）：

```bat
curl http://127.0.0.1:8081/v1/audio/transcriptions -H "Content-Type: application/json" -d "{\"model\":\"qwen3-asr\",\"audio\":\"D:/path/to/input.wav\"}"
```

查看状态：

```bat
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

---

## 3. `run_webui.bat` — 图形界面

启动 Gradio 网页界面（`webui.py`），浏览器访问 **http://127.0.0.1:7860**。

- **按需加载**：不需要先跑 `run_server.bat`——在界面里选模型点“加载”/“生成”时，WebUI 会自动
  起/切换底层的 `audiocpp_server`（一次一个模型在显存里，换模型即重启）。
- 界面里可上传参考音色、下载未安装的模型、填 HF token / 代理等。
- 后端自动检测（同上：有 CUDA 用 GPU，否则 CPU）；`AUDIOCPP_BACKEND=gpu|cpu` 可强制。
  CPU 模式下 ggml 线程数自动设为核数-1（可用 `AUDIOCPP_THREADS=N` 覆盖），且不再显示显存警告。

> 网页界面（7860）是给人用的；要给**其它程序**当 API，请用 `run_server.bat` 起的 **8080** 那个服务，
> 或让 WebUI 起来后直接打它的 8080 端口（见 `run_server.bat` 的端点表）。

---

## WebUI 高级参数（按模型自动生成控件）

TTS 标签页「合成设置 → 高级参数」里的控件由 **`configs/model_params.json`** 驱动：选中某个模型后，WebUI 按其 `family` **动态生成对应的滑块/数字框/开关/文本框**（`gr.render`），不用再手写 JSON。控件下方还留了一个可折叠的「其它参数（JSON）」兜底框，用于传配置里没列出的键。通用规则：

- **只有你改动过的控件值才会随请求发送**（未动的用模型自身默认值）；`options` 合并顺序：家族默认 → 生成的控件 → JSON 框（JSON 覆盖控件）。
- `seed`、`max_tokens` 已有专用输入框（合成设置），不在此重复。
- 参考音色用「上传/录制」或「内置参考音色」；参考音频对应的原话用「参考文本」框（等价 `reference_text`）。
- 填错的值通常被忽略，或由 server 报错——错误显示在**输出音频下方**（不弹卡片）。
- **Chatterbox** 的克隆参数在模型加载时固定：改动后需重新点『📥 加载模型』才生效（否则会报 “session config is fixed”）。

### 自定义控件（编辑 `configs/model_params.json`）

按 `family` 分组，每项一个控件规格：

```json
{"name": "guidance_scale", "type": "slider", "label": "guidance_scale",
 "default": 1.3, "minimum": 0.0, "maximum": 5.0, "step": 0.1, "info": "CFG 引导强度"}
```

- `name`：透传给请求 `options` 的键名。`type`：`slider` / `number`（`precision:0` 表整数）/ `bool` / `text` / `choice`（配 `choices:[...]`）。
- `default` 应等于模型默认值（已按各 `src/models/<family>/*.cpp` 校对）。
- 改完点界面上的『🔄 刷新列表』即可重新加载本文件，无需重启。
- 文件路径 / parity 类少见参数（如 `*_noise_file`）未纳入控件，可用「其它参数（JSON）」框传。量化键（如 `vibevoice.weight_type`）见项目根 `README.md`。

下表是每个模型 `session.cpp` **实际读取**的完整可用键（控件是其中精选的常用子集；其余键仍可用 JSON 框传）：

| 模型（family） | 可用键（JSON 框也可传） | 示例 |
|---|---|---|
| **Qwen3-TTS**（qwen3_tts）0.6B / 1.7B / CustomVoice | `do_sample` `temperature` `top_k` `top_p`；CustomVoice 版另有 `speaker` | `{"do_sample": true, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`<br>CustomVoice 选内置音色：`{"speaker": "<CustomVoice 音色名>"}` |
| **VibeVoice**（vibevoice）1.5B 长文/多说话人 | `num_inference_steps` `guidance_scale` `max_length_times` `do_sample` `temperature` `top_k` `top_p`；多说话人 `voice_samples`（逗号分隔 wav，最多 4，**不能**与参考音色同用） | `{"num_inference_steps": 10, "guidance_scale": 1.3, "max_length_times": 2.0}`<br>多说话人：`{"voice_samples": "D:/a.wav,D:/b.wav"}` |
| **VoxCPM2**（voxcpm2） | `num_inference_steps` `guidance_scale` `min_tokens` `retry_badcase` `retry_badcase_max_times` `retry_badcase_ratio_threshold`；参考原话 `prompt_text` | `{"num_inference_steps": 10, "guidance_scale": 2.0, "retry_badcase": true}` |
| **MioTTS**（miotts，需 MioCodec） | `temperature` `top_k` `top_p` `repetition_penalty` `presence_penalty` `frequency_penalty` `do_sample` `best_of_n` `best_of_n_enabled` `best_of_n_language` | `{"temperature": 0.9, "top_p": 0.9, "repetition_penalty": 1.1, "best_of_n": 3}` |
| **Chatterbox**（chatterbox，声音克隆） | `exaggeration` `guidance_scale` `temperature` `repetition_penalty` `min_p` `top_p` `s3gen_cfg_rate` `max_new_tokens` `do_sample` `greedy` `stop_on_eos` | `{"exaggeration": 0.5, "guidance_scale": 0.5, "temperature": 0.8, "repetition_penalty": 1.2}` |
| **OmniVoice**（omnivoice） | `instruct`（风格/指令文本）；`reference_text`（一般用「参考文本」框即可） | `{"instruct": "以轻快的语气朗读"}` |
| **Pocket TTS**（pocket_tts） | 无专用高级参数（只需参考音色 + 语言） | — |

> 键名取自各模型 `src/models/<family>/session.cpp` 实际读取的选项；同一键在不同模型里的取值范围/含义可能不同。量化相关键（如 `vibevoice.weight_type`、`voxcpm2.*_weight_type`）见项目根 `README.md` 的量化章节，不是通用默认项。

### 音乐生成 / 声音转换参数详解

页面上的提示已精简，完整说明集中在这里。

**ACE-Step（音乐生成/编辑）**

- 提示词写风格/乐器/情绪（英文效果最好），可选填歌词；时长填 `-1` 表示自动。
- `task_route` 操作类型：`text2music`＝纯文生曲（默认，不需要源音频）；`cover`＝换词翻唱
  （原版 Remix 主路线，配合下面两个 cover 滑条）；`cover-nofsq`＝cover 变体（不过 FSQ 量化）；
  `remix`＝flow-edit 精细换词；`complete` / `lego` / `extract` / `repaint` 为其它编辑路线。
  **除 text2music 外都需要上传源音频。**
- 上传源音频后建议先点『🔍 分析源音频』：反推源曲描述/歌词/BPM/调性并自动填入高级参数
  （remix/cover 换词前尤其建议；首次需先『📥 加载模型』，1 分钟音频约需几十秒）。
  分析结果可复现：同一音频每次分析一致（VAE 均值编码；seed=-1 时分析固定用 1234，
  想重抽歌词转写可换一个具体 seed）。
- 扩散参数：`num_inference_steps` turbo 上限 20，remix 路由不填时默认 16、其他路由默认 8；
  `shift`（时间步弯曲）默认 3.0 对齐原版 turbo UI——调回 1.0 会明显劣化 remix 换词咬字。
- cover 路线两个滑条：
  - `audio_cover_strength`（Remix 强度）：多少比例的去噪步参考源曲结构，1=贴近原曲、
    0=自由发挥；原版 Remix 建议 0.5。仅 cover/cover-nofsq 生效。
  - `cover_noise_strength`（旋律保持）：从源曲部分加噪的起点开始去噪，0=不保旋律、
    0.1~0.25=推荐区间（保旋律又能换词换风格）、越高越贴原曲。仅 cover 生效。
- remix（flow-edit）参数：
  - `source_caption` / `source_lyrics`：源侧文本条件（源歌曲本来的风格描述 / 原歌词，
    带 `[Verse]` `[Chorus]` 标签）；留空 caption 用主提示词；『🔍 分析』可自动填。
    **新歌词写在主界面『歌词』框。**
  - `flow_edit_n_min`（编辑起点）：跳过前面高噪声步的比例，0=从头编辑；调大更保源曲但换词更弱。
  - `flow_edit_n_max`（编辑终点）：1=全程配对编辑；调低到 0.7~0.9 时收尾只朝新歌词去噪——
    **歌词唱不出来时优先调这个**。
  - `flow_edit_n_avg`：每步多次采样取平均（remix 默认 2，更稳），1=最快。
    注意 remix 默认 16 步 × n_avg 2 ≈ 旧默认（8 步 ×1）4 倍耗时，求快可手动调回。
- 曲谱参数 `bpm` / `keyscale`（如 `F major`、`c# minor`）/ `timesignature`（如 `4`）：
  0/留空=不指定；『🔍 分析』后自动填。

**Stable Audio（音乐/音效）**：提示词**仅支持英文**，不使用歌词；music 版生成音乐、sfx 版生成音效。
上传源音频可做续写/修补：`audio_input_kind` 选 `init_audio`（配 `init_noise_level` 强度）或
`inpaint_audio`。

**HeartMuLa（歌词+标签生成歌曲）**：高级参数 `tags` 必填（逗号分隔，如
`pop,bright,drums,female vocals`），『歌词』填唱词。3B 模型，官方 120 秒长歌实测峰值显存
~25G（docs/memory_saver.md），8G 显卡跑不动；已默认开 mem_saver，长歌曲可开 `infinite_mode`。

**Chatterbox VC（语音转换）**：源语音提供内容，目标音色参考提供说话人身份，输出 24kHz
单声道语音。`s3gen_cfg_rate` 控制音色引导强度，`num_inference_steps` 控制生成步数；默认分别为
0.7 和 10。该入口与 TTS 页的 Chatterbox 声音克隆共用同一套模型文件。

**Seed-VC（语音转换）**：源语音 + 目标音色参考（几秒到几十秒干净人声）。`route` 留空按任务默认
（vc 条目→`v2_vc`，svc 条目→`v1_svc`）；`v1_whisper_bigvgan_vc` / `v1_xlsr_hift_vc` 为 v1 旧路线；
`v1_svc` 只能配 svc 条目。`intelligibility_cfg_rate` / `similarity_cfg_rate` 仅 v2 生效，
`inference_cfg_rate` 仅 v1 生效。

**Vevo2（语音转换）**：默认 `route=style_preserved_vc`（保留源语音的说话风格，只换音色）。
`route` 留空按条目任务默认（vc→style_preserved_vc，svc→style_preserved_svc，s2s→editing），
且须与所选条目任务匹配；`style_converted_*` / `editing` 需在「其它参数(JSON)」里补
`style_ref`（服务器本地 wav 路径）/ `style_ref_text` / `target_text`。
`use_pitch_shift`（按源/目标中位音高差整体移调）留空按路线默认：style_preserved_* 及
singing 路线默认开，style_converted_vc / editing 默认关。
长音频按『目标音色时长 + 每段源时长 ≤ 显存预算』自适应分段后拼接，参考音色超过约 10s
自动截短（8G 显存限制）。

### 各任务页输入要求详解

- **VibeVoice**：多说话人脚本每行 `Speaker N: 内容`（N 从 0 起），只填普通文字会自动包成
  `Speaker 0: ...`。多角色不同音色用高级参数 `voice_samples`（逗号分隔服务器本地 wav，≤4 个），
  此时**不要**再上传参考音色。
- **VoxCPM2 / Qwen3-TTS**：上传/选一段干净的单人参考音色并在『参考文本』填该音频的原话，
  否则可能提前截断。长文本自动分段合成后拼接；VoxCPM2 在 8G 显卡默认 q8_0 量化。
- **Chatterbox**：语言只支持 english / spanish / french / german / italian / portuguese / korean
  （无中文/日文/俄文，也没有自动检测）；『留空』=英语。
- **Qwen3-ASR**：长音频自动在静音处按 ≤60 秒分段转写后拼接。『上下文提示』填人名/术语/背景
  （如：会议讨论 ggml 量化，参会人：张伟、李娜）帮助认出专有名词。对话模式（限 120s）先用
  Sortformer 说话人分离（≤4 人）再逐段转写成带说话人和时间戳的对话稿，需已安装 Sortformer 模型。
- **音频分析（VAD/分离/对齐）**：WAV 输入自动转 16 kHz 单声道后送模型，结果时间轴按 16 kHz 换算。
  Qwen3 强制对齐单次音频上限约 115 秒。
- **音源分离**：HTDemucs 输出 drums/bass/other/vocals 四轨（长音频耗时较长）；
  Mel-Band RoFormer 输出人声轨 + 伴奏轨（mixture − vocals）。
- **IndexTTS2**（0.3 新增）：中/英声音克隆，**必须**提供参考音色。情感控制在高级参数：
  `emotion_text` 填情感描述（填了会自动开启 `use_emotion_text`）+ `emotion_alpha` 调强度；
  或勾 `use_emotion_text` 从朗读文本自动推断；`emotion_vector`（8 个浮点）走 JSON 兜底框。
- **Irodori-TTS**（0.3 新增，日语）：500M 默认无参考直接生成，上传参考音色自动切克隆
  （界面替你发 `no_ref=false`）；600M VoiceDesign 在『声音设计』页用日语 caption 描述音色。
  语言下拉只认 japanese/留空。
- **MOSS-TTS**（0.3 新增）：Local v1.5 纯文本可生成，克隆时建议配『参考文本』，输出 48kHz
  立体声；Nano 100M 轻量，无参考=续写式生成（音色随机），有参考=克隆。
- **Supertonic 3**（0.3 新增）：预置音色多语种 TTS（英/日/韩/欧洲语种，**无中文**），
  高级参数选 `voice`（M1-M5 男 / F1-F5 女）和 `speaking_rate`；不支持参考音频克隆。
- **模型下载**在后台进行，进度自动刷新，也可点「📊 下载进度」手动查看。

### GGUF 转换、检查与加载

每个任务页的「模型管理」卡片都提供同一组 GGUF 操作：选择类型（默认 `q8_0`）后点「🧊 转换 GGUF」，
会把结果写为所选模型目录下的 `model.gguf`；已有文件不会被覆盖。点「🔎 检查 GGUF」会在页面上执行
`audiocpp_gguf.exe --inspect` 并显示包的元数据。对已接入原生 GGUF 的模型，目录存在 `model.gguf` 时，普通
「📥 加载模型」会自动优先使用 GGUF；点「🗑️ 删除 GGUF」会删除该文件（以及同名残留 `.tmp`），下次普通加载
即恢复原始权重。

- 转换器按顺序查找开发构建的 `build\windows-cuda-release\bin` / `build\windows-cpu-release\bin`，以及整合包的
  `audiocpp-portable\gpu` / `audiocpp-portable\cpu`；也可用 `AUDIOCPP_GGUF` 指向自定义 `audiocpp_gguf.exe`。
- 页面只会把已接入原生 GGUF 模型规格、且能明确整理转换输入的模型标为「可转换」；存在 `.safetensors` 不代表对应
  C++ 后端已支持 GGUF。支持转换但尚未完整安装的模型会提前显示「可转换，但模型未完整安装」，便于下载前判断；
  Stable Audio 当前仍使用原始权重，不会标为可转换。
- 页面自动处理受支持的单个 `model.safetensors`、分片索引和 Qwen3-TTS 复合权重。其他需要多个命名
  `--input namespace=...` 的复合模型仍应使用命令行，避免 UI 猜错权重命名空间。
- 仅 audio.cpp-native GGUF 可加载；量化兼容性因模型和推理路线而异。转换成功也应先用短样本检查输出质量。

---

## 模型 id 速查

完整清单见 `configs\models_catalog.json`（每条含 `id` / `family` / `path` / `task` / `download_id`）。
常用：

| id | 家族 | 任务 | 说明 |
|---|---|---|---|
| `qwen3-tts` | qwen3_tts | tts | Qwen3-TTS 0.6B（声音克隆） |
| `qwen3-asr` | qwen3_asr | asr | Qwen3-ASR 0.6B |
| `vibevoice` | vibevoice | tts | VibeVoice 1.5B（长文/多说话人，`Speaker N:` 脚本） |
| `omnivoice` | omnivoice | tts | OmniVoice |
| `pocket-tts` | pocket_tts | tts | Pocket TTS（需参考音色） |
| `index-tts2` | index_tts2 | tts | IndexTTS2（中英克隆+情感，需参考音色） |
| `irodori-tts` | irodori_tts | tts | Irodori-TTS 500M（日语） |
| `irodori-tts-vdesign` | irodori_tts | vdes | Irodori-TTS 600M VoiceDesign（日语 caption） |
| `moss-tts-local` | moss_tts_local | tts | MOSS-TTS-Local v1.5（48kHz 立体声） |
| `moss-tts-nano` | moss_tts_nano | tts | MOSS-TTS-Nano 100M（轻量） |
| `supertonic` | supertonic | tts | Supertonic 3（预置音色，无中文） |

未安装的 id 运行时会提示，可在 WebUI 里点“下载”，或
`python tools\model_manager.py install <download_id> --models-root <bundle>\models`。

---

## 环境变量

| 变量 | 作用 | 适用 |
|---|---|---|
| `AUDIOCPP_BACKEND` | `gpu`(=cuda) / `cpu` 强制后端 | cli / server / webui |
| `AUDIOCPP_HOST` | server 绑定地址（`0.0.0.0` 开放局域网） | server |
| `AUDIOCPP_BUNDLE` | 手动指定整合包根目录 | 全部 |
| `AUDIOCPP_SERVER` | 让 WebUI 连一个已在跑的外部 server | webui |
| `AUDIOCPP_LOAD_TIMEOUT` | WebUI 等待模型加载的秒数（默认 300） | webui |

---

## 常见问题

- **`.bat` 双击闪退 / 命令语法错误**：这些脚本必须是 **CRLF** 行尾（LF 会让 cmd 解析出错），
  编辑后请保持 CRLF。
- **端口被占用**：`run_server.bat` 和 WebUI 默认都用 8080。要同时用，就给 server 换端口，
  或设 `AUDIOCPP_SERVER` 让 WebUI 复用外部 server。
- **`model path does not exist` / not installed**：模型没装。用上面的 model_manager 命令或 WebUI 下载。
- **显存不足**：8GB 下同时跑两个 server 时，两个模型都要装得下；1.7B 建议单开。
- **声音克隆生成过短（~0.4s 就结束）**：`voice_ref` 音色不干净或缺 `reference_text`；换单一说话人的
  干净参考音频并配上对应文本。

---

## API 方式 vs 命令行的性能

同一套引擎、同一后端 → **推理本身完全一样**。差别主要在**模型加载的摊销**：

- `run_cli_tts.bat` **每次调用都要把模型重新装进显存**（每次固定几秒开销）。
- `run_server.bat` 的服务**只加载一次、常驻**，之后每个请求只花“推理 + 极小的传输”。
  本机 HTTP + 几 MB 的 wav 传输 ≈ 毫秒级，相对多秒的推理可忽略（建议用默认二进制 wav，
  别用 `response_format:"json"` 的 base64，会大约 +33%）。
- 网页界面（7860）比直连 8080 多一跳代理；其它程序直接打 8080 就没有这一跳。

**结论**：走 API 每次生成几乎没有额外成本，只有一次性的预热被服务端摊掉了——除了“只生成一次”的
场景，API 方式通常比反复调 CLI **更快**。
