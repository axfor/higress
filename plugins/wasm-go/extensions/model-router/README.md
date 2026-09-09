## 功能说明
`model-router`插件实现了基于LLM协议中的model参数路由的功能

## 配置字段

| 名称                 | 数据类型        | 填写要求                | 默认值                   | 描述                                                  |
| -----------          | --------------- | ----------------------- | ------                   | -------------------------------------------           |
| `modelKey`           | string          | 选填                    | model                    | 请求body中model参数的位置                             |
| `addProviderHeader`  | string          | 选填                    | -                        | 从model参数中解析出的provider名字放到哪个请求header中 |
| `modelToHeader`      | string          | 选填                    | -                        | 直接将model参数放到哪个请求header中                   |
| `enableOnPathSuffix` | array of string | 选填                    | ["/completions","/embeddings","/images/generations","/audio/speech","/fine_tuning/jobs","/moderations","/image-synthesis","/video-synthesis","/rerank","/messages"] | 只对这些特定路径后缀的请求生效，可以配置为 "*" 以匹配所有路径 |
| `keepOriginalModelName` | bool         | 选填                    | false                    | 配合 `addProviderHeader` 使用，设为 true 时仍提取 provider 写入 header，但不改写请求体中的 model 字段 |
| `autoRouting`        | object          | 选填                    | -                        | 自动路由配置，详见下方说明                            |
| `streamEarlyCommit`  | bool            | 选填                    | false                    | 请求体流式处理时，一旦读到 `model` 就开始向上游转发，不再等满 64KB 窗口；每个在途请求持有的字节更少，但放弃"窗口内的请求体与全量路径行为完全一致"的保证（窗口内出现的 JSON 语法错误不再回落）。详见下方"请求体流式处理" |

### autoRouting 配置

| 名称           | 数据类型        | 填写要求 | 默认值 | 描述                                                         |
| -------------- | --------------- | -------- | ------ | ------------------------------------------------------------ |
| `enable`       | bool            | 必填     | false  | 是否启用自动路由功能                                         |
| `defaultModel` | string          | 选填     | -      | 当没有规则匹配时使用的默认模型                               |
| `rules`        | array of object | 选填     | -      | 路由规则数组，按顺序匹配                                     |

### rules 配置

| 名称      | 数据类型 | 填写要求 | 描述                                                         |
| --------- | -------- | -------- | ------------------------------------------------------------ |
| `pattern` | string   | 必填     | 正则表达式，用于匹配用户消息内容                             |
| `model`   | string   | 必填     | 匹配成功时设置的模型名称，将设置到 `x-higress-llm-model` 请求头 |

## 请求体流式处理

JSON 请求体且 `modelKey` 是顶层普通字段时，插件不再缓冲整份请求体：只在请求体开头（64KB 窗口）内寻找 `model`，
找到后立即设置路由请求头、原位改写 `model` 字段（与 sjson 改写逐字节一致），之后的字节原样转发、不再扫描。
单个请求占用的内存与请求体大小无关。

以下情形自动走原来的全量缓冲路径，结果与之前完全一致：`multipart/form-data`、`modelKey` 是 gjson 路径（含 `.` 等）、
自动路由（需要读最后一条 user 消息）、`model` 不是字符串、窗口内没有出现 `model`（如 SDK 把很长的 `messages` 放在 `model` 之前）、
窗口内出现 JSON 语法错误。

`streamEarlyCommit: true` 时不等窗口读满：`model` 一到就设置路由头并开始转发，64KB 只做上限。转换器改写完 `model`
之后，要等引擎把已读入的字节全部写出（例如正在读的下一个 key、等着逗号的空白）才会被丢掉，之后的字节原样转发；
这一点在窗口模式下同样成立。

指标：`model_router.stream.streamed` / `fallback`（Envoy 统计前缀 `wasmcustom.`）。

## 运行属性

插件执行阶段：认证阶段
插件执行优先级：900

## 效果说明

### 基于 model 参数进行路由

需要做如下配置：

```yaml
modelToHeader: x-higress-llm-model
```

插件会将请求中 model 参数提取出来，设置到 x-higress-llm-model 这个请求 header 中，用于后续路由，举例来说，原生的 LLM 请求体是：

```json
{
    "model": "qwen-long",
    "frequency_penalty": 0,
    "max_tokens": 800,
    "stream": false,
    "messages": [{
        "role": "user",
        "content": "higress项目主仓库的github地址是什么"
    }],
    "presence_penalty": 0,
    "temperature": 0.7,
    "top_p": 0.95
}
```

经过这个插件后，将添加下面这个请求头(可以用于路由匹配)：

x-higress-llm-model: qwen-long

### 提取 model 参数中的 provider 字段用于路由

> 注意这种模式需要客户端在 model 参数中通过`/`分隔的方式，来指定 provider

需要做如下配置：

```yaml
addProviderHeader: x-higress-llm-provider
```

插件会将请求中 model 参数的 provider 部分（如果有）提取出来，设置到 x-higress-llm-provider 这个请求 header 中，用于后续路由，并将 model 参数重写为模型名称部分。举例来说，原生的 LLM 请求体是：

```json
{
    "model": "dashscope/qwen-long",
    "frequency_penalty": 0,
    "max_tokens": 800,
    "stream": false,
    "messages": [{
        "role": "user",
        "content": "higress项目主仓库的github地址是什么"
    }],
    "presence_penalty": 0,
    "temperature": 0.7,
    "top_p": 0.95
}
```

经过这个插件后，将添加下面这个请求头(可以用于路由匹配)：

x-higress-llm-provider: dashscope

原始的 LLM 请求体将被改成：

```json
{
    "model": "qwen-long",
    "frequency_penalty": 0,
    "max_tokens": 800,
    "stream": false,
    "messages": [{
        "role": "user",
        "content": "higress项目主仓库的github地址是什么"
    }],
    "presence_penalty": 0,
    "temperature": 0.7,
    "top_p": 0.95
}
```

### 保留原始模型名（keepOriginalModelName）

当使用 AI 模型聚合平台（如百炼/DashScope）接入第三方厂商模型时，部分模型名称本身包含 `/`（如 `MiniMax/MiniMax-M2.7`），并非 `provider/model` 格式。此时配合 `addProviderHeader` 使用会导致请求体中的 model 字段被错误改写。

通过设置 `keepOriginalModelName: true`，可以在保留 provider header 提取能力的同时，不改写请求体中的 model 字段：

```yaml
addProviderHeader: x-higress-llm-provider
keepOriginalModelName: true
```

以 model 为 `MiniMax/MiniMax-M2.7` 为例，经过插件后：
- 请求头 `x-higress-llm-provider` 设置为 `MiniMax`
- 请求体中的 model 字段保持为 `MiniMax/MiniMax-M2.7`（不改写）

### 自动路由模式（基于用户消息内容）

当请求中的 model 参数设置为 `higress/auto` 时，插件会自动分析用户消息内容，并根据配置的正则规则选择合适的模型进行路由。

配置示例：

```yaml
autoRouting:
  enable: true
  defaultModel: "qwen-turbo"
  rules:
    - pattern: "(?i)(画|绘|生成图|图片|image|draw|paint)"
      model: "qwen-vl-max"
    - pattern: "(?i)(代码|编程|code|program|function|debug)"
      model: "qwen-coder"
    - pattern: "(?i)(翻译|translate|translation)"
      model: "qwen-turbo"
    - pattern: "(?i)(数学|计算|math|calculate)"
      model: "qwen-math"
```

#### 工作原理

1. 当检测到请求体中的 model 参数值为 `higress/auto` 时，触发自动路由逻辑
2. 从请求体的 `messages` 数组中提取最后一个 `role` 为 `user` 的消息内容
3. 按配置的规则顺序，依次使用正则表达式匹配用户消息
4. 匹配成功时，将对应的 model 值设置到 `x-higress-llm-model` 请求头
5. 如果所有规则都未匹配，则使用 `defaultModel` 配置的默认模型
6. 如果未配置 `defaultModel` 且无规则匹配，则不设置路由头（会记录警告日志）

#### 使用示例

客户端请求：

```json
{
    "model": "higress/auto",
    "messages": [
        {
            "role": "system",
            "content": "你是一个有帮助的助手"
        },
        {
            "role": "user",
            "content": "请帮我画一只可爱的小猫"
        }
    ]
}
```

由于用户消息中包含"画"关键词，匹配到第一条规则，插件会设置请求头：

```
x-higress-llm-model: qwen-vl-max
```

#### 支持的消息格式

自动路由支持两种常见的 content 格式：

1. **字符串格式**（标准文本消息）：
```json
{
    "role": "user",
    "content": "用户消息内容"
}
```

2. **数组格式**（多模态消息，如包含图片）：
```json
{
    "role": "user",
    "content": [
        {"type": "text", "text": "用户消息内容"},
        {"type": "image_url", "image_url": {"url": "..."}}
    ]
}
```

对于数组格式，插件会提取最后一个 `type` 为 `text` 的内容进行匹配。

#### 正则表达式说明

- 规则按配置顺序依次匹配，第一个匹配成功的规则生效
- 支持标准 Go 正则语法
- 推荐使用 `(?i)` 标志实现大小写不敏感匹配
- 使用 `|` 可以匹配多个关键词
