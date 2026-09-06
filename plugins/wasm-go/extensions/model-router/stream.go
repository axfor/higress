package main

import (
	"encoding/json"
	"strings"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform/guard"
)

// 请求体流式路径。
//
// 官方实现把整份 body 缓冲（上限 100MB）后用 gjson 取 model、sjson 改写；这里只在 body 开头
// （streamxform.CommitBytes 的窗口内）找 model：找到就设请求头、原位改写，之后的字节原样直通、不再扫描。
// 以下情形保持官方全量路径，结果与官方逐字节一致：
//   - 非 JSON（multipart）、modelKey 不是顶层普通 key、没有 content-type；
//   - auto 路由命中（要看最后一条 user 消息）、model 不是字符串；
//   - 窗口内没见到 model（SDK 把 messages 放前面且超过窗口）。
// 已知差异：body 后半段有 JSON 语法错误时，官方 json.Valid 失败会整体不动，流式已按开头改写并原样送出剩余字节。

const ctxKeyStream = "model_router_stream"

var streamMetric = guard.NewMetric("model_router.stream")

// streamable 报告这次请求能否走流式路径。
func streamable(config ModelRouterConfig, contentType string) bool {
	if !strings.Contains(contentType, "application/json") || config.modelKey == "" {
		return false
	}
	for i := 0; i < len(config.modelKey); i++ {
		c := config.modelKey[i]
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			return false // gjson 路径语法（. # * ? @ | \）：交给官方路径
		}
	}
	return true
}

type jsonRoute struct {
	config   ModelRouterConfig
	model    string
	modelOK  bool // 见到了字符串 model
	needFull bool // 必须整份 body：auto 路由 / 非字符串 model
}

func newStream(ctx wrapper.HttpContext, config ModelRouterConfig) *guard.State {
	r := &jsonRoute{config: config}
	tr := streamxform.NewKeyProbe(streamxform.KeyProbeOptions{
		Keys:     map[string]int{config.modelKey: 4096},
		ModelKey: config.modelKey,
		OnKey:    r.onModel,
	})
	return guard.New(&guard.Plan{
		Tr:       tr,
		Mode:     guard.PrefixTransform,
		OnCommit: r.commit,
		Fallback: func(body []byte) types.Action { return onHttpRequestBody(ctx, config, body) },
		Metric:   streamMetric,
		Log:      log.Warnf,
	})
}

// onModel：model 值到齐。决定是否原位改写（provider/model 去前缀）。
func (r *jsonRoute) onModel(t *streamxform.Transformer, key string, raw []byte) ([]byte, bool) {
	var s string
	if json.Unmarshal(raw, &s) != nil {
		r.needFull = true // 官方对非字符串走 gjson 的 String() 语义，交给全量路径复刻
		return nil, false
	}
	r.model, r.modelOK = s, true
	if s == "" {
		return nil, false
	}
	if r.config.enableAutoRouting && s == AutoModelPrefix {
		r.needFull = true
		return nil, false
	}
	if r.config.addProviderHeader != "" && !r.config.keepOriginalModelName {
		if i := strings.Index(s, "/"); i >= 0 {
			return sjsonString(s[i+1:]), true
		}
	}
	return nil, false
}

// commit：放行请求头之前。与官方 handleJsonBody 设头的顺序一致。
func (r *jsonRoute) commit(pre streamxform.Prelude, last bool) bool {
	if r.needFull {
		return false
	}
	if !r.modelOK {
		return last // 整份都到了还没有 model：官方不动；否则 model 可能在窗口之外，回落
	}
	if r.model == "" {
		return true
	}
	if r.config.modelToHeader != "" {
		_ = proxywasm.ReplaceHttpRequestHeader(r.config.modelToHeader, r.model)
	}
	if r.config.addProviderHeader != "" {
		parts := strings.SplitN(r.model, "/", 2)
		if len(parts) == 2 {
			_ = proxywasm.ReplaceHttpRequestHeader(r.config.addProviderHeader, parts[0])
			log.Debugf("model route to provider: %s, model: %s", parts[0], parts[1])
		} else {
			log.Debugf("model route to provider not work, model: %s", r.model)
		}
	}
	return true
}

// sjsonString 复刻 sjson 的字符串编码：纯可见 ASCII 直接加引号，否则走 encoding/json。
func sjsonString(s string) []byte {
	for i := 0; i < len(s); i++ {
		if s[i] < ' ' || s[i] > 0x7f || s[i] == '"' || s[i] == '\\' {
			b, _ := json.Marshal(s)
			return b
		}
	}
	return []byte(`"` + s + `"`)
}

func onHttpStreamingRequestBody(ctx wrapper.HttpContext, config ModelRouterConfig, chunk []byte, last bool) ([]byte, types.Action) {
	st, _ := ctx.GetContext(ctxKeyStream).(*guard.State)
	if st == nil {
		st = newStream(ctx, config)
		ctx.SetContext(ctxKeyStream, st)
	}
	return st.Feed(chunk, last)
}
