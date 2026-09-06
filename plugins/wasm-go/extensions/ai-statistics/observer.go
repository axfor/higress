package main

import (
	"encoding/json"
	"strings"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/tokenusage"
	"github.com/higress-group/wasm-go/pkg/wrapper"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform/guard"
)

// 请求体流式观察。
//
// 官方实现总是把整份请求体缓冲下来，只为取顶层 model 和数 user 轮数（messages[].role == "user"，
// Gemini 则数 contents[] 里 role 缺失或为 user 的元素）。轻量模式（属性里没有 request_body 来源）下
// 这些都能边扫边算：请求体原样逐块转发，一个字节不缓冲；只有配置了从请求体提取属性
// （默认属性集的 messages / question / system，或自定义 request_body 属性）时才保留官方缓冲路径。
//
// 观察语义对齐 gjson：顶层同名 key 取第一个；messages 是数组就用它（哪怕为空），否则才看 contents；
// 元素不是对象、role 不是字符串都不计数。请求体不是合法 JSON 时停止观察、继续原样转发。

const ctxKeyObserver = "ai_statistics_observer"

var observeMetric = guard.NewMetric("ai_statistics.stream")

// requestObserver 是只读协议：Capture 小字段、Skip 其余，输出被丢弃。
type requestObserver struct {
	streamxform.BaseProtocol
	seen         map[string]bool
	model        string
	modelSeen    bool
	msgsIsArray  bool
	contsIsArray bool
	userMsgs     int
	userConts    int
	elemRole     string
	elemRoleSeen bool
	inKey        string // 当前在哪个顶层数组里（messages / contents）
}

func newRequestObserver() *streamxform.Transformer {
	p := &requestObserver{seen: map[string]bool{}}
	tr := streamxform.NewTransformer(p)
	tr.DupKeyBail = false // gjson 取第一个：重复 key 原样跳过即可
	return tr
}

func (p *requestObserver) OnKey(t *streamxform.Transformer) streamxform.Action {
	switch t.Depth() {
	case 1:
		k := t.Last()
		if p.seen[k] {
			return streamxform.Skip()
		}
		switch k {
		case "model":
			p.seen[k] = true
			return streamxform.Capture(4096)
		case "messages", "contents":
			p.seen[k] = true
			return streamxform.Probe()
		}
	case 3:
		if t.Last() == "role" && !p.elemRoleSeen {
			return streamxform.Probe()
		}
	}
	return streamxform.Skip()
}

func (p *requestObserver) OnElem(t *streamxform.Transformer) streamxform.Action {
	if t.Depth() == 2 {
		return streamxform.Probe()
	}
	return streamxform.Skip()
}

func (p *requestObserver) OnStart(t *streamxform.Transformer, kind streamxform.ValueKind) streamxform.Action {
	switch t.Depth() {
	case 1: // messages / contents
		if kind != streamxform.KindArray {
			return streamxform.Skip() // 官方 IsArray() 为假：不用它
		}
		p.inKey = t.Last()
		if p.inKey == "messages" {
			p.msgsIsArray = true
		} else {
			p.contsIsArray = true
		}
		return streamxform.Enter().Lazy()
	case 2: // 元素
		if kind != streamxform.KindObject {
			if p.inKey == "contents" {
				p.userConts++ // gjson：非对象元素没有 role → Gemini 规则算作 user
			}
			return streamxform.Skip() // messages：非对象元素的 role 为空，不计数
		}
		p.elemRole, p.elemRoleSeen = "", false
		return streamxform.Enter().Lazy()
	case 3: // role
		if kind != streamxform.KindString {
			p.elemRoleSeen = true // 非字符串：官方 String() 不等于 "user"；Gemini 视为"有 role 但不是 user"
			p.elemRole = "\x00"
			return streamxform.Skip()
		}
		return streamxform.Capture(256)
	}
	return streamxform.Skip()
}

func (p *requestObserver) OnValue(t *streamxform.Transformer, raw []byte) {
	switch t.Depth() {
	case 1: // model
		p.modelSeen = true
		var s string
		if json.Unmarshal(raw, &s) == nil {
			p.model = s
		} else if string(raw) == "null" {
			p.model = ""
		} else {
			p.model = string(raw) // gjson 的 String()：数字 / 布尔 / 容器给原文
		}
	case 3: // role
		var s string
		if json.Unmarshal(raw, &s) == nil {
			p.elemRole = s
		} else {
			p.elemRole = "\x00"
		}
		p.elemRoleSeen = true
	}
}

func (p *requestObserver) OnLeave(t *streamxform.Transformer) {
	if t.Depth() != 2 {
		return
	}
	switch p.inKey {
	case "messages":
		if p.elemRoleSeen && p.elemRole == "user" {
			p.userMsgs++
		}
	case "contents":
		if !p.elemRoleSeen || p.elemRole == "user" {
			p.userConts++
		}
	}
}

// rounds 复刻官方的轮数规则：messages 是数组就用它，否则看 contents。
func (p *requestObserver) rounds() int {
	if p.msgsIsArray {
		return p.userMsgs
	}
	if p.contsIsArray {
		return p.userConts
	}
	return 0
}

// requestStreamable：轻量模式（没有任何属性从请求体提取）才走流式观察。
func requestStreamable(config AIStatisticsConfig) bool { return !config.shouldBufferRequestBody }

func onHttpStreamingRequestBody(ctx wrapper.HttpContext, config AIStatisticsConfig, chunk []byte, last bool) ([]byte, types.Action) {
	if ctx.GetBoolContext(SkipProcessing, false) {
		return chunk, types.ActionContinue
	}
	st, _ := ctx.GetContext(ctxKeyObserver).(*guard.State)
	if st == nil {
		tr := newRequestObserver()
		st = guard.New(&guard.Plan{
			Tr:   tr,
			Mode: guard.Observe,
			OnFinish: func(streamxform.Prelude) {
				p := tr.Protocol().(*requestObserver)
				model := "UNKNOWN"
				if p.modelSeen {
					model = p.model
				}
				finishRequestBody(ctx, model, p.rounds())
			},
			Metric: observeMetric,
			Log:    log.Warnf,
		})
		ctx.SetContext(ctxKeyObserver, st)
	}
	return st.Feed(chunk, last)
}

// finishRequestBody 是官方 onHttpRequestBody 取到 model 与轮数之后的收尾，两条路径共用。
func finishRequestBody(ctx wrapper.HttpContext, requestModel string, userPromptCount int) {
	// If model not found in body, try to extract from path (Gemini style)
	if requestModel == "UNKNOWN" {
		requestPath := ctx.GetStringContext(RequestPath, "")
		if strings.Contains(requestPath, "generateContent") || strings.Contains(requestPath, "streamGenerateContent") { // Google Gemini GenerateContent
			matches := geminiModelPathRe.FindStringSubmatch(requestPath)
			if len(matches) == 3 {
				requestModel = matches[2]
			}
		}
	}
	ctx.SetContext(tokenusage.CtxKeyRequestModel, requestModel)
	setSpanAttribute(ArmsRequestModel, requestModel)
	ctx.SetUserAttribute(ChatRound, userPromptCount)

	// Write log
	debugLogAiLog(ctx)
	_ = ctx.WriteUserAttributeToLogWithKey(wrapper.AILogKey)
}
