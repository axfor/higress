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

// Streaming observation of the request body.
//
// The buffered implementation always collects the whole request body just to read the top-level model and count user turns
// (messages[].role == "user", or for Gemini the contents[] elements whose role is missing or user). In lightweight mode (no
// request_body attribute source) all of that can be computed while scanning: the body is forwarded chunk by chunk without
// buffering a byte; the buffered path stays only when attributes are extracted from the request body (messages / question / system of the default set, or custom request_body attributes).
//
// Observation follows gjson semantics: the first of duplicate top-level keys wins; messages is used when it is an array (even empty),
// contents only otherwise; non-object elements and non-string roles are not counted. When the body is not valid JSON observation stops and forwarding continues verbatim.

const ctxKeyObserver = "ai_statistics_observer"

var observeMetric = guard.NewMetric("ai_statistics.stream")

// requestObserver is a read-only protocol: Capture the small fields, Skip the rest, the output is discarded.
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
	inKey        string // which top-level array we are in (messages / contents)
}

func newRequestObserver() *streamxform.Transformer {
	p := &requestObserver{seen: map[string]bool{}}
	tr := streamxform.NewTransformer(p)
	tr.DupKeyBail = false // gjson takes the first: duplicate keys are simply skipped
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
			return streamxform.Skip() // buffered IsArray() is false: not used
		}
		p.inKey = t.Last()
		if p.inKey == "messages" {
			p.msgsIsArray = true
		} else {
			p.contsIsArray = true
		}
		return streamxform.Enter().Lazy()
	case 2: // element
		if kind != streamxform.KindObject {
			if p.inKey == "contents" {
				p.userConts++ // gjson: a non-object element has no role → counts as user under the Gemini rule
			}
			return streamxform.Skip() // messages: a non-object element has an empty role, not counted
		}
		p.elemRole, p.elemRoleSeen = "", false
		return streamxform.Enter().Lazy()
	case 3: // role
		if kind != streamxform.KindString {
			p.elemRoleSeen = true // non-string: buffered String() is not "user"; Gemini treats it as "has a role but not user"
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
			p.model = string(raw) // gjson String(): numbers / booleans / containers give the raw text
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

// rounds reproduces the buffered turn rule: messages when it is an array, contents otherwise.
func (p *requestObserver) rounds() int {
	if p.msgsIsArray {
		return p.userMsgs
	}
	if p.contsIsArray {
		return p.userConts
	}
	return 0
}

// requestStreamable: only lightweight mode (no attribute extracted from the request body) uses streaming observation.
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

// finishRequestBody is the tail of the buffered onHttpRequestBody after model and turns are known, shared by both paths.
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
