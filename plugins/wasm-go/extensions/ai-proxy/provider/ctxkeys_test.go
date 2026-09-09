package provider

import (
	"encoding/json"
	"math/rand"
	"testing"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
	"github.com/higress-group/wasm-go/pkg/wrapper"
)

// 响应侧的行为依赖请求阶段设下的上下文键：isStreaming 决定按 SSE 还是整份 JSON 解析响应，
// finalRequestModel 决定回填哪个模型名。这些键在缓冲路径里由结构体解码直接给出，
// 在流式路径里由 prelude 逐步凑出来 —— 两者必须给出同一组值，否则响应会被按错误的形态处理。
// 这一层过去没有测过：网关差分只能看到发往上游的请求，看不见上下文键。

type ctxProbe struct {
	wrapper.HttpContext
	m map[string]interface{}
}

func newCtxProbe() *ctxProbe { return &ctxProbe{m: map[string]interface{}{}} }

func (c *ctxProbe) SetContext(k string, v interface{}) { c.m[k] = v }
func (c *ctxProbe) GetContext(k string) interface{}    { return c.m[k] }
func (c *ctxProbe) GetBoolContext(k string, d bool) bool {
	if v, ok := c.m[k].(bool); ok {
		return v
	}
	return d
}
func (c *ctxProbe) GetStringContext(k, d string) string {
	if v, ok := c.m[k].(string); ok {
		return v
	}
	return d
}

// wantedContext 是缓冲路径会设出来的那一组值，直接从结构体解码得出 —— 缓冲路径做的就是这件事。
func wantedContext(in string, mapping map[string]string) (map[string]interface{}, bool) {
	req := &chatCompletionRequest{}
	if json.Unmarshal([]byte(in), req) != nil || len(req.Messages) == 0 || req.Model == "" {
		return nil, false // 缓冲路径会在这里失败，流式则回落，两边都不产生上下文
	}
	return map[string]interface{}{
		ctxKeyIsStreaming:          req.Stream,
		ctxKeyOriginalRequestModel: req.Model,
		ctxKeyFinalRequestModel:    getMappedModel(req.Model, mapping),
	}, true
}

// streamContext 跑流式路径的上下文写入。headersMutable=false 是提交点之后那一次，
// 也是唯一可能漏设键的时机；true 的那一次会写请求头，需要 wasm 宿主，由网关差分覆盖。
func streamContext(in string, mapping map[string]string, chunk int) (map[string]interface{}, bool) {
	c := &ProviderConfig{modelMapping: mapping}
	tr := typed(streamxform.NewClaude(streamxform.ClaudeOptions{
		MapModel: func(m string) (string, error) { return getMappedModel(m, mapping), nil },
	}))
	for i := 0; i < len(in); i += chunk {
		j := i + chunk
		if j > len(in) {
			j = len(in)
		}
		tr.Write([]byte(in[i:j]))
		tr.Out()
	}
	tr.Finish()
	if bad, _ := tr.Unsupported(); bad {
		return nil, false
	}
	pre := streamxform.Prelude{}
	if p, ok := tr.Protocol().(streamxform.Preluder); ok {
		pre = p.Prelude()
	}
	plan := &StreamPlan{Tr: tr, ApplyStream: true, ApplyModel: true}
	ctx := newCtxProbe()
	c.StreamApplyPrelude(ctx, ApiNameChatCompletion, plan, pre, false)
	c.StreamFinalizeContext(ctx, ApiNameChatCompletion, plan, pre)
	return ctx.m, true
}

func TestContextKeysMatchBufferedPath(t *testing.T) {
	mapping := map[string]string{"m": "mapped-m", "*": "fallback"}
	cases := []string{
		`{"model":"m","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m","stream":true,"messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m","stream":false,"messages":[{"role":"user","content":"U"}]}`,
		`{"messages":[{"role":"user","content":"U"}],"model":"m","stream":true}`,
		`{"model":"zzz","messages":[{"role":"user","content":"U"}]}`,
		`{"model":"m","stream":null,"messages":[{"role":"user","content":"U"}]}`,
	}
	for _, in := range cases {
		want, ok := wantedContext(in, mapping)
		if !ok {
			continue
		}
		for _, chunk := range []int{1, 7, 4096} {
			got, sok := streamContext(in, mapping, chunk)
			if !sok {
				t.Errorf("流式回落了，但缓冲路径能处理：%s", in)
				continue
			}
			for k, w := range want {
				if got[k] != w {
					t.Errorf("上下文键 %s 不一致（分块 %d）：流式 %v，缓冲 %v\n  输入 %s", k, chunk, got[k], w, in)
				}
			}
		}
	}
}

// 随机语料上跑同一条不变式：任何流式没有回落的请求，三个上下文键都必须与缓冲路径一致。
func TestContextKeysMatchOnRandomRequests(t *testing.T) {
	mapping := map[string]string{"m": "mapped-m"}
	r := rand.New(rand.NewSource(fuzzSeed()))
	checked, skipped := 0, 0
	for i := 0; i < 3000; i++ {
		in := genRequest(r)
		want, ok := wantedContext(in, mapping)
		if !ok {
			skipped++
			continue
		}
		got, sok := streamContext(in, mapping, []int{1, 17, 4096}[r.Intn(3)])
		if !sok {
			skipped++
			continue
		}
		checked++
		for k, w := range want {
			if got[k] != w {
				t.Fatalf("上下文键 %s 不一致：流式 %v，缓冲 %v\n  输入 %s", k, got[k], w, in)
			}
		}
	}
	t.Logf("比对 %d 例，跳过 %d 例（缓冲路径本就失败或流式回落）", checked, skipped)
}
