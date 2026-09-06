package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/higress-group/wasm-go/pkg/test"
	"github.com/stretchr/testify/require"
)

// 流式观察的集成测试：轻量模式下逐块 Continue、body 原样转发、model 与轮数写入上下文；
// 默认属性集（要从请求体提取 messages 等）保持官方缓冲路径。

func statHeaders() [][2]string {
	return [][2]string{{":authority", "example.com"}, {":path", "/v1/chat/completions"}, {":method", "POST"}, {"Content-Type", "application/json"}, {"Content-Length", "1"}}
}

func TestStreamObserve_Lightweight(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		cfg, _ := json.Marshal(map[string]interface{}{"use_default_response_attributes": true})
		host, status := test.NewTestHost(cfg)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(statHeaders())
		body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"` + strings.Repeat("y", 300<<10) + `"},{"role":"assistant","content":"a"},{"role":"user","content":"b"}]}`)
		var upstream []byte
		for i := 0; i < len(body); i += 4096 {
			j := i + 4096
			if j > len(body) {
				j = len(body)
			}
			require.Equal(t, types.ActionContinue, host.CallOnHttpStreamingRequestBody(body[i:j], j == len(body)), "观察形态永远 Continue")
			upstream = append(upstream, host.GetRequestBody()...)
		}
		require.Equal(t, string(body), string(upstream), "请求体原样转发")
		attrs := getAILogAttributes(t, host)
		round, ok := aiLogInt64(attrs, ChatRound)
		require.True(t, ok)
		require.Equal(t, int64(2), round)
		model, ok := getSpanValue(host, ArmsRequestModel)
		require.True(t, ok)
		require.Equal(t, "gpt-4o", model)
	})
}

func TestStreamObserve_DefaultAttributesKeepBuffering(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(json.RawMessage(`{}`))
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(statHeaders())
		body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)
		// 官方路径：整份 body 一次交付
		require.Equal(t, types.ActionContinue, host.CallOnHttpRequestBody(body))
		attrs := getAILogAttributes(t, host)
		round, ok := aiLogInt64(attrs, ChatRound)
		require.True(t, ok)
		require.Equal(t, int64(1), round)
		model, ok := getSpanValue(host, ArmsRequestModel)
		require.True(t, ok)
		require.Equal(t, "gpt-4o", model)
	})
}
