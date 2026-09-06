package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/higress-group/wasm-go/pkg/test"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/sjson"
)

// 流式路径的集成测试：按块喂 body，对照官方语义（gjson 取首个 model、sjson 原地改写、请求头）。

func streamHeaders(ct string) [][2]string {
	return [][2]string{{":authority", "example.com"}, {":path", "/v1/chat/completions"}, {":method", "POST"}, {"Content-Type", ct}, {"Content-Length", "1"}}
}

func feedStream(host test.TestHost, body []byte, chunk int) (acts []types.Action, upstream []byte) {
	for i := 0; i < len(body); i += chunk {
		j := i + chunk
		if j > len(body) {
			j = len(body)
		}
		a := host.CallOnHttpStreamingRequestBody(body[i:j], j == len(body))
		acts = append(acts, a)
		if a == types.ActionContinue {
			upstream = append(upstream, host.GetRequestBody()...)
		}
	}
	return
}

func header(host test.TestHost, name string) string {
	for _, h := range host.GetRequestHeaders() {
		if strings.EqualFold(h[0], name) {
			return h[1]
		}
	}
	return ""
}

func bigContent(n int) string { return strings.Repeat("y", n) }

func TestStream_ModelFirstBigBody(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(basicConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
		body := []byte(`{"model" : "openai/gpt-4o" ,"messages":[{"role":"user","content":"` + bigContent(300<<10) + `"}]}` + "\n")
		acts, up := feedStream(host, body, 4096)
		require.Equal(t, types.ActionPause, acts[0], "提交点前应 Pause")
		require.Equal(t, types.ActionContinue, acts[len(acts)-1])
		nContinue := 0
		for _, a := range acts {
			if a == types.ActionContinue {
				nContinue++
			}
		}
		require.Greater(t, nContinue, len(acts)/2, "提交点后应逐块 Continue")
		want, _ := sjson.SetBytes(body, "model", "gpt-4o")
		require.Equal(t, string(want), string(up), "上游 body 应与 sjson 原地改写逐字节一致")
		require.Equal(t, "openai/gpt-4o", header(host, "x-model"))
		require.Equal(t, "openai", header(host, "x-provider"))
	})
}

func TestStream_SdkOrderBeyondWindowFallsBack(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(basicConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
		body := []byte(`{"messages":[{"role":"user","content":"` + bigContent(300<<10) + `"}],"model":"openai/gpt-4o"}`)
		acts, up := feedStream(host, body, 4096)
		for _, a := range acts[:len(acts)-1] {
			require.Equal(t, types.ActionPause, a, "model 在窗口之外：应一直 Pause 到末块再走官方路径")
		}
		require.Equal(t, types.ActionContinue, acts[len(acts)-1])
		want, _ := sjson.SetBytes(body, "model", "gpt-4o")
		require.Equal(t, string(want), string(up))
		require.Equal(t, "openai/gpt-4o", header(host, "x-model"))
		require.Equal(t, "openai", header(host, "x-provider"))
	})
}

func TestStream_SdkOrderSmallBody(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(basicConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
		body := []byte(`{"messages":[{"role":"user","content":"hi"}],"model":"openai/gpt-4o","stream":true}`)
		_, up := feedStream(host, body, 7)
		want, _ := sjson.SetBytes(body, "model", "gpt-4o")
		require.Equal(t, string(want), string(up))
		require.Equal(t, "openai", header(host, "x-provider"))
	})
}

func TestStream_AutoRoutingFallsBack(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		cfg, _ := json.Marshal(map[string]interface{}{
			"modelToHeader": "x-model",
			"autoRouting":   map[string]interface{}{"enable": true, "defaultModel": "gpt-4o-mini", "rules": []map[string]string{{"pattern": "(?i)code", "model": "gpt-4o"}}},
		})
		host, status := test.NewTestHost(cfg)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
		body := []byte(`{"model":"higress/auto","messages":[{"role":"user","content":"` + bigContent(100<<10) + ` please write code"}]}`)
		acts, up := feedStream(host, body, 4096)
		for _, a := range acts[:len(acts)-1] {
			require.Equal(t, types.ActionPause, a)
		}
		want, _ := sjson.SetBytes(body, "model", "gpt-4o")
		require.Equal(t, string(want), string(up))
		require.Equal(t, "gpt-4o", header(host, "x-higress-llm-model"))
	})
}

func TestStream_InvalidJsonFallsBack(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(basicConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
		body := []byte(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":tru}`)
		_, up := feedStream(host, body, 5)
		require.Equal(t, string(body), string(up), "官方对非法 JSON 整体不动")
		require.Equal(t, "", header(host, "x-provider"))
	})
}

func TestStream_KeepOriginalAndModelToHeaderOnly(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		cfg, _ := json.Marshal(map[string]interface{}{"modelToHeader": "x-model", "addProviderHeader": "x-provider", "keepOriginalModelName": true})
		host, status := test.NewTestHost(cfg)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
		body := []byte(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"` + bigContent(200<<10) + `"}]}`)
		_, up := feedStream(host, body, 4096)
		require.Equal(t, string(body), string(up), "keepOriginalModelName：body 不动")
		require.Equal(t, "openai/gpt-4o", header(host, "x-model"))
		require.Equal(t, "openai", header(host, "x-provider"))
	})
}

func TestStream_NoModel(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(basicConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
		body := []byte(`{"messages":[{"role":"user","content":"` + bigContent(200<<10) + `"}]}`)
		acts, up := feedStream(host, body, 4096)
		for _, a := range acts[:len(acts)-1] {
			require.Equal(t, types.ActionPause, a, "窗口内没见到 model：回落官方")
		}
		require.Equal(t, string(body), string(up))
		require.Equal(t, "", header(host, "x-model"))
	})
}

func TestStream_NestedModelKeyUsesOfficialPath(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		cfg, _ := json.Marshal(map[string]interface{}{"modelKey": "meta.model", "modelToHeader": "x-model"})
		host, status := test.NewTestHost(cfg)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
		body := []byte(`{"meta":{"model":"m1"},"messages":[]}`)
		require.Equal(t, types.ActionContinue, host.CallOnHttpRequestBody(body))
		require.Equal(t, "m1", header(host, "x-model"))
	})
}
