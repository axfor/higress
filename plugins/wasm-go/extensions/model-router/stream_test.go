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

// Integration tests of the streaming path: the body is fed chunk by chunk and compared with the buffered semantics (gjson takes the first model, sjson rewrites in place, request headers).

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
		require.Equal(t, types.ActionPause, acts[0], "should Pause before the commit point")
		require.Equal(t, types.ActionContinue, acts[len(acts)-1])
		nContinue := 0
		for _, a := range acts {
			if a == types.ActionContinue {
				nContinue++
			}
		}
		require.Greater(t, nContinue, len(acts)/2, "should Continue chunk by chunk after the commit point")
		want, _ := sjson.SetBytes(body, "model", "gpt-4o")
		require.Equal(t, string(want), string(up), "the upstream body should be byte-identical to sjson's in-place rewrite")
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
			require.Equal(t, types.ActionPause, a, "model beyond the window: should Pause until the last chunk, then take the buffered path")
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
		require.Equal(t, string(body), string(up), "the buffered path leaves invalid JSON untouched")
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
		require.Equal(t, string(body), string(up), "keepOriginalModelName: body untouched")
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
			require.Equal(t, types.ActionPause, a, "model not seen inside the window: fall back to the buffered path")
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

var earlyCommitConfig = func() json.RawMessage {
	data, _ := json.Marshal(map[string]interface{}{
		"modelKey":           "model",
		"addProviderHeader":  "x-provider",
		"modelToHeader":      "x-model",
		"enableOnPathSuffix": []string{"/v1/chat/completions"},
		"streamEarlyCommit":  true,
	})
	return data
}()

// With streamEarlyCommit the first chunk carrying model is released at once instead of waiting for the window.
func TestStream_EarlyCommit_ModelFirstBigBody(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		host, status := test.NewTestHost(earlyCommitConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
		body := []byte(`{"model" : "openai/gpt-4o" ,"messages":[{"role":"user","content":"` + bigContent(300<<10) + `"}]}` + "\n")
		acts, up := feedStream(host, body, 4096)
		require.Equal(t, types.ActionContinue, acts[0], "the first chunk carries model: released at once")
		want, _ := sjson.SetBytes(body, "model", "gpt-4o")
		require.Equal(t, string(want), string(up))
		require.Equal(t, "openai", header(host, "x-provider"))
		require.Equal(t, "openai/gpt-4o", header(host, "x-model"))
	})
}

// Early commit releases right after the model value, where the engine still holds the comma and the start
// of the next key; the body has to reach the upstream complete under small chunks.
func TestStream_EarlyCommit_SdkOrderSmallBody(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		for _, chunk := range []int{1, 5, 7, 64} {
			host, status := test.NewTestHost(earlyCommitConfig)
			require.Equal(t, types.OnPluginStartStatusOK, status)
			require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamHeaders("application/json")))
			body := []byte(`{"messages":[{"role":"user","content":"hi"}],"model":"openai/gpt-4o","stream":true}`)
			_, up := feedStream(host, body, chunk)
			want, _ := sjson.SetBytes(body, "model", "gpt-4o")
			require.Equal(t, string(want), string(up), "chunk=%d", chunk)
			require.Equal(t, "openai", header(host, "x-provider"))
			host.Reset()
		}
	})
}
