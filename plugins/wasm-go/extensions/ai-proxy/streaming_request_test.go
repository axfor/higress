package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	wasmtest "github.com/higress-group/wasm-go/pkg/test"
	"github.com/stretchr/testify/require"
)

// Integration tests of streaming request body transformation: the host emulator feeds the body chunk by chunk and the
// control flow is checked: Pause before the commit point / Continue after it, fallback to the buffered path, failure after the commit point, passthrough.
// Equivalence of the converted output with the buffered implementation is covered by the differential tests in the provider package; only the integration layer is checked here.

var streamingClaudeConfig = json.RawMessage(`{"provider":{"type":"claude","apiTokens":["sk-test"],"modelMapping":{"m":"claude-3"}}}`)
var streamingGenericConfig = json.RawMessage(`{"provider":{"type":"generic","genericHost":"generic.example.com","apiTokens":["t"]}}`)

func streamingRequestHeaders(path string) [][2]string {
	return [][2]string{
		{":authority", "example.com"},
		{":path", path},
		{":method", "POST"},
		{"Content-Type", "application/json"},
		{"Content-Length", "1"},
	}
}

// feedChunks feeds the body to the plugin in chunks of the given size and returns the action per chunk and the concatenated upstream body.
func feedChunks(host wasmtest.TestHost, body []byte, chunk int) (actions []types.Action, upstream []byte) {
	for i := 0; i < len(body); i += chunk {
		j := i + chunk
		if j > len(body) {
			j = len(body)
		}
		a := host.CallOnHttpStreamingRequestBody(body[i:j], j == len(body))
		actions = append(actions, a)
		if a == types.ActionContinue {
			upstream = append(upstream, host.GetRequestBody()...)
		}
	}
	return
}

func requestHeader(host wasmtest.TestHost, name string) string {
	for _, h := range host.GetRequestHeaders() {
		if strings.EqualFold(h[0], name) {
			return h[1]
		}
	}
	return ""
}

func TestStreamingRequest_ClaudeSdkOrder(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		require.Equal(t, types.HeaderStopIteration, host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions")))

		// field order of the official SDK: messages first, model / stream at the end; 200KB of content, far beyond the commit point
		big := strings.Repeat("y", 200000)
		body := `{"messages":[{"role":"system","content":"S"},{"role":"user","content":"` + big + `"}],"model":"m","stream":true,"max_tokens":50}`
		actions, upstream := feedChunks(host, []byte(body), 4096)

		// always Pause before the commit point (64KB), Continue after it
		pauses := 0
		for _, a := range actions {
			if a == types.ActionPause {
				pauses++
			} else {
				break
			}
		}
		// the chunk that crosses the commit point returns Continue itself: every earlier chunk Paused, and the total lands right around 64KB
		require.Less(t, pauses*4096, 64<<10, "should keep Pausing before the commit point")
		require.GreaterOrEqual(t, (pauses+1)*4096, 64<<10, "should release as soon as the commit point is passed, not accumulate further")
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		require.Greater(t, len(actions)-pauses, 1, "should Continue chunk by chunk after the commit point instead of accumulating to the end")

		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out), "the upstream body should be valid JSON: %s", truncate(upstream))
		require.Equal(t, "claude-3", out["model"])
		require.Equal(t, "S", out["system"])
		require.Equal(t, true, out["stream"])
		require.Equal(t, float64(50), out["max_tokens"])
		msgs := out["messages"].([]any)
		require.Len(t, msgs, 1)
		require.Equal(t, "user", msgs[0].(map[string]any)["role"])
		require.Len(t, msgs[0].(map[string]any)["content"].(string), 200000)
	})
}

func TestStreamingRequest_AcceptHeaderWhenStreamKnownEarly(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		body := `{"model":"m","stream":true,"messages":[{"role":"user","content":"` + strings.Repeat("y", 200000) + `"}]}`
		_, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, "text/event-stream", requestHeader(host, "Accept"), "Accept should be rewritten when stream is known before the commit point")
		require.True(t, json.Valid(upstream))
	})
}

func TestStreamingRequest_FallbackBeforeCommit(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		// claude_content_blocks is not reproduced by streaming and appears before the commit point → Pause throughout, the last chunk takes the buffered path
		body := `{"model":"m","messages":[{"role":"user","claude_content_blocks":[{"type":"text","text":"B"}],"content":"` + strings.Repeat("y", 200000) + `"}]}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		for _, a := range actions[:len(actions)-1] {
			require.Equal(t, types.ActionPause, a)
		}
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		msgs := out["messages"].([]any)
		blocks := msgs[0].(map[string]any)["content"].([]any)
		require.Equal(t, "B", blocks[0].(map[string]any)["text"], "after the fallback this should be the buffered mapping of claude_content_blocks")
		require.Equal(t, "claude-3", out["model"])
	})
}

func TestStreamingRequest_FailAfterCommit(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		// the unsupported shape appears after 200KB: bytes already released, only failure is possible
		body := `{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("y", 200000) + `"},{"role":"user","content":"U","claude_content_blocks":[{"type":"text","text":"B"}]}]}`
		actions, _ := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[len(actions)-1])
		resp := host.GetLocalResponse()
		require.NotNil(t, resp, "a local 500 response should be sent")
		require.Equal(t, uint32(500), resp.StatusCode)
		require.Contains(t, string(resp.Data), "bailed after commit")
	})
}

func TestStreamingRequest_GenericPassthrough(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingGenericConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		body := "this is not even json " + strings.Repeat("z", 100000)
		actions, upstream := feedChunks(host, []byte(body), 4096)
		for _, a := range actions {
			require.Equal(t, types.ActionContinue, a, "generic should release every chunk directly")
		}
		require.Equal(t, body, string(upstream))
	})
}

func TestStreamingRequest_SingleChunk(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingClaudeConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
		require.Equal(t, types.ActionContinue, host.CallOnHttpStreamingRequestBody([]byte(body), true))
		var out map[string]any
		require.NoError(t, json.Unmarshal(host.GetRequestBody(), &out))
		require.Equal(t, "claude-3", out["model"])
		require.Equal(t, float64(4096), out["max_tokens"])
		require.Len(t, out["tools"].([]any), 1)
	})
}

func truncate(b []byte) string {
	if len(b) > 200 {
		return string(b[:200]) + "..."
	}
	return string(b)
}

var streamingGeminiConfig = json.RawMessage(`{"provider":{"type":"gemini","apiTokens":["g-key"]}}`)
var streamingOriginalOpenAIConfig = json.RawMessage(`{"provider":{"type":"openai","apiTokens":["t"],"protocol":"original","modelMapping":{"m":"never-applied"}}}`)
var streamingBedrockMantleConfig = json.RawMessage(`{"provider":{"type":"bedrock","apiTokens":["bk"],"awsRegion":"us-east-1","modelMapping":{"m":"anthropic.claude-3"}}}`)
var streamingVertexExpressConfig = json.RawMessage(`{"provider":{"type":"vertex","apiTokens":["vk"],"protocol":"original"}}`)

// The native Gemini endpoints are forwarded untouched by the buffered path; streaming forwards every chunk as it comes.
func TestStreamingRequest_GeminiNativePassthrough(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingGeminiConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1beta/models/gemini-2.0-flash:generateContent"))

		body := `{"contents":[{"role":"user","parts":[{"text":"` + strings.Repeat("g", 100000) + `"}]}],"generationConfig":{"temperature":0.2}}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		for _, a := range actions {
			require.Equal(t, types.ActionContinue, a, "native gemini should release every chunk directly")
		}
		require.Equal(t, body, string(upstream))
		require.Equal(t, "generativelanguage.googleapis.com", requestHeader(host, ":authority"))
		require.Equal(t, "g-key", requestHeader(host, "x-goog-api-key"))
		require.Equal(t, "/v1beta/models/gemini-2.0-flash:generateContent", requestHeader(host, ":path"))
	})
}

// Under the original protocol handleRequestBody returns before touching the body: the streaming path is a passthrough.
func TestStreamingRequest_OriginalProtocolPassthrough(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingOriginalOpenAIConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		body := `{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("o", 100000) + `"}],"stream":true}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		for _, a := range actions {
			require.Equal(t, types.ActionContinue, a, "original protocol should release every chunk directly")
		}
		require.Equal(t, body, string(upstream), "no model mapping under the original protocol")
		require.Equal(t, "api.openai.com", requestHeader(host, ":authority"))
		require.Equal(t, "Bearer t", requestHeader(host, "Authorization"))
	})
}

// Bedrock Mantle keeps the Anthropic body: with API tokens the buffered path only maps model and sets Accept from stream.
func TestStreamingRequest_BedrockMantleMessages(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingBedrockMantleConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/messages"))

		body := `{"model":"m","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"` + strings.Repeat("b", 100000) + `"}]}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0], "held until the commit point")
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		require.Equal(t, "anthropic.claude-3", out["model"])
		require.Equal(t, float64(64), out["max_tokens"])
		require.Len(t, out["messages"].([]any), 1)
		require.Equal(t, "bedrock-mantle.us-east-1.api.aws", requestHeader(host, ":authority"))
		require.Equal(t, "/anthropic/v1/messages", requestHeader(host, ":path"))
		require.Equal(t, "bk", requestHeader(host, "x-api-key"))
		require.Equal(t, "text/event-stream", requestHeader(host, "Accept"))
	})
}

// Vertex raw endpoints in Express mode: body untouched, the API key goes into the query string, the Authorization header goes away.
func TestStreamingRequest_VertexRawExpress(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingVertexExpressConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		hdrs := streamingRequestHeaders("/v1/publishers/google/models/gemini-2.0-flash:generateContent")
		hdrs = append(hdrs, [2]string{"Authorization", "Bearer client-token"})
		host.CallOnHttpRequestHeaders(hdrs)

		body := `{"contents":[{"role":"user","parts":[{"text":"` + strings.Repeat("v", 100000) + `"}]}]}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		for _, a := range actions {
			require.Equal(t, types.ActionContinue, a, "vertex raw should release every chunk directly")
		}
		require.Equal(t, body, string(upstream))
		require.Equal(t, "aiplatform.googleapis.com", requestHeader(host, ":authority"))
		require.Equal(t, "/v1/publishers/google/models/gemini-2.0-flash:generateContent?key=vk", requestHeader(host, ":path"))
		require.Equal(t, "", requestHeader(host, "Authorization"))
	})
}

var streamingVertexAnthropicExpressConfig = json.RawMessage(`{"provider":{"type":"vertex","apiTokens":["vk"],"modelMapping":{"m":"claude-sonnet-4@20250514"}}}`)

// Vertex /v1/messages in Express mode: the Anthropic body is kept, model moves into the path, the vertex-side fields are adjusted.
func TestStreamingRequest_VertexAnthropicExpress(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingVertexAnthropicExpressConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		hdrs := streamingRequestHeaders("/v1/messages")
		hdrs = append(hdrs, [2]string{"Authorization", "Bearer client-token"}, [2]string{"anthropic-version", "2023-06-01"})
		host.CallOnHttpRequestHeaders(hdrs)

		body := `{"model":"m","stream":true,"context_management":{"edits":[]},"messages":[{"role":"user","content":"` + strings.Repeat("a", 100000) + `"}]}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0], "held until the commit point")
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		_, hasModel := out["model"]
		require.False(t, hasModel, "vertex :rawPredict rejects model in the body")
		require.Equal(t, "vertex-2023-10-16", out["anthropic_version"])
		require.Equal(t, float64(4096), out["max_tokens"])
		_, hasCM := out["context_management"]
		require.False(t, hasCM)
		require.Len(t, out["messages"].([]any), 1)
		require.Equal(t, "/v1/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict?key=vk", requestHeader(host, ":path"))
		require.Equal(t, "aiplatform.googleapis.com", requestHeader(host, ":authority"))
		require.Equal(t, "", requestHeader(host, "Authorization"))
		require.Equal(t, "", requestHeader(host, "anthropic-version"))
	})
}

var streamingVertexExpressClaudeModelConfig = json.RawMessage(`{"provider":{"type":"vertex","apiTokens":["vk"],"modelMapping":{"m":"claude-sonnet-4@20250514","g":"gemini-2.0-flash"}}}`)

// Vertex chat with a claude-prefixed mapped model: the probe finds model, the Claude transformer takes over from the first byte.
func TestStreamingRequest_VertexChatClaudeModel(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingVertexExpressClaudeModelConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		hdrs := append(streamingRequestHeaders("/v1/chat/completions"), [2]string{"Authorization", "Bearer client-token"})
		host.CallOnHttpRequestHeaders(hdrs)

		body := `{"messages":[{"role":"system","content":"S"},{"role":"user","content":"` + strings.Repeat("c", 100000) + `"}],"model":"m","stream":true,"max_tokens":32}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0])
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		_, hasModel := out["model"]
		require.False(t, hasModel)
		require.Equal(t, "vertex-2023-10-16", out["anthropic_version"])
		require.Equal(t, float64(32), out["max_tokens"])
		require.Equal(t, "S", out["system"])
		require.Equal(t, true, out["stream"])
		require.Len(t, out["messages"].([]any), 1)
		require.Equal(t, "/v1/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict?key=vk", requestHeader(host, ":path"))
		require.Equal(t, "", requestHeader(host, "Authorization"))
	})
}

// A mapped model without the claude prefix takes Vertex's Gemini shape: the probe finds model, the Vertex transformer takes over.
func TestStreamingRequest_VertexChatGeminiModel(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingVertexExpressClaudeModelConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		body := `{"model":"g","stream":true,"messages":[{"role":"system","content":"S"},{"role":"user","content":"` + strings.Repeat("g", 100000) + `"}],"reasoning_effort":"low"}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0])
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		require.Less(t, 1, len(actions))
		require.Equal(t, types.ActionContinue, actions[len(actions)/2+8], "released past the window, not held for the buffered path")
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		contents := out["contents"].([]any)
		require.Len(t, contents, 3, "system as user, the dummy model message, the user message")
		require.Equal(t, "model", contents[1].(map[string]any)["role"])
		require.Equal(t, "Okay", contents[1].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"])
		_, hasSafety := out["safetySettings"]
		require.False(t, hasSafety, "nothing configured: omitted like the buffered omitempty field")
		gc := out["generationConfig"].(map[string]any)
		require.Equal(t, float64(1024), gc["thinkingConfig"].(map[string]any)["thinkingBudget"])
		require.Contains(t, requestHeader(host, ":path"), "gemini-2.0-flash:streamGenerateContent")
		require.Contains(t, requestHeader(host, ":path"), "key=vk")
	})
}

var streamingCohereConfig = json.RawMessage(`{"provider":{"type":"cohere","apiTokens":["ck"],"modelMapping":{"m":"command-r-plus"}}}`)

// Cohere rebuilds the request from the first message's text and a few scalars; everything else is dropped.
func TestStreamingRequest_CohereChat(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingCohereConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		text := strings.Repeat("q", 100000)
		// stream before the large message: past the window the Accept header can no longer be rewritten (documented)
		body := `{"model":"m","stream":true,"messages":[{"role":"user","content":"` + text + `"},{"role":"assistant","content":"dropped"}],"n":2,"top_p":0.5,"stop":["x"],"user":"u"}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0])
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		require.Equal(t, text, out["message"])
		require.Equal(t, "command-r-plus", out["model"])
		require.Equal(t, true, out["stream"])
		require.Equal(t, float64(2), out["k"])
		require.Equal(t, 0.5, out["p"])
		require.Equal(t, []any{"x"}, out["stop_sequences"])
		_, hasMessages := out["messages"]
		require.False(t, hasMessages)
		_, hasUser := out["user"]
		require.False(t, hasUser)
		require.Equal(t, "api.cohere.com", requestHeader(host, ":authority"))
		require.Equal(t, "/v1/chat", requestHeader(host, ":path"))
		require.Equal(t, "text/event-stream", requestHeader(host, "Accept"))
	})
}

var streamingDeepLConfig = json.RawMessage(`{"provider":{"type":"deepl","apiTokens":["dk"],"targetLang":"ZH"}}`)
var streamingKlingConfig = json.RawMessage(`{"provider":{"type":"kling","apiTokens":["kk"],"modelMapping":{"m":"kling-v2-master"}}}`)

// DeepL: messages become the text array, the system message the context, the host follows model.
func TestStreamingRequest_DeepL(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingDeepLConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		text := strings.Repeat("d", 100000)
		body := `{"model":"Pro","messages":[{"role":"system","content":"S"},{"role":"user","content":"` + text + `"},{"role":"user","content":"more"}]}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0])
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		require.Equal(t, []any{text, "more"}, out["text"])
		require.Equal(t, "ZH", out["target_lang"])
		require.Equal(t, "S", out["context"])
		require.Equal(t, "api.deepl.com", requestHeader(host, ":authority"))
		require.Equal(t, "/v2/translate", requestHeader(host, ":path"))
	})
}

// Kling: the body passes through with model mapped into model_name; an image key sends it to the image-to-video path.
func TestStreamingRequest_KlingVideos(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		for _, c := range []struct{ body, path, task string }{
			{`{"model":"m","prompt":"a cat","duration":"5"}`, "/v1/videos/text2video", "text2video"},
			{`{"prompt":"a cat","image":"data:image/png;base64,AAAA","model_name":"k"}`, "/v1/videos/image2video", "image2video"},
		} {
			host, status := wasmtest.NewTestHost(streamingKlingConfig)
			require.Equal(t, types.OnPluginStartStatusOK, status)
			host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/videos"))
			require.Equal(t, types.ActionContinue, host.CallOnHttpStreamingRequestBody([]byte(c.body), true))
			var out map[string]any
			require.NoError(t, json.Unmarshal(host.GetRequestBody(), &out))
			_, hasModel := out["model"]
			require.False(t, hasModel, c.body)
			require.Contains(t, []any{"kling-v2-master", "k"}, out["model_name"], c.body)
			require.Equal(t, "a cat", out["prompt"])
			require.Equal(t, c.path, requestHeader(host, ":path"), c.body)
			host.Reset()
		}
	})
}

var streamingMiniMaxProConfig = json.RawMessage(`{"provider":{"type":"minimax","apiTokens":["mk"],"minimaxApiType":"pro","minimaxGroupId":"g1","modelMapping":{"m":"abab6.5s-chat"}}}`)
var streamingDifyConfig = json.RawMessage(`{"provider":{"type":"dify","apiTokens":["dk"],"botType":"Chat"}}`)
var streamingTritonConfig = json.RawMessage(`{"provider":{"type":"triton","apiTokens":["tk"],"tritonDomain":"triton.local","tritonModelVersion":"2","modelMapping":{"m":"llama"}}}`)

// MiniMax Pro: system becomes bot_setting, user / assistant become sender messages, the GroupId goes into the path.
func TestStreamingRequest_MiniMaxPro(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingMiniMaxProConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		text := strings.Repeat("m", 100000)
		body := `{"model":"m","messages":[{"role":"system","name":"Bot","content":"S"},{"role":"user","content":"` + text + `"}],"stream":true,"max_tokens":7}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0])
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		require.Equal(t, "abab6.5s-chat", out["model"])
		require.Equal(t, true, out["stream"])
		require.Equal(t, float64(7), out["tokens_to_generate"])
		require.Equal(t, true, out["mask_sensitive_info"])
		msgs := out["messages"].([]any)
		require.Len(t, msgs, 1)
		require.Equal(t, "USER", msgs[0].(map[string]any)["sender_type"])
		require.Equal(t, text, msgs[0].(map[string]any)["text"])
		bots := out["bot_setting"].([]any)
		require.Equal(t, "Bot", bots[0].(map[string]any)["bot_name"])
		require.Equal(t, "Bot", out["reply_constraints"].(map[string]any)["sender_name"])
		require.Equal(t, "/v1/text/chatcompletion_pro?GroupId=g1", requestHeader(host, ":path"))
	})
}

// Dify: the conversation becomes one query string, written across the messages; the rest follows in Tail.
func TestStreamingRequest_Dify(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingDifyConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		hdrs := append(streamingRequestHeaders("/v1/chat/completions"), [2]string{"ConversationId", "conv-9"})
		host.CallOnHttpRequestHeaders(hdrs)

		text := strings.Repeat("d", 100000)
		body := `{"model":"m","stream":true,"messages":[{"role":"system","content":"S"},{"role":"user","content":"` + text + `"}],"user":"bob"}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0])
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		require.Equal(t, "SYSTEM: \nS\nUSER: \n"+text+"\n", out["query"])
		require.Equal(t, map[string]any{}, out["inputs"])
		require.Equal(t, "streaming", out["response_mode"])
		require.Equal(t, "bob", out["user"])
		require.Equal(t, "conv-9", out["conversation_id"])
		require.Equal(t, false, out["auto_generate_name"])
		require.Equal(t, "api.dify.ai", requestHeader(host, ":authority"))
	})
}

// Triton: id and text_input come from the last message; path and host are set from model and stream.
func TestStreamingRequest_Triton(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingTritonConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		text := strings.Repeat("t", 100000)
		body := `{"model":"m","stream":true,"messages":[{"id":"a","role":"user","content":"` + text + `"},{"id":"b","role":"user","content":"last"}],"temperature":0.5}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0])
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		require.Equal(t, "b", out["id"])
		require.Equal(t, "last", out["text_input"])
		require.Equal(t, map[string]any{"stream": true, "temperature": 0.5}, out["parameters"])
		require.Equal(t, "/v2/models/llama/versions/2/generate_stream", requestHeader(host, ":path"))
		require.Equal(t, "triton.local", requestHeader(host, ":authority"))
	})
}

var streamingBedrockConverseConfig = json.RawMessage(`{"provider":{"type":"bedrock","apiTokens":["bk"],"awsRegion":"us-east-1","modelMapping":{"m":"anthropic.claude-3-5-sonnet-20241022-v2:0"}}}`)

// Bedrock Converse with API tokens: the request is rebuilt in the Converse shape; path and Accept follow the buffered path.
func TestStreamingRequest_BedrockConverse(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingBedrockConverseConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))

		text := strings.Repeat("c", 100000)
		body := `{"model":"m","stream":true,"messages":[{"role":"system","content":"S"},{"role":"user","content":"` + text + `"}],"max_tokens":64,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0])
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		require.Equal(t, []any{map[string]any{"text": "S"}}, out["system"])
		msgs := out["messages"].([]any)
		require.Len(t, msgs, 1)
		require.Equal(t, text, msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"])
		require.Equal(t, float64(64), out["inferenceConfig"].(map[string]any)["maxTokens"])
		require.Equal(t, "standard", out["performanceConfig"].(map[string]any)["latency"])
		require.Contains(t, out["toolConfig"].(map[string]any)["toolChoice"], "auto")
		require.Equal(t, "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse-stream", requestHeader(host, ":path"))
		require.Equal(t, "*/*", requestHeader(host, "Accept"))
		require.Equal(t, "bedrock-runtime.us-east-1.amazonaws.com", requestHeader(host, ":authority"))
	})
}

var streamingGeminiEmbConfig = json.RawMessage(`{"provider":{"type":"gemini","apiTokens":["g-key"],"modelMapping":{"m":"text-embedding-004"}}}`)
var streamingQwenNativeEmbConfig = json.RawMessage(`{"provider":{"type":"qwen","apiTokens":["q"],"qwenEnableCompatible":false,"modelMapping":{"m":"text-embedding-v3"}}}`)
var streamingVertexEmbConfig = json.RawMessage(`{"provider":{"type":"vertex","apiTokens":["vk"],"modelMapping":{"m":"text-embedding-005"}}}`)

// Embeddings: the input array streams element by element into each provider's shape; the path follows the mapped model.
func TestStreamingRequest_Embeddings(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		big := strings.Repeat("e", 100000)
		body := `{"input":["` + big + `","second"],"model":"m","dimensions":3}`
		for _, c := range []struct {
			cfg   json.RawMessage
			check func(out map[string]any)
			path  string
		}{
			{streamingGeminiEmbConfig, func(out map[string]any) {
				reqs := out["requests"].([]any)
				require.Len(t, reqs, 2)
				require.Equal(t, "models/text-embedding-004", reqs[0].(map[string]any)["model"])
				require.Equal(t, "second", reqs[1].(map[string]any)["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"])
			}, "/v1beta/models/text-embedding-004:batchEmbedContents"},
			{streamingVertexEmbConfig, func(out map[string]any) {
				inst := out["instances"].([]any)
				require.Len(t, inst, 2)
				require.Equal(t, big, inst[0].(map[string]any)["content"])
				require.Equal(t, "", inst[0].(map[string]any)["task_type"])
			}, "/v1/publishers/google/models/text-embedding-005:predict?key=vk"},
			{streamingQwenNativeEmbConfig, func(out map[string]any) {
				require.Equal(t, "text-embedding-v3", out["model"])
				require.Equal(t, []any{big, "second"}, out["input"].(map[string]any)["texts"])
				require.Equal(t, map[string]any{}, out["parameters"])
			}, "/api/v1/services/embeddings/text-embedding/text-embedding"},
		} {
			host, status := wasmtest.NewTestHost(c.cfg)
			require.Equal(t, types.OnPluginStartStatusOK, status)
			host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/embeddings"))
			actions, upstream := feedChunks(host, []byte(body), 4096)
			require.Equal(t, types.ActionPause, actions[0])
			require.Equal(t, types.ActionContinue, actions[len(actions)-1])
			var out map[string]any
			require.NoError(t, json.Unmarshal(upstream, &out), truncate(upstream))
			c.check(out)
			require.Equal(t, c.path, requestHeader(host, ":path"))
			host.Reset()
		}
	})
}

// Gemini image generation: a tiny rebuilt body and the predict path.
func TestStreamingRequest_GeminiImageGeneration(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingGeminiEmbConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/images/generations"))
		require.Equal(t, types.ActionContinue, host.CallOnHttpStreamingRequestBody([]byte(`{"model":"m","prompt":"a cat","n":2,"size":"1024x1024"}`), true))
		var out map[string]any
		require.NoError(t, json.Unmarshal(host.GetRequestBody(), &out))
		require.Equal(t, []any{map[string]any{"prompt": "a cat"}}, out["instances"])
		require.Equal(t, map[string]any{"sampleCount": float64(2)}, out["parameters"])
		require.Equal(t, "/v1beta/models/text-embedding-004:predict", requestHeader(host, ":path"))
	})
}

var streamingClaudeFbtConfig = json.RawMessage(`{"provider":{"type":"claude","apiTokens":["sk-test"],"modelMapping":{"m":"claude-3"},"firstByteTimeout":3000}}`)
var streamingQwenNativeBasePathConfig = json.RawMessage(`{"provider":{"type":"qwen","apiTokens":["q"],"qwenEnableCompatible":false,"providerBasePath":"/pre","modelMapping":{"m":"qwen-vl-plus"}}}`)
var streamingQwenCompatBasePathConfig = json.RawMessage(`{"provider":{"type":"qwen","apiTokens":["q"],"providerBasePath":"/pre"}}`)

// firstByteTimeout: the header is set at the commit point from stream; stream past the window means the buffered path.
func TestStreamingRequest_FirstByteTimeout(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		const hdr = "x-envoy-upstream-rq-first-byte-timeout-ms"
		big := strings.Repeat("f", 100000)
		for _, c := range []struct {
			name, body, want string
			buffered         bool
		}{
			{"stream first", `{"model":"m","stream":true,"messages":[{"role":"user","content":"` + big + `"}]}`, "3000", false},
			{"not streaming", `{"model":"m","stream":false,"messages":[{"role":"user","content":"` + big + `"}]}`, "", false},
			{"stream past the window", `{"model":"m","messages":[{"role":"user","content":"` + big + `"}],"stream":true}`, "3000", true},
		} {
			host, status := wasmtest.NewTestHost(streamingClaudeFbtConfig)
			require.Equal(t, types.OnPluginStartStatusOK, status)
			host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))
			actions, upstream := feedChunks(host, []byte(c.body), 4096)
			require.Equal(t, types.ActionContinue, actions[len(actions)-1], c.name)
			if c.buffered {
				for i, a := range actions[:len(actions)-1] {
					require.Equal(t, types.ActionPause, a, "%s chunk %d: held for the buffered path", c.name, i)
				}
			} else {
				require.Equal(t, types.ActionContinue, actions[len(actions)/2+8], "%s: released past the window", c.name)
			}
			var out map[string]any
			require.NoError(t, json.Unmarshal(upstream, &out), c.name)
			require.Equal(t, "claude-3", out["model"], c.name)
			require.Equal(t, c.want, requestHeader(host, hdr), c.name)
			host.Reset()
		}
	})
}

// providerBasePath: applied again to a path set in the body phase (native qwen's multimodal endpoint), and streaming stays on.
func TestStreamingRequest_ProviderBasePath(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		big := strings.Repeat("p", 100000)
		// stream before the large message: native qwen's headers depend on it, so it has to be known at the commit point
		body := `{"model":"m","stream":false,"messages":[{"role":"user","content":"` + big + `"}]}`
		for _, c := range []struct {
			cfg  json.RawMessage
			path string
		}{
			{streamingQwenNativeBasePathConfig, "/pre/api/v1/services/aigc/multimodal-generation/generation"},
			{streamingQwenCompatBasePathConfig, "/pre/compatible-mode/v1/chat/completions"},
		} {
			func() {
				host, status := wasmtest.NewTestHost(c.cfg)
				defer host.Reset() // a failed assertion must not leave the host locked for the wasm mode
				require.Equal(t, types.OnPluginStartStatusOK, status)
				host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))
				actions, upstream := feedChunks(host, []byte(body), 4096)
				require.Equal(t, types.ActionPause, actions[0], c.path)
				require.Equal(t, types.ActionContinue, actions[len(actions)/2+8], "%s: released past the window: streamed, not buffered", c.path)
				require.Equal(t, types.ActionContinue, actions[len(actions)-1], c.path)
				require.NotEmpty(t, upstream, c.path)
				require.Equal(t, c.path, requestHeader(host, ":path"))
			}()
		}
	})
}

var streamingOpenAIForClaudeInputConfig = json.RawMessage(`{"provider":{"type":"openai","apiTokens":["t"],"modelMapping":{"claude-3":"gpt-4o"}}}`)

// A Claude request to an OpenAI provider: converted to OpenAI in the engine, then the provider's own transform.
func TestStreamingRequest_ClaudeInputToOpenAI(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(streamingOpenAIForClaudeInputConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/messages"))

		// many medium messages: each is held whole and converted when it closes, so the output grows as they come
		var turns []string
		for i := 0; i < 40; i++ {
			role := "user"
			if i%2 == 1 {
				role = "assistant"
			}
			turns = append(turns, `{"role":"`+role+`","content":"`+strings.Repeat("k", 3000)+`"}`)
		}
		body := `{"model":"claude-3","max_tokens":64,"stream":true,"system":"S","messages":[` + strings.Join(turns, ",") + `,{"role":"assistant","content":[{"type":"text","text":"calling"},{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"r1"}]}],"thinking":{"type":"enabled","budget_tokens":2048}}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		require.Equal(t, types.ActionPause, actions[0])
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		require.Equal(t, types.ActionContinue, actions[len(actions)*3/4], "released past the window: streamed, not buffered")
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out), truncate(upstream))
		require.Equal(t, "gpt-4o", out["model"])
		require.Equal(t, true, out["stream"])
		require.Equal(t, map[string]any{"include_usage": true}, out["stream_options"])
		require.Equal(t, float64(64), out["max_tokens"])
		require.Equal(t, "low", out["reasoning_effort"])
		msgs := out["messages"].([]any)
		require.Len(t, msgs, 43)
		require.Equal(t, "system", msgs[0].(map[string]any)["role"])
		require.Equal(t, strings.Repeat("k", 3000), msgs[1].(map[string]any)["content"])
		require.Equal(t, "calling", msgs[41].(map[string]any)["content"])
		require.Len(t, msgs[41].(map[string]any)["tool_calls"].([]any), 1)
		require.Equal(t, "tool", msgs[42].(map[string]any)["role"])
		require.Equal(t, "t1", msgs[42].(map[string]any)["tool_call_id"])
		_, hasBlocks := msgs[41].(map[string]any)["claude_content_blocks"]
		require.False(t, hasBlocks, "internal fields are stripped for an OpenAI provider")
		_, hasThinking := out["claude_thinking"]
		require.False(t, hasThinking)
		require.Equal(t, "/v1/chat/completions", requestHeader(host, ":path"))
		require.Equal(t, "api.openai.com", requestHeader(host, ":authority"))
	})
}

var streamingVertexImageConfig = json.RawMessage(`{"provider":{"type":"vertex","apiTokens":["vk"],"modelMapping":{"m":"gemini-2.5-flash-image"},"geminiSafetySetting":{"HARM_CATEGORY_HATE_SPEECH":"BLOCK_NONE"}}}`)

// Vertex images: generations rebuild a tiny body; edits and variations stream the image inputs (a data URL's payload
// goes out as it arrives) with the prompt written last; the path is generateContent for all three.
func TestStreamingRequest_VertexImages(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		const path = "/v1/publishers/google/models/gemini-2.5-flash-image:generateContent?key=vk"
		payload := strings.Repeat("iVBORw0KGgoAAAANSUhEUg", 15000) // ~330KB of base64
		parts := func(out map[string]any) []any {
			contents := out["contents"].([]any)
			require.Len(t, contents, 1)
			require.Equal(t, "user", contents[0].(map[string]any)["role"])
			return contents[0].(map[string]any)["parts"].([]any)
		}
		for _, c := range []struct {
			name, endpoint, body string
			streamed             bool
			check                func(out map[string]any)
		}{
			{"generation", "/v1/images/generations", `{"model":"m","prompt":"a cat","size":"1792x1024","output_format":"jpeg","n":2}`, false, func(out map[string]any) {
				require.Equal(t, []any{map[string]any{"text": "a cat"}}, parts(out))
				cfg := out["generationConfig"].(map[string]any)
				require.Equal(t, []any{"TEXT", "IMAGE"}, cfg["responseModalities"])
				img := cfg["imageConfig"].(map[string]any)
				require.Equal(t, "16:9", img["aspectRatio"])
				require.Equal(t, "2k", img["imageSize"])
				require.Equal(t, map[string]any{"mimeType": "image/jpeg"}, img["imageOutputOptions"])
				require.Equal(t, []any{map[string]any{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}}, out["safetySettings"])
			}},
			{"edit", "/v1/images/edits", `{"model":"m","prompt":"make it blue","image":"data:image/png;base64,` + payload + `","mask":""}`, true, func(out map[string]any) {
				p := parts(out)
				require.Len(t, p, 2)
				require.Equal(t, map[string]any{"inlineData": map[string]any{"mimeType": "image/png", "data": payload}}, p[0])
				require.Equal(t, map[string]any{"text": "make it blue"}, p[1])
			}},
			{"variation", "/v1/images/variations", `{"model":"m","images":[{"image_url":{"url":"data:image/jpeg;base64,` + payload + `"}},"https://x.test/a.png"]}`, true, func(out map[string]any) {
				p := parts(out)
				require.Len(t, p, 3)
				require.Equal(t, map[string]any{"inlineData": map[string]any{"mimeType": "image/jpeg", "data": payload}}, p[0])
				require.Equal(t, map[string]any{"fileData": map[string]any{"mimeType": "image/png", "fileUri": "https://x.test/a.png"}}, p[1])
				require.Equal(t, map[string]any{"text": "Create variations of the provided image."}, p[2])
			}},
		} {
			func() {
				host, status := wasmtest.NewTestHost(streamingVertexImageConfig)
				defer host.Reset()
				require.Equal(t, types.OnPluginStartStatusOK, status)
				host.CallOnHttpRequestHeaders(streamingRequestHeaders(c.endpoint))
				actions, upstream := feedChunks(host, []byte(c.body), 4096)
				require.Equal(t, types.ActionContinue, actions[len(actions)-1], c.name)
				if c.streamed {
					require.Equal(t, types.ActionContinue, actions[len(actions)/2], "%s: released past the window", c.name)
				}
				var out map[string]any
				require.NoError(t, json.Unmarshal(upstream, &out), c.name)
				c.check(out)
				require.Equal(t, path, requestHeader(host, ":path"), c.name)
				require.Equal(t, "application/json", requestHeader(host, "Content-Type"), c.name)
				require.Equal(t, "", requestHeader(host, "Authorization"), c.name)
			}()
		}
	})
}

// Vertex edits with the model after a large image: the path needs the model, so the request takes the buffered path
// and still comes out converted; a non-empty mask is rejected on both paths.
func TestStreamingRequest_VertexImagesFallback(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		payload := strings.Repeat("iVBORw0KGgoAAAANSUhEUg", 5000)
		host, status := wasmtest.NewTestHost(streamingVertexImageConfig)
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/images/edits"))
		body := `{"prompt":"blue","image":"data:image/png;base64,` + payload + `","model":"m"}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		for i, a := range actions[:len(actions)-1] {
			require.Equal(t, types.ActionPause, a, "chunk %d: held for the buffered path", i)
		}
		require.Equal(t, types.ActionContinue, actions[len(actions)-1])
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		p := out["contents"].([]any)[0].(map[string]any)["parts"].([]any)
		require.Len(t, p, 2)
		require.Equal(t, payload, p[0].(map[string]any)["inlineData"].(map[string]any)["data"])
		require.Equal(t, "/v1/publishers/google/models/gemini-2.5-flash-image:generateContent?key=vk", requestHeader(host, ":path"))
	})
}

// customSettings: the settings stage runs in front of every plan, as ReplaceByCustomSettings runs before every
// handler. One config per plan kind: a conversion (claude), a passthrough (native gemini under the original
// protocol, where the names are adjusted to gemini's), the Claude-input pipeline (three stages), and a replan (vertex).
func TestStreamingRequest_CustomSettings(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		big := strings.Repeat("s", 100000)
		// the Claude-input stage holds each message whole (and all of them until system, which comes first here):
		// many medium messages release as they close
		var msgs []string
		for i := 0; i < 40; i++ {
			role := "user"
			if i%2 == 1 {
				role = "assistant"
			}
			msgs = append(msgs, `{"role":"`+role+`","content":"`+strings.Repeat("t", 4000)+`"}`)
		}
		turns := strings.Join(msgs, ",")
		for _, c := range []struct {
			name, cfg, endpoint, body string
			check                     func(out map[string]any)
			path                      string
		}{
			{"claude", `{"provider":{"type":"claude","apiTokens":["sk-test"],"modelMapping":{"m":"claude-3"},"customSettings":[{"name":"max_tokens","value":100,"overwrite":false},{"name":"temperature","value":0.3},{"name":"stop","value":["END"],"mode":"raw"},{"name":"top_k","value":5,"mode":"raw"}]}}`,
				"/v1/chat/completions", `{"model":"m","temperature":1,"messages":[{"role":"user","content":"` + big + `"}],"max_tokens":50,"stop":["X"]}`,
				func(out map[string]any) {
					require.Equal(t, float64(50), out["max_tokens"], "present: not overwritten")
					require.Equal(t, 0.3, out["temperature"], "overwritten")
					require.Equal(t, []any{"END"}, out["stop_sequences"], "overwritten on the OpenAI body, converted")
					_, hasTopK := out["top_k"]
					require.False(t, hasTopK, "set on the OpenAI body, dropped by the conversion as buffered")
					require.Equal(t, "claude-3", out["model"])
				}, "/v1/messages"},
			{"gemini original", `{"provider":{"type":"gemini","apiTokens":["g-key"],"protocol":"original","customSettings":[{"name":"max_tokens","value":64},{"name":"temperature","value":0.9,"overwrite":false}]}}`,
				"/v1beta/models/gemini-2.0-flash:generateContent", `{"contents":[{"role":"user","parts":[{"text":"` + big + `"}]}],"generation_config":{"temperature":0.2,"maxOutputTokens":1}}`,
				func(out map[string]any) {
					require.Equal(t, map[string]any{"temperature": 0.2, "maxOutputTokens": float64(64)}, out["generation_config"])
					require.Len(t, out["contents"].([]any), 1)
				}, "/v1beta/models/gemini-2.0-flash:generateContent"},
			{"claude input pipeline", `{"provider":{"type":"openai","apiTokens":["t"],"modelMapping":{"claude-3":"gpt-4o"},"customSettings":[{"name":"max_tokens","value":77},{"name":"temperature","value":0.5,"overwrite":false}]}}`,
				"/v1/messages", `{"model":"claude-3","system":"S","max_tokens":10,"messages":[` + turns + `],"temperature":0.1}`,
				func(out map[string]any) {
					require.Equal(t, float64(77), out["max_tokens"], "set on the Claude body before the conversion")
					require.Equal(t, 0.1, out["temperature"])
					require.Equal(t, "gpt-4o", out["model"])
					require.Len(t, out["messages"].([]any), 41)
				}, "/v1/chat/completions"},
			{"vertex replan", `{"provider":{"type":"vertex","apiTokens":["vk"],"modelMapping":{"m":"claude-sonnet-4@20250514"},"customSettings":[{"name":"temperature","value":0.5},{"name":"max_tokens","value":99,"overwrite":false}]}}`,
				"/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"` + big + `"}],"temperature":1}`,
				func(out map[string]any) {
					require.Equal(t, 0.5, out["temperature"])
					require.Equal(t, float64(99), out["max_tokens"])
					require.Equal(t, "vertex-2023-10-16", out["anthropic_version"])
					require.Equal(t, true, out["stream"])
				}, "/v1/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict?key=vk"},
		} {
			func() {
				host, status := wasmtest.NewTestHost(json.RawMessage(c.cfg))
				defer host.Reset()
				require.Equal(t, types.OnPluginStartStatusOK, status, c.name)
				host.CallOnHttpRequestHeaders(streamingRequestHeaders(c.endpoint))
				actions, upstream := feedChunks(host, []byte(c.body), 4096)
				require.Equal(t, types.ActionContinue, actions[len(actions)-1], c.name)
				require.Equal(t, types.ActionContinue, actions[len(actions)-2], "%s: released past the window", c.name)
				var out map[string]any
				require.NoError(t, json.Unmarshal(upstream, &out), "%s: %s", c.name, truncate(upstream))
				c.check(out)
				require.Equal(t, c.path, requestHeader(host, ":path"), c.name)
			}()
		}
	})
}

// A customSettings path the stage does not take (an array index) keeps the buffered path, which still applies it.
func TestStreamingRequest_CustomSettingsFallback(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		host, status := wasmtest.NewTestHost(json.RawMessage(`{"provider":{"type":"claude","apiTokens":["sk-test"],"modelMapping":{"m":"claude-3"},"customSettings":[{"name":"stop.0","value":"END","mode":"raw"}]}}`))
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))
		body := `{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("f", 100000) + `"}]}`
		actions, upstream := feedChunks(host, []byte(body), 4096)
		for i, a := range actions[:len(actions)-1] {
			require.Equal(t, types.ActionPause, a, "chunk %d: held for the buffered path", i)
		}
		var out map[string]any
		require.NoError(t, json.Unmarshal(upstream, &out))
		require.Equal(t, []any{"END"}, out["stop_sequences"])
	})
}

// responseJsonSchema (openai / longcat): the configured schema replaces response_format, on the chat plan and behind
// the Claude-input conversion alike. The buffered path's incidental struct round trip is not reproduced (see the
// provider tests for the one visible difference).
func TestStreamingRequest_ResponseJsonSchema(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		const cfg = `{"provider":{"type":"openai","apiTokens":["t"],"modelMapping":{"claude-3":"gpt-4o","m":"gpt-4o"},"responseJsonSchema":{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"}}}}}`
		big := strings.Repeat("j", 100000)
		for _, c := range []struct{ name, endpoint, body string }{
			{"chat", "/v1/chat/completions", `{"model":"m","response_format":{"type":"text"},"messages":[{"role":"user","content":"` + big + `"}],"stream":true}`},
			{"claude input", "/v1/messages", `{"model":"claude-3","system":"S","max_tokens":10,"messages":[{"role":"user","content":"` + big + `"}]}`},
		} {
			func() {
				host, status := wasmtest.NewTestHost(json.RawMessage(cfg))
				defer host.Reset()
				require.Equal(t, types.OnPluginStartStatusOK, status, c.name)
				host.CallOnHttpRequestHeaders(streamingRequestHeaders(c.endpoint))
				actions, upstream := feedChunks(host, []byte(c.body), 4096)
				require.Equal(t, types.ActionContinue, actions[len(actions)-1], c.name)
				var out map[string]any
				require.NoError(t, json.Unmarshal(upstream, &out), "%s: %s", c.name, truncate(upstream))
				require.Equal(t, "json_schema", out["response_format"].(map[string]any)["type"], c.name)
				require.Equal(t, "answer", out["response_format"].(map[string]any)["json_schema"].(map[string]any)["name"], c.name)
				require.Equal(t, "gpt-4o", out["model"], c.name)
				require.Equal(t, "/v1/chat/completions", requestHeader(host, ":path"), c.name)
			}()
		}
	})
}

// The setting `context`: the first request takes the buffered path, which fetches the file and caches it; from then
// on chat requests stream with the file's system message inserted before the first non-system message (openai) or
// in front of system (claude).
func TestStreamingRequest_Context(t *testing.T) {
	wasmtest.RunTest(t, func(t *testing.T) {
		const ctxCfg = `,"context":{"fileUrl":"http://ctxfile/context.txt","serviceName":"ctxfile.default.svc.cluster.local","servicePort":80}`
		const fileContent = "Context from the file."
		big := strings.Repeat("x", 100000)
		for _, c := range []struct {
			name, cfg string
			check     func(out map[string]any)
		}{
			{"openai", `{"provider":{"type":"openai","apiTokens":["t"],"modelMapping":{"m":"gpt-4o"}` + ctxCfg + `}}`, func(out map[string]any) {
				msgs := out["messages"].([]any)
				require.Len(t, msgs, 3)
				require.Equal(t, "S", msgs[0].(map[string]any)["content"])
				require.Equal(t, map[string]any{"role": "system", "content": fileContent}, msgs[1])
				require.Equal(t, "user", msgs[2].(map[string]any)["role"])
			}},
			{"claude", `{"provider":{"type":"claude","apiTokens":["sk-test"],"modelMapping":{"m":"claude-3"}` + ctxCfg + `}}`, func(out map[string]any) {
				require.Equal(t, fileContent+"\nS", out["system"])
				require.Len(t, out["messages"].([]any), 1)
			}},
		} {
			func() {
				host, status := wasmtest.NewTestHost(json.RawMessage(c.cfg))
				defer host.Reset()
				require.Equal(t, types.OnPluginStartStatusOK, status, c.name)
				body := `{"model":"m","messages":[{"role":"system","content":"S"},{"role":"user","content":"` + big + `"}]}`

				// first request: buffered, fetches the file
				host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))
				actions, _ := feedChunks(host, []byte(body), 4096)
				require.Equal(t, types.ActionPause, actions[len(actions)-1], "%s: the buffered path waits for the file", c.name)
				require.Len(t, host.GetHttpCalloutAttributes(), 1, c.name)
				host.CallOnHttpCall([][2]string{{":status", "200"}}, []byte(fileContent))
				var out map[string]any
				require.NoError(t, json.Unmarshal(host.GetRequestBody(), &out), "%s: %s", c.name, truncate(host.GetRequestBody()))
				c.check(out)
				host.CompleteHttp()

				// second request: the file is cached, the request streams
				host.CallOnHttpRequestHeaders(streamingRequestHeaders("/v1/chat/completions"))
				actions, upstream := feedChunks(host, []byte(body), 4096)
				require.Equal(t, types.ActionContinue, actions[len(actions)-1], c.name)
				require.Equal(t, types.ActionContinue, actions[len(actions)-2], "%s: released past the window", c.name)
				require.Empty(t, host.GetHttpCalloutAttributes(), "%s: no second fetch", c.name)
				out = nil
				require.NoError(t, json.Unmarshal(upstream, &out), "%s: %s", c.name, truncate(upstream))
				c.check(out)
			}()
		}
	})
}
