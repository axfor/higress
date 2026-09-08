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
