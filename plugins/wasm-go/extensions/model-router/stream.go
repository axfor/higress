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

// Streaming request body path.
//
// The buffered implementation collects the whole body (up to 100MB), reads model with gjson and rewrites it with sjson. Here
// model is only looked for at the start of the body (within the streamxform.CommitBytes window): once found the headers are set
// and it is rewritten in place; the bytes after that are forwarded verbatim without scanning. These cases keep the buffered path, byte-identical to it:
//   - non-JSON (multipart), a modelKey that is not a plain top-level key, no content-type;
//   - an auto route match (needs the last user message), a model that is not a string;
//   - model not seen inside the window (SDKs that put messages first, beyond the window).
// Known difference: with a JSON syntax error late in the body the buffered json.Valid fails and nothing is changed, while streaming has already rewritten the start and forwards the rest verbatim.

const ctxKeyStream = "model_router_stream"

var streamMetric = guard.NewMetric("model_router.stream")

// streamable reports whether this request can take the streaming path.
func streamable(config ModelRouterConfig, contentType string) bool {
	if !strings.Contains(contentType, "application/json") || config.modelKey == "" {
		return false
	}
	for i := 0; i < len(config.modelKey); i++ {
		c := config.modelKey[i]
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			return false // gjson path syntax (. # * ? @ | \): leave it to the buffered path
		}
	}
	return true
}

type jsonRoute struct {
	config   ModelRouterConfig
	model    string
	modelOK  bool // a string model was seen
	needFull bool // the whole body is required: auto route / non-string model
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

// onModel: the model value is complete. Decides whether to rewrite in place (strip the provider/ prefix).
func (r *jsonRoute) onModel(t *streamxform.Transformer, key string, raw []byte) ([]byte, bool) {
	var s string
	if json.Unmarshal(raw, &s) != nil {
		r.needFull = true // the buffered path applies gjson String() semantics to non-strings: let the full path reproduce that
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

// commit: before the headers are released. Same header order as the buffered handleJsonBody.
func (r *jsonRoute) commit(pre streamxform.Prelude, last bool) bool {
	if r.needFull {
		return false
	}
	if !r.modelOK {
		return last // the whole body arrived without a model: the buffered path leaves it alone; otherwise model may be beyond the window, fall back
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

// sjsonString reproduces sjson's string encoding: plain printable ASCII is just quoted, anything else goes through encoding/json.
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
