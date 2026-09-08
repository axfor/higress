package provider

import (
	"errors"
	"sort"
	"strings"

	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-proxy/util"
	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm"
	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
)

// Entry points of streaming request body transformation.
//
// Two questions are answered here: (1) can this provider + apiName + configuration stream; (2) when it does, who applies
// the side effects the buffered path does "on the way" (request headers, context keys).
// Every condition is derived line by line from handleRequestBody / defaultTransformRequestBody; any setting under which
// the buffered path would touch the body further is declared not applicable and falls back to that path instead of guessing.

// StreamPlan is the streaming plan of one request.
type StreamPlan struct {
	Tr *streamxform.Transformer
	// Passthrough: the buffered path does not touch the body at all (generic); chunks are released directly without a transformer.
	Passthrough bool
	// ApplyStream: the buffered path sets the Accept header and isStreaming according to stream
	// (true for chat / videos / videoremix on the default path; false in Qwen compatible mode and for non-streaming endpoints).
	ApplyStream bool
	// ApplyModel: the buffered path writes the originalRequestModel / finalRequestModel context keys.
	ApplyModel bool
	// NoAcceptHeader: the buffered path calls parseRequestAndMapModel but then overwrites the headers with the snapshot taken
	// at the start of the body phase (ReplaceRequestHeaders), so the Accept rewrite never takes effect; Gemini works this way.
	NoAcceptHeader bool
	// RequireModelBeforeCommit: model must have been seen before the request headers are released (the request path depends on it).
	// Not seen by the commit point means fallback; this is where "bounded lookahead, fall back past the window" lands.
	RequireModelBeforeCommit bool
	// RequireStreamBeforeCommit: likewise for stream (Gemini's generateContent / streamGenerateContent).
	// Not seen by the end of the body counts as false, as on the buffered path.
	RequireStreamBeforeCommit bool
	// AfterPrelude is called after the context keys are written and before the headers are released: header changes that depend on body facts (the Azure path).
	AfterPrelude func(ctx wrapper.HttpContext)
	// OnFinish is called after the whole body has been scanned: context keys that are only known at the end and only used on the response side.
	OnFinish func(ctx wrapper.HttpContext)
}

// streamDefaultProviders are the providers that go through defaultTransformRequestBody
// (no TransformRequestBody*, or one that only delegates to the default implementation).
var streamDefaultProviders = map[string]bool{
	providerTypeAi360: true, providerTypeBaichuan: true, providerTypeBaidu: true,
	providerTypeCloudflare: true, providerTypeDeepSeek: true, providerTypeFireworks: true,
	providerTypeGaladriel: true, providerTypeGithub: true, providerTypeGrok: true,
	providerTypeGroq: true, providerTypeMistral: true, providerTypeMoonshot: true,
	providerTypeOllama: true, providerTypeSpark: true, providerTypeStepfun: true,
	providerTypeTogetherAI: true, providerTypeYi: true,
	providerTypeVllm: true, // TransformRequestBody only delegates to the default implementation
}

// NewStreamPlan picks the streaming protocol for one request. A nil plan comes with why; the caller takes the buffered path.
func (c *ProviderConfig) NewStreamPlan(ctx wrapper.HttpContext, apiName ApiName, prov Provider) (plan *StreamPlan, why string) {
	if c.typ == providerTypeGeneric {
		// generic's OnRequestBody writes the body back unchanged without handleRequestBody:
		// only the two settings in main.go that apply to every provider touch the body / context
		if len(c.customSettings) > 0 {
			return nil, "customSettings rewrites the body"
		}
		if c.IsRetryOnFailureEnabled() {
			return nil, "retryOnFailure needs the whole body stored in the context"
		}
		return &StreamPlan{Passthrough: true}, ""
	}
	if c.IsOriginal() {
		return nil, "original protocol"
	}
	if c.firstByteTimeout != 0 {
		return nil, "firstByteTimeout needs stream before the headers are released, and the field position is not under our control"
	}
	if len(c.customSettings) > 0 {
		return nil, "customSettings rewrites the body"
	}
	if c.context != nil {
		return nil, "context injection needs the whole body"
	}
	if len(c.contextCleanupCommands) > 0 {
		return nil, "contextCleanupCommands"
	}
	if c.mergeConsecutiveMessages {
		return nil, "mergeConsecutiveMessages"
	}
	if c.IsRetryOnFailureEnabled() {
		return nil, "retryOnFailure needs the whole body stored in the context"
	}
	if need, _ := ctx.GetContext("needClaudeResponseConversion").(bool); need {
		return nil, "automatic conversion of Claude protocol input"
	}
	if !c.isSupportedAPI(apiName) {
		return nil, "apiName not supported"
	}
	if !c.needToProcessRequestBody(apiName) {
		return nil, "the buffered path does not handle the request body for this apiName"
	}
	if ct, _ := proxywasm.GetHttpRequestHeader("content-type"); !strings.Contains(ct, "application/json") {
		return nil, "non-JSON request body (multipart etc.) takes the buffered path"
	}
	isChat := apiName == ApiNameChatCompletion
	// defaultTransformRequestBody reads stream only for these three endpoint kinds
	detectStream := isChat || apiName == ApiNameVideos || apiName == ApiNameVideoRemix
	mapLenient := func(m string) string { return getMappedModel(m, c.modelMapping) }
	normalize := c.IsOpenAIProtocol() && !c.IsGeneric() && (isChat || apiName == ApiNameCompletion) && !c.disableStreamUsageStats
	defaultOpts := func(v streamxform.OpenAIVariant) streamxform.OpenAIOptions {
		return streamxform.OpenAIOptions{
			MapModel:               mapLenient,
			DetectStream:           detectStream,
			NormalizeUsage:         normalize,
			DeveloperRoleSupported: isDeveloperRoleSupported(c.typ),
			CheckMessages:          isChat,
			Variant:                v,
		}
	}
	defaultPlan := func(v streamxform.OpenAIVariant) *StreamPlan {
		return &StreamPlan{Tr: streamxform.NewOpenAI(defaultOpts(v)), ApplyStream: detectStream, ApplyModel: true}
	}
	// among the providers on the default path, these endpoint kinds are handled separately by the buffered path
	inDefaultApis := true
	if c.typ == providerTypeDoubao && (apiName == ApiNameResponses || apiName == ApiNameImageGeneration) {
		inDefaultApis = false
	}

	switch {
	case c.typ == providerTypeClaude && !isChat:
		// /v1/messages (native Claude protocol), /v1/complete, embeddings: the buffered path uses defaultTransformRequestBody
		return defaultPlan(nil), ""

	case c.typ == providerTypeClaude:
		return checkChatRequestTypes(&StreamPlan{Tr: streamxform.NewClaude(streamxform.ClaudeOptions{
			MapModel: func(m string) (string, error) {
				if m == "" {
					return "", errors.New("missing model in request")
				}
				mapped := getMappedModel(m, c.modelMapping)
				if mapped == "" {
					return "", errors.New("model becomes empty after applying the configured mapping")
				}
				return mapped, nil
			},
			ClaudeCodeMode: c.claudeCodeMode,
		}), ApplyStream: true, ApplyModel: true}), ""

	case c.typ == providerTypeQwen && !c.qwenEnableCompatible:
		// native DashScope protocol: the buffered onChatCompletionRequestBody changes the path and headers by model / stream in the body phase
		if !isChat {
			return nil, "native qwen protocol streams chat completion only"
		}
		if c.providerBasePath != "" {
			return nil, "providerBasePath needs :path changed in the body phase"
		}
		if len(c.qwenFileIds) > 0 {
			return nil, "qwenFileIds inserts file messages into messages"
		}
		mapStrict := func(m string) (string, error) {
			if m == "" {
				return "", errors.New("missing model in request")
			}
			mapped := getMappedModel(m, c.modelMapping)
			if mapped == "" {
				return "", errors.New("model becomes empty after applying the configured mapping")
			}
			return mapped, nil
		}
		tr := streamxform.NewQwenNative(streamxform.QwenNativeOptions{
			MapModel:                 mapStrict,
			SupportsPreserveThinking: qwenSupportsPreserveThinking,
			EnableSearch:             c.qwenEnableSearch,
			DeveloperToSystem:        !isDeveloperRoleSupported(c.typ),
		})
		p := checkChatRequestTypes(&StreamPlan{Tr: tr, ApplyStream: true, ApplyModel: true})
		p.RequireModelBeforeCommit = true
		p.RequireStreamBeforeCommit = true
		p.AfterPrelude = func(ctx wrapper.HttpContext) {
			// reproduce the header / path handling of onChatCompletionRequestBody
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			if strings.HasPrefix(model, qwenVlModelPrefixName) {
				_ = util.OverwriteRequestPath(qwenMultimodalGenerationPath)
			}
			if stream, _ := ctx.GetContext(ctxKeyIsStreaming).(bool); stream {
				_ = proxywasm.ReplaceHttpRequestHeader("Accept", "text/event-stream")
				_ = proxywasm.ReplaceHttpRequestHeader("X-DashScope-SSE", "enable")
			} else {
				_ = proxywasm.ReplaceHttpRequestHeader("Accept", "*/*")
				_ = proxywasm.RemoveHttpRequestHeader("X-DashScope-SSE")
			}
		}
		p.OnFinish = func(ctx wrapper.HttpContext) {
			if stream, _ := ctx.GetContext(ctxKeyIsStreaming).(bool); stream {
				if q, ok := tr.Protocol().(interface{ IncrementalOutput() bool }); ok {
					ctx.SetContext(ctxKeyIncrementalStreaming, q.IncrementalOutput())
				}
			}
		}
		return p, ""

	case c.typ == providerTypeQwen:
		if c.providerBasePath != "" {
			return nil, "providerBasePath needs :path changed in the body phase"
		}
		if !inDefaultApis {
			return nil, "this apiName is not covered by streaming"
		}
		opts := defaultOpts(&streamxform.QwenVariant{SupportsPreserveThinking: qwenSupportsPreserveThinking})
		opts.ModelOnlyIfPresent = true
		opts.DetectStream = false // the compatible branch does not call defaultTransformRequestBody: no Accept / isStreaming
		return &StreamPlan{Tr: streamxform.NewOpenAI(opts), ApplyStream: false, ApplyModel: false}, ""

	case c.typ == providerTypeMinimax:
		// V2 endpoint (default): the buffered handleRequestBodyByChatCompletionV2 only changes model and pins the path to chatcompletion_v2;
		// the Pro endpoint has a request structure of its own and does not stream.
		if c.minimaxApiType == minimaxApiTypePro {
			return nil, "the minimax Pro endpoint is not covered by streaming"
		}
		if c.providerBasePath != "" {
			return nil, "providerBasePath needs :path changed in the body phase"
		}
		if !isChat {
			return nil, "minimax streams chat completion only"
		}
		opts := defaultOpts(nil)
		opts.DetectStream = false // this buffered branch sets neither Accept / isStreaming nor the model context keys
		p := &StreamPlan{Tr: streamxform.NewOpenAI(opts), ApplyStream: false, ApplyModel: false}
		p.AfterPrelude = func(ctx wrapper.HttpContext) {
			// the buffered path only switches to the v2 endpoint in the body phase (not in the header phase); the path is fixed and independent of body fields
			if err := util.OverwriteRequestPath(minimaxChatCompletionV2Path); err != nil {
				log.Errorf("minimaxProvider: overwrite request path failed: %v", err)
			}
		}
		return p, ""

	case c.typ == providerTypeZhipuAi:
		if !inDefaultApis {
			return nil, "this apiName is not covered by streaming"
		}
		var v streamxform.OpenAIVariant
		if isChat {
			v = &streamxform.ZhipuVariant{}
		}
		return defaultPlan(v), ""

	case c.typ == providerTypeOpenRouter:
		if !inDefaultApis {
			return nil, "this apiName is not covered by streaming"
		}
		var v streamxform.OpenAIVariant
		if isChat {
			v = &streamxform.OpenRouterVariant{}
		}
		return defaultPlan(v), ""

	case c.typ == providerTypeGemini:
		gp, ok := prov.(*geminiProvider)
		if !ok {
			return nil, "unexpected gemini provider instance type"
		}
		if !isChat {
			return nil, "gemini streams chat completion only"
		}
		var ss []streamxform.GeminiSafetySetting
		for k, v := range c.geminiSafetySetting {
			ss = append(ss, streamxform.GeminiSafetySetting{Category: k, Threshold: v})
		}
		sort.Slice(ss, func(i, j int) bool { return ss[i].Category < ss[j].Category })
		mapStrict := func(m string) (string, error) {
			if m == "" {
				return "", errors.New("missing model in request")
			}
			mapped := getMappedModel(m, c.modelMapping)
			if mapped == "" {
				return "", errors.New("model becomes empty after applying the configured mapping")
			}
			return mapped, nil
		}
		p := checkChatRequestTypes(&StreamPlan{Tr: streamxform.NewGemini(streamxform.GeminiOptions{
			MapModel:       mapStrict,
			ThinkingModel:  func(m string) bool { return geminiThinkingModels[m] },
			ThinkingBudget: c.geminiThinkingBudget,
			SafetySettings: ss,
		}), ApplyStream: true, ApplyModel: true, NoAcceptHeader: true})
		// buffered onChatCompletionRequestBody: path = /{version}/models/{mapped model}:{generateContent|streamGenerateContent}
		p.RequireModelBeforeCommit = true
		p.RequireStreamBeforeCommit = true
		p.AfterPrelude = func(ctx wrapper.HttpContext) {
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			stream, _ := ctx.GetContext(ctxKeyIsStreaming).(bool)
			if err := util.OverwriteRequestPath(gp.getRequestPath(ApiNameChatCompletion, model, stream)); err != nil {
				log.Errorf("geminiProvider: overwrite request path failed: %v", err)
			}
		}
		return p, ""

	case c.typ == providerTypeAzure:
		ap, ok := prov.(*azureProvider)
		if !ok {
			return nil, "unexpected azure provider instance type"
		}
		if !inDefaultApis {
			return nil, "this apiName is not covered by streaming"
		}
		// buffered TransformRequestBody: after the default transform, :path is rewritten from the final model in the context.
		// Without a deployment name in serviceUrl (DomainOnly / OpenAI v1 base) the path holds a {model} placeholder and model
		// must be known before the headers are released; the other two forms have paths independent of the body.
		p := defaultPlan(nil)
		p.RequireModelBeforeCommit = !azureModelIrrelevantApis[apiName] &&
			(ap.serviceUrlType == azureServiceUrlTypeDomainOnly || ap.serviceUrlType == azureServiceUrlTypeOpenAIV1Base)
		p.AfterPrelude = func(ctx wrapper.HttpContext) {
			if path := ap.transformRequestPath(ctx, apiName); path != "" {
				if err := util.OverwriteRequestPath(path); err != nil {
					log.Errorf("azureProvider: overwrite request path to %s failed: %v", path, err)
				}
			}
		}
		return p, ""

	case c.typ == providerTypeOpenAI || c.typ == providerTypeLongcat || c.typ == providerTypeDoubao || streamDefaultProviders[c.typ]:
		if (c.typ == providerTypeOpenAI || c.typ == providerTypeLongcat) && c.responseJsonSchema != nil {
			return nil, "responseJsonSchema is re-serialized through a struct"
		}
		if !inDefaultApis {
			return nil, "the default path of this apiName is not covered by streaming"
		}
		return defaultPlan(nil), ""
	}
	return nil, "no streaming protocol implemented for provider " + c.typ
}

// chatRequestFieldTypes reproduces the type checking the buffered path gets for free.
//
// The buffered path decodes the whole body into chatCompletionRequest, so a field of the wrong type fails the
// request before anything reaches the provider. The streaming path only judges the fields its transform reads
// and passes the rest through, which on a gateway let twelve fields diverge: a wrong type in metadata, n,
// presence_penalty, seed, user, response_format, service_tier, stream_options, top_logprobs, logit_bias,
// logprobs or frequency_penalty was rejected when buffered and forwarded to the provider when streamed.
//
// The table is derived from the struct rather than written out, so adding a field to chatCompletionRequest
// cannot silently leave the streaming path more permissive than the buffered one.
var chatRequestFieldTypes = streamxform.FieldTypesOf(&chatCompletionRequest{})

// checkChatRequestTypes applies that table. It belongs only to the providers whose buffered path really does
// decode into the struct -- claude, gemini and native qwen. The rest go through defaultTransformRequestBody,
// which reads the body with gjson and type-checks nothing; adding the check there would make the streaming
// path stricter than the buffered one, which is a deviation in the other direction.
func checkChatRequestTypes(p *StreamPlan) *StreamPlan {
	if p != nil && p.Tr != nil && ChatRequestTypeCheck {
		p.Tr.SetFieldTypes(chatRequestFieldTypes)
	}
	return p
}

// ChatRequestTypeCheck turns the check off for a rollout that needs the previous, looser behaviour back.
// See streamTypeCheck in main.go for why it exists.
var ChatRequestTypeCheck = true

// StreamApplyPrelude applies the side effects of the buffered path:
//   - request header Accept: text/event-stream (only when stream is true; only effective while the headers are still held)
//   - context keys isStreaming / originalRequestModel / finalRequestModel
//
// headersMutable false means the headers have already been sent (past the commit point); only the context keys are written then.
func (c *ProviderConfig) StreamApplyPrelude(ctx wrapper.HttpContext, apiName ApiName, plan *StreamPlan, pre streamxform.Prelude, headersMutable bool) {
	if plan.ApplyStream && pre.StreamSeen {
		if pre.Stream && !plan.NoAcceptHeader {
			if headersMutable {
				_ = proxywasm.ReplaceHttpRequestHeader("Accept", "text/event-stream")
			} else {
				log.Warnf("[stream-xform] stream=true appeared after the commit point, Accept header not rewritten")
			}
		}
		ctx.SetContext(ctxKeyIsStreaming, pre.Stream)
	}
	if plan.ApplyModel && pre.ModelSeen {
		ctx.SetContext(ctxKeyOriginalRequestModel, pre.Model)
		ctx.SetContext(ctxKeyFinalRequestModel, getMappedModel(pre.Model, c.modelMapping))
	}
}

// StreamFinalizeContext fills in the defaults for fields never seen once the whole body is scanned, as the buffered path does:
// chat requests always get isStreaming written (false by default).
func (c *ProviderConfig) StreamFinalizeContext(ctx wrapper.HttpContext, apiName ApiName, plan *StreamPlan, pre streamxform.Prelude) {
	if plan.ApplyStream && !pre.StreamSeen {
		ctx.SetContext(ctxKeyIsStreaming, false)
	}
}
