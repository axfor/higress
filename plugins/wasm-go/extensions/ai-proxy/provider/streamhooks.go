package provider

import (
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
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
	Tr streamxform.Xform // a Transformer, or a Pipeline of two
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
	AfterPrelude func(ctx wrapper.HttpContext, pre streamxform.Prelude)
	// OnFinish is called after the whole body has been scanned: context keys that are only known at the end and only used on the response side.
	OnFinish func(ctx wrapper.HttpContext)
	// CommitGate, when set, is asked before the headers are released: false while a fact the headers depend on
	// (an image key deciding Kling's path) can still turn up later in the body; that is a fallback unless the
	// whole body has been seen.
	CommitGate func(pre streamxform.Prelude, last bool) bool
	// Replan picks the real transformer once model is known, for providers whose wire format depends on the mapped
	// model (Vertex: a claude-prefixed model takes the Anthropic format, anything else the Gemini one). Tr is then only
	// a probe for model. See guard.Plan.Replan.
	Replan func(pre streamxform.Prelude) (streamxform.Xform, string)
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
	plan, why = c.newStreamPlan(ctx, apiName, prov)
	if plan == nil {
		return nil, why
	}
	plan, why = c.withFirstByteTimeout(plan, apiName)
	if plan == nil {
		return nil, why
	}
	return c.withCustomSettings(plan)
}

// withCustomSettings reproduces main.go's ReplaceByCustomSettings, which rewrites the body before any handler
// sees it: the settings stage goes in front of whatever the plan transforms with, as the first stage of a
// pipeline, and stands alone where the buffered path would otherwise forward the body untouched. A replanned
// transformer gets its own fresh stage, since the replay starts from the first byte. Settings whose path uses
// gjson syntax beyond plain dotted keys keep the buffered path.
func (c *ProviderConfig) withCustomSettings(p *StreamPlan) (*StreamPlan, string) {
	if len(c.customSettings) == 0 {
		return p, ""
	}
	settings, why := c.streamSettings()
	if why != "" {
		return nil, why
	}
	if p.Passthrough {
		p.Passthrough = false
		p.Tr = streamxform.NewSettings(settings)
		return p, ""
	}
	if p.Tr != nil {
		p.Tr = streamxform.NewPipeline(streamxform.NewSettings(settings), p.Tr)
	}
	if p.Replan != nil {
		inner := p.Replan
		p.Replan = func(pre streamxform.Prelude) (streamxform.Xform, string) {
			tr, why := inner(pre)
			if tr == nil {
				return nil, why
			}
			return streamxform.NewPipeline(streamxform.NewSettings(settings), tr), ""
		}
	}
	return p, ""
}

// streamSettings converts the configured customSettings (names already adjusted to the protocol) into the
// stage's form. sjson path syntax other than dotted keys -- escapes, wildcards, array indexes, modifiers --
// is declared unsupported rather than reproduced.
func (c *ProviderConfig) streamSettings() ([]streamxform.Setting, string) {
	out := make([]streamxform.Setting, 0, len(c.customSettings))
	for _, s := range c.customSettings {
		if strings.ContainsAny(s.name, `\*?#|@!`) {
			return nil, "customSettings path uses gjson syntax the streaming stage does not reproduce"
		}
		parts := strings.Split(s.name, ".")
		for _, part := range parts {
			if part == "" || isArrayIndex(part) {
				return nil, "customSettings path with an array index or an empty component keeps the buffered path"
			}
		}
		out = append(out, streamxform.Setting{Path: parts, Value: []byte(s.value), Overwrite: s.overwrite})
	}
	return out, ""
}

// isArrayIndex reports whether a path component means an array position to sjson: digits, or -1 for append.
func isArrayIndex(part string) bool {
	if part == "-1" {
		return true
	}
	for _, r := range part {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// withFirstByteTimeout reproduces handleRequestBody's first-byte timeout header: set when the request streams
// (isStreamingAPI reads stream from the body). For chat the plan already observes stream, so the header is set
// at the commit point and stream has to be known by then; the native Gemini streaming endpoint always streams;
// the other endpoint kinds that read stream keep the buffered path when the setting is on.
func (c *ProviderConfig) withFirstByteTimeout(p *StreamPlan, apiName ApiName) (*StreamPlan, string) {
	if c.firstByteTimeout == 0 {
		return p, ""
	}
	set := func() {
		_ = proxywasm.ReplaceHttpRequestHeader("x-envoy-upstream-rq-first-byte-timeout-ms", strconv.FormatUint(uint64(c.firstByteTimeout), 10))
	}
	prev := p.AfterPrelude
	switch {
	case apiName == ApiNameGeminiStreamGenerateContent:
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			if prev != nil {
				prev(ctx, pre)
			}
			set()
		}
		return p, ""
	case apiName == ApiNameChatCompletion && p.Tr != nil && !p.Passthrough:
		p.RequireStreamBeforeCommit = true
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			if prev != nil {
				prev(ctx, pre)
			}
			if pre.Stream {
				set()
			}
		}
		return p, ""
	case c.isStreamingAPI(apiName, nil) || apiName == ApiNameCompletion || apiName == ApiNameImageGeneration || apiName == ApiNameImageEdit ||
		apiName == ApiNameResponses || apiName == ApiNameQwenAsyncAIGC || apiName == ApiNameAnthropicMessages || apiName == ApiNameAnthropicComplete:
		return nil, "firstByteTimeout reads stream on an endpoint whose plan does not observe it"
	}
	return p, "" // endpoints that never stream: the header is never set
}

func (c *ProviderConfig) newStreamPlan(ctx wrapper.HttpContext, apiName ApiName, prov Provider) (plan *StreamPlan, why string) {
	if c.typ == providerTypeGeneric {
		// generic's OnRequestBody writes the body back unchanged without handleRequestBody:
		// only the two settings in main.go that apply to every provider touch the body / context
		if c.IsRetryOnFailureEnabled() {
			return nil, "retryOnFailure needs the whole body stored in the context"
		}
		return &StreamPlan{Passthrough: true}, ""
	}
	if c.typ == providerTypeVertex && apiName == ApiNameVertexRaw {
		// Checked before IsOriginal on the buffered path too: the raw endpoints are normally used with the original protocol.
		return c.vertexRawPlan(ctx, prov)
	}
	if c.IsOriginal() {
		// handleRequestBody returns before it touches the body under the original protocol, and so do the providers with
		// their own OnRequestBody -- except the two that sign the body, which need all of it.
		if c.typ == providerTypeHunyuan || (c.typ == providerTypeBedrock && len(c.apiTokens) == 0) {
			return nil, "original protocol with a signature over the body"
		}
		if c.typ == providerTypeMinimax && c.minimaxApiType == minimaxApiTypePro {
			return nil, "minimax Pro rebuilds the body under the original protocol too"
		}
		if !c.isSupportedAPI(apiName) {
			return nil, "apiName not supported"
		}
		return &StreamPlan{Passthrough: true}, ""
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
		return c.claudeInputPlan(ctx, apiName, prov)
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

	case c.typ == providerTypeQwen && !c.qwenEnableCompatible && apiName == ApiNameEmbeddings:
		// onEmbeddingsRequestBody: parseRequestAndMapModel and buildQwenTextEmbeddingRequest; the path is header-phase
		// work, and re-applying providerBasePath to it in the body phase changes nothing (applyProviderBasePath is idempotent).
		p := &StreamPlan{Tr: streamxform.NewQwenEmbeddings(streamxform.EmbeddingsOptions{MapModel: c.mapStrict()}), ApplyModel: true, RequireModelBeforeCommit: true}
		if ChatRequestTypeCheck {
			p.Tr.SetFieldTree(embeddingsFieldTree)
		}
		return p, ""

	case c.typ == providerTypeQwen && !c.qwenEnableCompatible:
		// native DashScope protocol: the buffered onChatCompletionRequestBody changes the path and headers by model / stream in the body phase
		if !isChat {
			return nil, "native qwen protocol streams chat completion only"
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
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			// reproduce the header / path handling of onChatCompletionRequestBody
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			if strings.HasPrefix(model, qwenVlModelPrefixName) {
				_ = util.OverwriteRequestPath(c.bodyPhasePath(qwenMultimodalGenerationPath))
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
		// providerBasePath: the path is header-phase work and its body-phase re-application is idempotent
		if !inDefaultApis {
			return nil, "this apiName is not covered by streaming"
		}
		opts := defaultOpts(&streamxform.QwenVariant{SupportsPreserveThinking: qwenSupportsPreserveThinking})
		opts.ModelOnlyIfPresent = true
		opts.DetectStream = c.firstByteTimeout != 0 // no Accept / isStreaming on this branch; stream is only observed for the first-byte timeout
		return &StreamPlan{Tr: streamxform.NewOpenAI(opts), ApplyStream: false, ApplyModel: false}, ""

	case c.typ == providerTypeMinimax && c.minimaxApiType == minimaxApiTypePro && isChat:
		// handleRequestBodyByChatCompletionPro: the request is rebuilt (system → bot_setting, user / assistant →
		// sender messages, other roles dropped), model mapped leniently, the path gets the GroupId in the body
		// phase. Neither Accept nor the model context keys are written there.
		p := checkChatRequestTypes(&StreamPlan{Tr: streamxform.NewMiniMaxPro(streamxform.MiniMaxProOptions{
			MapModel: mapLenient, DefaultBotName: defaultBotName, DefaultSenderName: defaultSenderName,
			DefaultBotSettingContent: defaultBotSettingContent, SenderTypeBot: senderTypeBot, SenderTypeUser: senderTypeUser,
		})})
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			if err := util.OverwriteRequestPath(fmt.Sprintf("%s?GroupId=%s", minimaxChatCompletionProPath, c.minimaxGroupId)); err != nil {
				log.Errorf("minimaxProvider: overwrite request path failed: %v", err)
			}
		}
		return p, ""

	case c.typ == providerTypeDify && isChat:
		// difyChatGenRequest: every message's text under a role heading, into query or inputs by bot type;
		// parseRequestAndMapModel's Accept is overwritten by the header snapshot, isStreaming and the model keys stay.
		conv, _ := proxywasm.GetHttpRequestHeader("ConversationId")
		return checkChatRequestTypes(&StreamPlan{Tr: streamxform.NewDify(streamxform.DifyOptions{
			MapModel: c.mapStrict(), BotType: c.botType, InputVariable: c.inputVariable, ConversationId: conv,
		}), ApplyStream: true, ApplyModel: true, NoAcceptHeader: true, RequireModelBeforeCommit: true}), ""

	case c.typ == providerTypeTriton && isChat:
		// BuildTritonTexGenRequest keeps the last message's id and text; path and host are set from model and stream
		// in the body phase.
		tp, ok := prov.(*tritonProvider)
		if !ok {
			return nil, "unexpected triton provider instance type"
		}
		p := checkChatRequestTypes(&StreamPlan{Tr: streamxform.NewTriton(streamxform.TritonOptions{MapModel: c.mapStrict()}),
			ApplyStream: true, ApplyModel: true, NoAcceptHeader: true, RequireModelBeforeCommit: true, RequireStreamBeforeCommit: true})
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			if err := util.OverwriteRequestPath(c.bodyPhasePath(tp.getFinalRequestPath(ctx, &chatCompletionRequest{Model: model}, pre.Stream))); err != nil {
				log.Errorf("tritonProvider: overwrite request path failed: %v", err)
			}
			_ = proxywasm.ReplaceHttpRequestHeader(util.HeaderAuthority, c.tritonDomain)
		}
		return p, ""

	case c.typ == providerTypeMinimax:
		// V2 endpoint (default): the buffered handleRequestBodyByChatCompletionV2 only changes model and pins the path to chatcompletion_v2;
		// the Pro endpoint has a request structure of its own and does not stream.
		if c.minimaxApiType == minimaxApiTypePro {
			return nil, "the minimax Pro endpoint is not covered by streaming"
		}
		if !isChat {
			return nil, "minimax streams chat completion only"
		}
		opts := defaultOpts(nil)
		opts.DetectStream = c.firstByteTimeout != 0 // neither Accept / isStreaming nor the model context keys on this branch
		p := &StreamPlan{Tr: streamxform.NewOpenAI(opts), ApplyStream: false, ApplyModel: false}
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			// the buffered path only switches to the v2 endpoint in the body phase (not in the header phase); the path is fixed and independent of body fields
			if err := util.OverwriteRequestPath(c.bodyPhasePath(minimaxChatCompletionV2Path)); err != nil {
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

	case c.typ == providerTypeGemini && (apiName == ApiNameEmbeddings || apiName == ApiNameImageGeneration):
		// onEmbeddingsRequestBody / onImageGenerationRequestBody: parseRequestAndMapModel, the path from the mapped model,
		// and a small rebuilt body. The Accept header is snapshot-overwritten as for chat.
		gp, ok := prov.(*geminiProvider)
		if !ok {
			return nil, "unexpected gemini provider instance type"
		}
		p := &StreamPlan{ApplyModel: true, RequireModelBeforeCommit: true, NoAcceptHeader: true}
		if apiName == ApiNameEmbeddings {
			p.Tr = streamxform.NewGeminiEmbeddings(streamxform.EmbeddingsOptions{MapModel: c.mapStrict()})
			if ChatRequestTypeCheck {
				p.Tr.SetFieldTree(embeddingsFieldTree)
			}
		} else {
			p.Tr = streamxform.NewGeminiImage(streamxform.EmbeddingsOptions{MapModel: c.mapStrict()})
			if ChatRequestTypeCheck {
				p.Tr.SetFieldTree(imageGenerationFieldTree)
			}
		}
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			if err := util.OverwriteRequestPath(gp.getRequestPath(apiName, model, false)); err != nil {
				log.Errorf("geminiProvider: overwrite request path failed: %v", err)
			}
		}
		return p, ""

	case c.typ == providerTypeGemini:
		gp, ok := prov.(*geminiProvider)
		if !ok {
			return nil, "unexpected gemini provider instance type"
		}
		if apiName == ApiNameGeminiGenerateContent || apiName == ApiNameGeminiStreamGenerateContent {
			// The buffered TransformRequestBodyHeaders hands these bodies back untouched; host and key are header-phase work.
			return &StreamPlan{Passthrough: true}, ""
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
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			stream, _ := ctx.GetContext(ctxKeyIsStreaming).(bool)
			if err := util.OverwriteRequestPath(gp.getRequestPath(ApiNameChatCompletion, model, stream)); err != nil {
				log.Errorf("geminiProvider: overwrite request path failed: %v", err)
			}
		}
		return p, ""

	case c.typ == providerTypeVertex && apiName == ApiNameEmbeddings && !c.vertexOpenAICompatible:
		// onEmbeddingsRequestBody: parseRequestAndMapModel, the predict path from the mapped model, instances from input.
		vp, ok := prov.(*vertexProvider)
		if !ok {
			return nil, "unexpected vertex provider instance type"
		}
		auth, why := c.vertexAuth(vp)
		if why != "" {
			return nil, why
		}
		p := &StreamPlan{Tr: streamxform.NewVertexEmbeddings(streamxform.EmbeddingsOptions{MapModel: c.mapStrict()}), ApplyModel: true, RequireModelBeforeCommit: true, NoAcceptHeader: true}
		if ChatRequestTypeCheck {
			p.Tr.SetFieldTree(embeddingsFieldTree)
		}
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			if err := util.OverwriteRequestPath(vp.getRequestPath(ctx, ApiNameEmbeddings, model, false)); err != nil {
				log.Errorf("vertexProvider: overwrite request path failed: %v", err)
			}
			auth()
		}
		return p, ""

	case c.typ == providerTypeVertex && (apiName == ApiNameImageGeneration || apiName == ApiNameImageEdit || apiName == ApiNameImageVariation) && !c.vertexOpenAICompatible:
		// onImageGenerationRequestBody / onImageEditRequestBody / onImageVariationRequestBody: parseRequestAndMapModel,
		// the generateContent path from the mapped model, one user turn of the image parts and the prompt. The image
		// inputs stream (a data URL's payload goes out as it arrives); a multipart body never gets here.
		vp, ok := prov.(*vertexProvider)
		if !ok {
			return nil, "unexpected vertex provider instance type"
		}
		auth, why := c.vertexAuth(vp)
		if why != "" {
			return nil, why
		}
		kind, tree := streamxform.VertexImageGeneration, imageGenerationFieldTree
		switch apiName {
		case ApiNameImageEdit:
			kind, tree = streamxform.VertexImageEdit, imageEditFieldTree
		case ApiNameImageVariation:
			kind, tree = streamxform.VertexImageVariation, imageVariationFieldTree
		}
		var ss []streamxform.GeminiSafetySetting
		for k, v := range c.geminiSafetySetting {
			ss = append(ss, streamxform.GeminiSafetySetting{Category: k, Threshold: v})
		}
		sort.Slice(ss, func(i, j int) bool { return ss[i].Category < ss[j].Category })
		p := &StreamPlan{Tr: streamxform.NewVertexImage(streamxform.VertexImageOptions{
			Kind: kind, MapModel: c.mapStrict(), SafetySettings: ss, DetectMime: detectMimeTypeFromURL, ParseSize: vp.parseImageSize,
		}), ApplyModel: true, RequireModelBeforeCommit: true, NoAcceptHeader: true}
		if ChatRequestTypeCheck {
			p.Tr.SetFieldTree(tree)
		}
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			if err := util.OverwriteRequestPath(vp.getRequestPath(ctx, apiName, model, false)); err != nil {
				log.Errorf("vertexProvider: overwrite request path failed: %v", err)
			}
			if apiName != ApiNameImageGeneration { // the buffered edit / variation handlers pin the content type
				_ = proxywasm.ReplaceHttpRequestHeader("Content-Type", util.MimeTypeApplicationJson)
			}
			auth()
		}
		return p, ""

	case c.typ == providerTypeVertex && apiName == ApiNameAnthropicMessages:
		// /v1/messages goes to :rawPredict / :streamRawPredict with the Anthropic body kept: model is read for the path
		// and dropped from the body, anthropic_version and a default max_tokens are added, context_management removed.
		// The buffered path does not set Accept or isStreaming here; stream is read only for the path.
		vp, ok := prov.(*vertexProvider)
		if !ok {
			return nil, "unexpected vertex provider instance type"
		}
		auth, why := c.vertexAuth(vp)
		if why != "" {
			return nil, why
		}
		p := &StreamPlan{Tr: streamxform.NewOpenAI(streamxform.OpenAIOptions{
			MapModel: mapLenient, DetectStream: true, OmitModel: true, DeveloperRoleSupported: true,
			Variant: &streamxform.VertexAnthropicVariant{Version: vertexAnthropicVersion, DefaultMaxTokens: claudeDefaultMaxTokens},
		}), ApplyModel: true, RequireModelBeforeCommit: true, RequireStreamBeforeCommit: true}
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			if err := util.OverwriteRequestPath(vp.getAhthropicRequestPath(ctx, ApiNameAnthropicMessages, model, pre.Stream)); err != nil {
				log.Errorf("vertexProvider: overwrite request path failed: %v", err)
			}
			auth()
		}
		return p, ""

	case c.typ == providerTypeVertex && c.vertexOpenAICompatible && isChat:
		// OpenAI-compatible endpoint: the body stays OpenAI, model is mapped, the path is fixed. parseRequestAndMapModel
		// decodes into the request struct (type check) and sets Accept from stream on the live headers, which the
		// header snapshot then overwrites (NoAcceptHeader); isStreaming is set.
		vp, ok := prov.(*vertexProvider)
		if !ok {
			return nil, "unexpected vertex provider instance type"
		}
		auth, why := c.vertexAuth(vp)
		if why != "" {
			return nil, why
		}
		opts := defaultOpts(nil)
		opts.NormalizeUsage = false
		opts.CheckMessages = false
		opts.DeveloperRoleSupported = true
		p := checkChatRequestTypes(&StreamPlan{Tr: streamxform.NewOpenAI(opts), ApplyStream: true, ApplyModel: true, NoAcceptHeader: true, RequireModelBeforeCommit: true})
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			ctx.SetContext(contextOpenAICompatibleMarker, true)
			if err := util.OverwriteRequestPath(vp.getOpenAICompatibleRequestPath()); err != nil {
				log.Errorf("vertexProvider: overwrite request path failed: %v", err)
			}
			auth()
		}
		return p, ""

	case c.typ == providerTypeVertex && isChat:
		// onChatCompletionRequestBody decides the wire format by the mapped model: claude-prefixed models go to the
		// Anthropic endpoint in the Claude format with model left out and anthropic_version added, everything else
		// to the Gemini endpoint in Vertex's own request shape. The decision needs the mapped model, so the plan
		// starts with a probe for model and picks the transformer when it turns up (guard replan).
		vp, ok := prov.(*vertexProvider)
		if !ok {
			return nil, "unexpected vertex provider instance type"
		}
		auth, why := c.vertexAuth(vp)
		if why != "" {
			return nil, why
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
		p := &StreamPlan{
			Tr:                        streamxform.NewKeyProbe(streamxform.KeyProbeOptions{Keys: map[string]int{"model": 4 << 10}, ModelKey: "model", Observe: true}),
			ApplyStream:               true,
			ApplyModel:                true,
			NoAcceptHeader:            true,
			RequireModelBeforeCommit:  true,
			RequireStreamBeforeCommit: true,
		}
		p.Replan = func(pre streamxform.Prelude) (streamxform.Xform, string) {
			mapped, err := mapStrict(pre.Model)
			if err != nil {
				return nil, err.Error()
			}
			if strings.HasPrefix(mapped, "claude") {
				return chatTyped(streamxform.NewClaude(streamxform.ClaudeOptions{
					MapModel: mapStrict, ClaudeCodeMode: c.claudeCodeMode, OmitModel: true, AnthropicVersion: vertexAnthropicVersion,
					KeepDeveloperRole: true, // vertex's handler does not run convertDeveloperRoleToSystem
				})), ""
			}
			var ss []streamxform.GeminiSafetySetting
			for k, v := range c.geminiSafetySetting {
				ss = append(ss, streamxform.GeminiSafetySetting{Category: k, Threshold: v})
			}
			sort.Slice(ss, func(i, j int) bool { return ss[i].Category < ss[j].Category })
			return chatTyped(streamxform.NewVertexGemini(streamxform.VertexGeminiOptions{
				MapModel:       mapStrict,
				SafetySettings: ss,
				ApplyResponseFormat: func(rf map[string]any, mapped string) (string, map[string]any, error) {
					var cfg vertexChatGenerationConfig
					if err := vp.applyResponseFormatToGenerationConfig(rf, &cfg, mapped); err != nil {
						return "", nil, err
					}
					return cfg.ResponseMimeType, cfg.ResponseSchema, nil
				},
				DetectMime: detectMimeTypeFromURL,
			})), ""
		}
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			var path string
			if strings.HasPrefix(model, "claude") {
				ctx.SetContext(contextClaudeMarker, true)
				path = vp.getAhthropicRequestPath(ctx, ApiNameChatCompletion, model, pre.Stream)
			} else {
				path = vp.getRequestPath(ctx, ApiNameChatCompletion, model, pre.Stream)
			}
			if err := util.OverwriteRequestPath(path); err != nil {
				log.Errorf("vertexProvider: overwrite request path failed: %v", err)
			}
			auth()
		}
		return p, ""

	case c.typ == providerTypeDeepl && isChat:
		// deeplTextGenRequest: non-system messages become text entries, the system message context, target_lang from the
		// setting. model is not mapped -- it selects the host and must be Free or Pro -- and only finalRequestModel is written.
		p := checkChatRequestTypes(&StreamPlan{Tr: streamxform.NewDeepL(streamxform.DeepLOptions{TargetLang: c.targetLang}), RequireModelBeforeCommit: true})
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			ctx.SetContext(ctxKeyFinalRequestModel, pre.Model)
			host := deeplHostFree
			if pre.Model == "Pro" {
				host = deeplHostPro
			}
			_ = proxywasm.ReplaceHttpRequestHeader(util.HeaderAuthority, host)
		}
		return p, ""

	case c.typ == providerTypeKling && apiName == ApiNameVideos:
		// transformOpenAIVideoRequest: the body passes through, model (or model_name) is mapped into model_name and model
		// removed; the path depends on whether an image input key is present, which only the whole body can deny.
		kp, ok := prov.(*klingOpenAIProvider)
		if !ok {
			return nil, "unexpected kling provider instance type"
		}
		tr := streamxform.NewKling(streamxform.KlingOptions{MapModel: mapLenient})
		kproto := tr.Protocol().(interface{ ImageToVideo() bool })
		p := &StreamPlan{Tr: tr, ApplyModel: true,
			CommitGate: func(pre streamxform.Prelude, last bool) bool { return last || kproto.ImageToVideo() }}
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			taskType, target := klingTaskTypeTextToVideo, kp.textCreateVideoPath()
			if kproto.ImageToVideo() {
				taskType, target = klingTaskTypeImageToVideo, kp.imageCreateVideoPath()
			}
			ctx.SetContext(ctxKeyKlingVideoTaskType, taskType)
			cur, _ := proxywasm.GetHttpRequestHeader(util.HeaderPath)
			if err := util.OverwriteRequestPath(c.bodyPhasePath(klingPathWithOriginalQuery(ctx, cur, target))); err != nil {
				log.Errorf("klingProvider: overwrite request path failed: %v", err)
			}
		}
		return p, ""

	case c.typ == providerTypeCohere && isChat:
		// buildCohereRequest rebuilds the request from the first message's text and a handful of scalars; the
		// Accept header set by parseRequestAndMapModel stays, cohere transforms the body without a header snapshot.
		return checkChatRequestTypes(&StreamPlan{Tr: streamxform.NewCohere(streamxform.CohereOptions{MapModel: c.mapStrict()}),
			ApplyStream: true, ApplyModel: true, RequireModelBeforeCommit: true}), ""

	case c.typ == providerTypeBedrock && isChat && len(c.apiTokens) > 0:
		// Converse with API tokens: no SigV4 over the body. onChatCompletionRequestBody sets the path from model and
		// stream, pins Accept to */* on the header snapshot and rebuilds the request (buildBedrockTextGenerationRequest).
		bp, ok := prov.(*bedrockProvider)
		if !ok {
			return nil, "unexpected bedrock provider instance type"
		}
		p := checkChatRequestTypes(&StreamPlan{Tr: streamxform.NewBedrock(streamxform.BedrockOptions{
			MapModel:             c.mapStrict(),
			AdditionalFields:     c.bedrockAdditionalFields,
			PromptCacheRetention: c.promptCacheRetention,
			PromptCacheSupported: isPromptCacheSupportedModel,
		}), ApplyStream: true, ApplyModel: true, NoAcceptHeader: true, RequireModelBeforeCommit: true, RequireStreamBeforeCommit: true})
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
			model := ctx.GetStringContext(ctxKeyFinalRequestModel, "")
			format := bedrockChatCompletionPath
			if pre.Stream {
				format = bedrockStreamChatCompletionPath
			}
			hdr := http.Header{}
			bp.overwriteRequestPathHeader(hdr, format, model)
			if err := util.OverwriteRequestPath(c.bodyPhasePath(hdr.Get(util.HeaderPath))); err != nil {
				log.Errorf("bedrockProvider: overwrite request path failed: %v", err)
			}
			_ = proxywasm.ReplaceHttpRequestHeader("Accept", "*/*")
		}
		return p, ""

	case c.typ == providerTypeBedrock && apiName == ApiNameAnthropicMessages && len(c.apiTokens) > 0:
		// Mantle keeps the Anthropic body: the buffered onAnthropicMessagesRequestBody reads stream for the Accept header
		// and maps model, nothing else. With API tokens there is no SigV4 over the body; with AK/SK there is, so no plan.
		// mapModel fails on a missing model where the default mapping would write an empty one: requiring model before
		// the commit point sends such a body to the buffered path, which fails it the same way.
		opts := defaultOpts(nil)
		opts.DetectStream = true
		return &StreamPlan{Tr: streamxform.NewOpenAI(opts), ApplyStream: true, ApplyModel: true, RequireModelBeforeCommit: true}, ""

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
		p.AfterPrelude = func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
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
// chatRequestFieldTypes is the flat root-level table. The tree below carries the same root types, so the
// plugin only sets the tree; this stays because the differential tests measure what each level catches.
var chatRequestFieldTypes = streamxform.FieldTypesOf(&chatCompletionRequest{})

// chatRequestFieldTree is the recursive form. The root-level table catches a field of the wrong type; the tree
// also catches a value of the wrong type inside a field whose own type is right ({"metadata":{"k":[1,2]}}
// against map[string]string). On the corpus this struct really sees, the root table reproduces 16.8% of what
// unmarshalling rejects and the tree reproduces all of it.
//
// Depth 6 covers chatCompletionRequest down to functionCall, which is as deep as it goes.
var chatRequestFieldTree = streamxform.FieldTreeOf(&chatCompletionRequest{}, 6)

// claudeRequestFieldTree is what the automatic Claude → OpenAI conversion decodes into.
var claudeRequestFieldTree = streamxform.FieldTreeOf(&claudeTextGenRequest{}, 6)

// The non-chat endpoints decode into their own structs; their trees serve the same purpose.
var embeddingsFieldTree = streamxform.FieldTreeOf(&embeddingsRequest{}, 3)
var imageGenerationFieldTree = streamxform.FieldTreeOf(&imageGenerationRequest{}, 3)
var imageEditFieldTree = streamxform.FieldTreeOf(&imageEditRequest{}, 3)
var imageVariationFieldTree = streamxform.FieldTreeOf(&imageVariationRequest{}, 3)

// checkChatRequestTypes applies that table. It belongs only to the providers whose buffered path really does
// decode into the struct -- claude, gemini and native qwen. The rest go through defaultTransformRequestBody,
// which reads the body with gjson and type-checks nothing; adding the check there would make the streaming
// path stricter than the buffered one, which is a deviation in the other direction.
// chatTyped is checkChatRequestTypes for a bare transformer (a replan result).
func chatTyped(tr *streamxform.Transformer) *streamxform.Transformer {
	if tr != nil && ChatRequestTypeCheck {
		tr.SetFieldTree(chatRequestFieldTree)
	}
	return tr
}

func checkChatRequestTypes(p *StreamPlan) *StreamPlan {
	if p != nil && p.Tr != nil && ChatRequestTypeCheck {
		p.Tr.SetFieldTree(chatRequestFieldTree) // the tree carries the root fields' own types as well
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

// claudeInputPlan streams the automatic Claude → OpenAI protocol conversion: handleRequestBody converts the body
// with ConvertClaudeRequestToOpenAIWithOptions, strips the Claude-internal fields again for every provider but
// bedrock and claude, and hands the result to the provider's own transform. That is two transforms in series, a
// Pipeline: the conversion first, the provider's default OpenAI transformer second. Only the OpenAI-compatible
// family that goes through defaultTransformRequestBody is covered; the providers whose variants read the
// thinking context keys the buffered path sets from the Claude body keep the buffered path.
func (c *ProviderConfig) claudeInputPlan(ctx wrapper.HttpContext, apiName ApiName, prov Provider) (*StreamPlan, string) {
	if apiName != ApiNameChatCompletion {
		return nil, "automatic conversion of Claude protocol input on an endpoint other than chat"
	}
	if !(c.typ == providerTypeOpenAI || c.typ == providerTypeLongcat || c.typ == providerTypeDoubao || streamDefaultProviders[c.typ]) {
		return nil, "automatic conversion of Claude protocol input for a provider with its own conversion"
	}
	if (c.typ == providerTypeOpenAI || c.typ == providerTypeLongcat) && c.responseJsonSchema != nil {
		return nil, "responseJsonSchema is re-serialized through a struct"
	}
	first := streamxform.NewClaudeToOpenAI(streamxform.ClaudeToOpenAIOptions{
		PreserveReasoning:       c.supportsMessageReasoningContent(),
		DisableStreamUsageStats: c.disableStreamUsageStats,
	})
	if ChatRequestTypeCheck {
		first.SetFieldTree(claudeRequestFieldTree)
	}
	second := streamxform.NewOpenAI(streamxform.OpenAIOptions{
		MapModel:               func(m string) string { return getMappedModel(m, c.modelMapping) },
		DetectStream:           true,
		NormalizeUsage:         c.IsOpenAIProtocol() && !c.disableStreamUsageStats,
		DeveloperRoleSupported: isDeveloperRoleSupported(c.typ),
		CheckMessages:          true,
	})
	return &StreamPlan{Tr: streamxform.NewPipeline(first, second), ApplyStream: true, ApplyModel: true}, ""
}

// bodyPhasePath reproduces what handleRequestBody does to a path a TransformRequestBodyHeaders handler set: the
// configured providerBasePath is applied to it again, as the header phase applied it to the original path.
func (c *ProviderConfig) bodyPhasePath(path string) string {
	if c.providerBasePath != "" {
		return c.applyProviderBasePath(path)
	}
	return path
}

// mapStrict reproduces mapModel for transformers that build the request from scratch: a missing model and a
// mapping to the empty string both fail, as the buffered decode does.
func (c *ProviderConfig) mapStrict() func(string) (string, error) {
	return func(m string) (string, error) {
		if m == "" {
			return "", errors.New("missing model in request")
		}
		mapped := getMappedModel(m, c.modelMapping)
		if mapped == "" {
			return "", errors.New("model becomes empty after applying the configured mapping")
		}
		return mapped, nil
	}
}

// vertexAuth returns what the buffered path does for authentication once the headers are final: Express mode
// carries the API key in the query string (added with the path) and drops the client's Authorization header;
// the standard mode needs the OAuth token, which the buffered path fetches asynchronously and caches. Until it
// is cached the request takes that path, so only the first request after a cold start is buffered.
func (c *ProviderConfig) vertexAuth(vp *vertexProvider) (apply func(), why string) {
	if vp.isExpressMode() {
		return func() { _ = proxywasm.RemoveHttpRequestHeader("Authorization") }, ""
	}
	token, err := vp.getCachedAccessToken(vp.buildTokenKey())
	if err != nil || token == "" {
		return nil, "vertex access token not cached yet, the buffered path fetches it"
	}
	return func() { _ = proxywasm.ReplaceHttpRequestHeader("Authorization", "Bearer "+token) }, ""
}

// vertexRawPlan streams the native Vertex REST endpoints, which the buffered path forwards untouched and only
// authenticates; in Express mode the key is appended to the path the client sent.
func (c *ProviderConfig) vertexRawPlan(ctx wrapper.HttpContext, prov Provider) (*StreamPlan, string) {
	vp, ok := prov.(*vertexProvider)
	if !ok {
		return nil, "unexpected vertex provider instance type"
	}
	if !c.isSupportedAPI(ApiNameVertexRaw) {
		return nil, "apiName not supported"
	}
	auth, why := c.vertexAuth(vp)
	if why != "" {
		return nil, why
	}
	ctx.SetContext(contextVertexRawMarker, true)
	return &StreamPlan{Passthrough: true, AfterPrelude: func(ctx wrapper.HttpContext, pre streamxform.Prelude) {
		if vp.isExpressMode() {
			path, _ := proxywasm.GetHttpRequestHeader(":path")
			if err := util.OverwriteRequestPath(appendOrReplaceAPIKey(path, vp.getExpressAPIKey(ctx))); err != nil {
				log.Errorf("vertexProvider: overwrite request path failed: %v", err)
			}
		}
		auth()
	}}, ""
}
