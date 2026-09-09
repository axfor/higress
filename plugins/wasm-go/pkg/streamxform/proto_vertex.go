package streamxform

import (
	"bytes"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
)

// Streaming conversion OpenAI → Vertex AI's Gemini request shape.
//
// Derived from the buffered vertex.go buildVertexChatRequest, which differs from the Gemini provider's:
//   - every message becomes a contents element; assistant maps to model, tool and system map to user, and a system
//     message is followed by a dummy model message saying "Okay"; other roles pass as they are;
//   - a message with tool_calls yields one functionCall part from the first call and nothing from its content; a
//     text part of a tool message becomes a functionResponse named after the last function call seen;
//   - image parts become fileData (http(s) URL, mime type from the URL) or inlineData (data URL); a URL of neither
//     shape is dropped;
//   - generationConfig carries temperature / topP / maxOutputTokens, a thinkingConfig derived from reasoning_effort
//     (always present, {} when unset, because the buffered struct holds it by value), and responseMimeType /
//     responseSchema derived from response_format by the provider's own rules; tools become functionDeclarations;
//     safetySettings only when something is configured (the buffered field has omitempty).
//
// A message's content has to wait for tool_calls only when the role is assistant (bounded, then fallback); for any
// other role the content streams, and tool_calls turning up afterwards on such a message falls back.

type VertexGeminiOptions struct {
	// MapModel reproduces the buffered mapModel: an error when model is empty or maps to empty.
	MapModel func(model string) (string, error)
	// SafetySettings mirrors the setting geminiSafetySetting (the buffered path iterates a map, so the order was never fixed).
	SafetySettings []GeminiSafetySetting
	// ApplyResponseFormat reproduces applyResponseFormatToGenerationConfig for the mapped model: the response mime type
	// and schema to write, or an error that fails the request. nil leaves generationConfig without them.
	ApplyResponseFormat func(rf map[string]any, mapped string) (mime string, schema map[string]any, err error)
	// DetectMime reproduces detectMimeTypeFromURL for http(s) image URLs.
	DetectMime func(url string) string
}

type vertexThinking struct {
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
	ThinkingBudget  int  `json:"thinkingBudget,omitempty"`
}

// aligned with the buffered vertexChatGenerationConfig (the fields the builder can set)
type vertexGenerationConfig struct {
	Temperature      float64        `json:"temperature,omitempty"`
	TopP             float64        `json:"topP,omitempty"`
	MaxOutputTokens  int            `json:"maxOutputTokens,omitempty"`
	ThinkingConfig   vertexThinking `json:"thinkingConfig,omitempty"` // a value, never omitted
	ResponseMimeType string         `json:"responseMimeType,omitempty"`
	ResponseSchema   map[string]any `json:"responseSchema,omitempty"`
}

type vertexFunctionCallPart struct {
	FunctionCall struct {
		Name string         `json:"name"`
		Args map[string]any `json:"args,omitempty"`
	} `json:"functionCall"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
}

type vxToolCall struct {
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
	ThoughtSignature string         `json:"thought_signature"`
	ExtraContent     map[string]any `json:"extra_content"`
}

type vxMsg struct {
	role            string
	roleSeen        bool
	roleWritten     bool
	partsWritten    bool
	finalizing      bool
	toolCallsRaw    []byte
	toolCallsSeen   bool
	toolCallsDone   bool
	toolCallsWrote  bool // a functionCall part went out: the content, whenever it comes, is ignored
	streamedContent bool
	part            gemPart
}

type vertexProto struct {
	opt VertexGeminiOptions

	model      string
	modelSeen  bool
	stream     bool
	streamSeen bool
	temp, topP float64
	maxTok     int
	effort     string
	rfRaw      []byte
	tools      ToolsHook

	messagesSeen bool
	inputMsgs    int
	lastFn       string
	pendingOkay  bool
	m            vxMsg
}

const vxURLWin = 8 << 10 // an http(s) image URL has to fit here whole: its mime type comes from the extension at its end

// NewVertexGemini builds the OpenAI → Vertex (Gemini shape) transformer.
func NewVertexGemini(opt VertexGeminiOptions) *Transformer {
	if opt.MapModel == nil {
		opt.MapModel = func(m string) (string, error) {
			if m == "" {
				return "", errors.New("missing model in request")
			}
			return m, nil
		}
	}
	if opt.DetectMime == nil {
		opt.DetectMime = func(string) string { return "" }
	}
	p := &vertexProto{opt: opt, tools: ToolsHook{ParamsKey: "parameters"}}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

func (p *vertexProto) Prelude() Prelude {
	return Prelude{Model: p.model, ModelSeen: p.modelSeen, Stream: p.stream, StreamSeen: p.streamSeen}
}

// ---- dispatch ----

func (p *vertexProto) OnKey(t *Transformer) Action {
	switch t.Depth() {
	case 1:
		switch t.Last() {
		case "model":
			return Capture(4 << 10)
		case "messages", "tools":
			return Probe()
		case "stream":
			return Capture(16)
		case "temperature", "top_p", "max_tokens":
			return Capture(64)
		case "reasoning_effort":
			return Capture(256)
		case "response_format":
			return Capture(1 << 20)
		}
		return Skip() // not read by buildVertexChatRequest
	case 3:
		m := &p.m
		switch t.Last() {
		case "role":
			return Capture(256)
		case "content":
			if !m.roleSeen {
				return Defer(roleWaitCap)
			}
			if m.toolCallsWrote {
				return Skip()
			}
			if m.role == "assistant" && !m.toolCallsSeen && !m.finalizing {
				return Defer(assistantWaitCap) // tool_calls, if any, replaces the content entirely
			}
			m.streamedContent = true
			return Probe()
		case "tool_calls":
			return Capture(assistantWaitCap)
		}
		return Skip() // name / tool_call_id / reasoning* ...: not read by the buffered path
	case 5:
		pt := &p.m.part
		if pt.dead {
			return Skip()
		}
		switch t.Last() {
		case "type":
			return Capture(256)
		case "text":
			if !pt.typSeen {
				return Defer(partWaitCap)
			}
			if pt.typ != "text" {
				return Skip()
			}
			return Probe()
		case "image_url":
			if !pt.typSeen {
				return Defer(partWaitCap)
			}
			if pt.typ != "image_url" {
				return Skip()
			}
			return Probe()
		}
		return Skip()
	case 6:
		if t.Last() == "url" {
			p.m.part.urlSeen = true
			return Prefix(vxURLWin)
		}
		return Skip()
	}
	return Bail("unexpected path: " + t.PathString())
}

func (p *vertexProto) OnElem(t *Transformer) Action {
	switch t.Depth() {
	case 2, 4:
		return Probe()
	}
	return Bail("unexpected array: " + t.PathString())
}

func (p *vertexProto) OnStart(t *Transformer, kind ValueKind) Action {
	w := t.W()
	if t.Depth() == 1 && t.Last() == "tools" { // tools → tools:[{functionDeclarations:[...]}] whenever the buffered Tools != nil
		switch kind {
		case KindNull:
			return Skip()
		case KindArray:
			w.PushArr("tools")
			w.PushObj("")
			w.PushArr("functionDeclarations")
			return Enter().Flat().Via(&p.tools)
		}
		return Bail("tools is not an array, the buffered struct decoding fails")
	}
	switch t.Depth() {
	case 1: // messages → contents (buffered make(...,0): always an array)
		if kind != KindArray {
			return Bail("messages is not an array")
		}
		return Enter().As("contents")
	case 2:
		if kind != KindObject {
			return Bail("message is not an object")
		}
		if p.pendingOkay {
			p.writeOkay(t)
		}
		p.m = vxMsg{}
		p.inputMsgs++
		return Enter() // every message is written, system included (as user)
	case 3: // content, streaming
		switch kind {
		case KindString:
			p.writeRole(t)
			p.m.partsWritten = true
			return Prefix(1) // Text has omitempty: an empty string is [{}], so peek first
		case KindArray:
			p.writeRole(t)
			p.m.partsWritten = true
			return Enter().As("parts")
		}
		return Skip() // object / scalar / null: ParseContent yields nothing → "parts":[]
	case 4:
		if kind != KindObject {
			return Skip() // ParseContent skips non-map elements
		}
		p.m.part = gemPart{}
		return Enter().Lazy()
	case 5:
		pt := &p.m.part
		switch t.Last() {
		case "text":
			if kind != KindString {
				pt.dead = true
				return Skip()
			}
			return Prefix(1)
		case "image_url":
			if kind != KindObject {
				pt.dead = true
				return Skip()
			}
			return Enter().Flat()
		}
	}
	return Bail("unexpected Probe: " + t.PathString())
}

// ---- values complete ----

func (p *vertexProto) OnValue(t *Transformer, raw []byte) {
	switch t.Depth() {
	case 1:
		p.topValue(t, raw)
	case 3:
		p.msgValue(t, raw)
	case 5:
		p.partValue(t, raw)
	}
}

func (p *vertexProto) topValue(t *Transformer, raw []byte) {
	isNull := string(raw) == "null"
	switch t.Last() {
	case "model":
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("model is not a string")
			return
		}
		p.model, p.modelSeen = s, true
	case "stream":
		switch string(raw) {
		case "true":
			p.stream = true
		case "false", "null":
		default:
			t.Bail("stream is not a boolean")
			return
		}
		p.streamSeen = !isNull
	case "temperature", "top_p":
		if isNull {
			return
		}
		f, err := strconv.ParseFloat(string(raw), 64)
		if err != nil || !isNumLiteral(raw) {
			t.Bail(t.Last() + " is not a number")
			return
		}
		if t.Last() == "temperature" {
			p.temp = f
		} else {
			p.topP = f
		}
	case "max_tokens":
		if isNull {
			return
		}
		if !isIntLiteral(raw) {
			t.Bail("max_tokens is not an integer")
			return
		}
		p.maxTok = atoi(raw)
	case "reasoning_effort":
		if isNull {
			return
		}
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("reasoning_effort is not a string")
			return
		}
		p.effort = s
	case "response_format":
		if isNull {
			return
		}
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Bail("response_format is not an object, the buffered struct decoding fails")
			return
		}
		p.rfRaw = append([]byte(nil), raw...)
	}
}

func (p *vertexProto) msgValue(t *Transformer, raw []byte) {
	m := &p.m
	switch t.Last() {
	case "role":
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("role is not a string")
			return
		}
		m.role, m.roleSeen = s, true
		if m.toolCallsSeen && !m.toolCallsDone {
			p.applyToolCalls(t)
			return
		}
		if len(t.Deferred()) > 0 {
			t.Release()
		}
	case "tool_calls":
		m.toolCallsRaw, m.toolCallsSeen = append([]byte(nil), raw...), true
		if m.streamedContent {
			var tcs []vxToolCall
			if json.Unmarshal(raw, &tcs) == nil && len(tcs) > 0 {
				t.Bail("tool_calls after the content of a non-assistant message, whose content already went out")
			}
			return
		}
		if m.roleSeen {
			p.applyToolCalls(t)
		}
	}
}

// applyToolCalls decides a message once both role and tool_calls are known: a non-empty tool_calls writes the
// functionCall part and drops whatever content was held; an empty one lets the held content through.
func (p *vertexProto) applyToolCalls(t *Transformer) {
	m := &p.m
	m.toolCallsDone = true
	var tcs []vxToolCall
	if string(m.toolCallsRaw) != "null" {
		if err := json.Unmarshal(m.toolCallsRaw, &tcs); err != nil {
			t.Bail("tool_calls failed to decode: " + err.Error())
			return
		}
	}
	if len(tcs) == 0 {
		if len(t.Deferred()) > 0 {
			t.Release()
		}
		return
	}
	t.DropDeferred()
	p.writeFunctionCall(t, tcs[0])
}

func (p *vertexProto) writeFunctionCall(t *Transformer, tc vxToolCall) {
	p.writeRole(t)
	var part vertexFunctionCallPart
	part.FunctionCall.Name = tc.Function.Name
	args := map[string]any{}
	_ = json.Unmarshal([]byte(tc.Function.Arguments), &args) // the buffered path logs the error and keeps an empty map
	part.FunctionCall.Args = args
	part.ThoughtSignature = tc.ThoughtSignature
	if part.ThoughtSignature == "" {
		if g, ok := tc.ExtraContent["google"].(map[string]any); ok {
			if s, ok := g["thought_signature"].(string); ok {
				part.ThoughtSignature = s
			}
		}
	}
	p.lastFn = tc.Function.Name
	b, _ := json.Marshal([]vertexFunctionCallPart{part})
	w := t.W()
	w.Key("parts")
	w.Raw(b)
	p.m.partsWritten = true
	p.m.toolCallsWrote = true
}

func (p *vertexProto) partValue(t *Transformer, raw []byte) {
	pt := &p.m.part
	if t.Last() != "type" {
		return
	}
	pt.typSeen = true
	s, ok := jsonUnquote(raw)
	if !ok {
		pt.dead = true
		t.DropDeferred()
		return
	}
	pt.typ = s
	switch s {
	case "text", "image_url":
		if len(t.Deferred()) > 0 {
			t.Release()
		}
	default:
		pt.dead = true // other part kinds are skipped by the buffered vertex branch
		t.DropDeferred()
	}
}

// functionResponseOpen writes the start of a functionResponse part for a tool message, named after the last
// function called: keyed inside a part object the writer manages (a text part), raw inside a literal one.
func (p *vertexProto) functionResponseOpen(w *Writer, keyed bool) {
	if keyed {
		w.Key("functionResponse")
		w.RawString(`{"name":`)
	} else {
		w.RawString(`"functionResponse":{"name":`)
	}
	w.JSONString(p.lastFn)
	w.RawString(`,"response":{`)
}

// OnPrefix: string content, text parts and the head of image_url.url.
func (p *vertexProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	w := t.W()
	isTool := p.m.role == "tool"
	switch t.Depth() {
	case 3: // string content → parts:[{text}] or, for a tool message, [{functionResponse}]
		w.Key("parts")
		if isTool {
			w.RawString("[{")
			p.functionResponseOpen(w, false)
			if complete && len(raw) == 0 {
				w.RawString("}}}]") // Output has omitempty
				return Skip(), 0
			}
			w.RawString(`"output":"`)
			return Pass().Wrap(nil, []byte(`"}}}]`)), 0
		}
		if complete && len(raw) == 0 {
			w.RawString(`[{}]`)
			return Skip(), 0
		}
		return Pass().Wrap(lit5, lit6), 0
	case 5: // text part
		if isTool {
			p.functionResponseOpen(w, true)
			if complete && len(raw) == 0 {
				w.RawString("}}")
				return Skip(), 0
			}
			w.RawString(`"output":"`)
			return Pass().Wrap(nil, []byte(`"}}`)), 0
		}
		if complete && len(raw) == 0 {
			w.Open() // {}
			return Skip(), 0
		}
		w.KeyRaw(t.KeyRaw())
		return Pass().Wrap(lit0, lit0), 0
	}
	// image_url.url: convertMediaContent
	dec, off := unescapePrefix(raw)
	if bytes.HasPrefix(dec, []byte("http")) { // strings.HasPrefix(mediaUrl, "http"), case-sensitive as on the buffered path
		if !complete {
			return Bail("http image URL exceeds the prefix window, the mime type needs its end"), 0
		}
		u := string(dec)
		w.Key("fileData")
		w.RawString(`{"mimeType":`)
		w.JSONString(p.opt.DetectMime(u))
		w.RawString(`,"fileUri":`)
		w.JSONString(u)
		w.Byte('}')
		return Skip(), 0
	}
	// data:<mime>;base64,<data>; anything else is an error on the buffered path and the part is dropped
	if !bytes.HasPrefix(dec, lit1) {
		if !complete && len(dec) < len(lit1) {
			return Bail("image URL cut at the window boundary"), 0
		}
		return Skip(), 0
	}
	semi := bytes.IndexByte(dec, ';')
	if semi < 0 {
		if !complete {
			return Bail("data URL header exceeds the prefix window"), 0
		}
		return Skip(), 0
	}
	mime := string(dec[5:semi])
	rest := dec[semi+1:]
	if !complete && len(rest) < len(lit4) {
		return Bail("data URL header cut at the window boundary"), 0
	}
	if !bytes.HasPrefix(rest, lit4) || len(strings.Split(mime, "/")) < 2 {
		return Skip(), 0
	}
	resume := off[semi+1+len(lit4)]
	if !complete && resume >= len(raw) {
		return Bail("data URL payload cut at the window boundary"), 0
	}
	w.Key("inlineData")
	w.RawString(`{"mimeType":`)
	w.JSONString(mime)
	w.RawString(`,"data":"`)
	return Pass().Wrap(nil, lit3), resume
}

// ---- containers closing ----

func (p *vertexProto) OnLeave(t *Transformer) {
	w := t.W()
	if t.Depth() == 1 && t.Last() == "tools" {
		w.Open() // an empty tools array is materialized as [{"functionDeclarations":[]}] as well
		w.Pop()
		w.Pop()
		w.Pop()
		return
	}
	switch t.Depth() {
	case 1:
		p.messagesSeen = true
		if p.inputMsgs == 0 {
			t.Bail("no message found in the request body")
			return
		}
		if p.pendingOkay {
			p.writeOkay(t)
		}
	case 2:
		p.finishMessage(t)
	case 3: // parts array: always materialized
		w.Open()
	case 4:
		if p.m.part.dead || !p.m.part.typSeen {
			t.DropDeferred()
		}
	}
}

func (p *vertexProto) writeOkay(t *Transformer) {
	w := t.W()
	w.Elem()
	w.RawString(`{"role":"model","parts":[{"text":"Okay"}]}`)
	p.pendingOkay = false
}

func (p *vertexProto) writeRole(t *Transformer) {
	m := &p.m
	if m.roleWritten {
		return
	}
	m.roleWritten = true
	w := t.W()
	w.Open()
	role := m.role
	switch role {
	case "assistant":
		role = "model"
	case "tool", "system":
		role = "user"
	}
	if role != "" {
		w.Key("role")
		w.JSONString(role)
	}
}

func (p *vertexProto) finishMessage(t *Transformer) {
	m := &p.m
	w := t.W()
	if !m.roleSeen {
		m.role, m.roleSeen = "", true
	}
	m.finalizing = true
	if m.toolCallsSeen && !m.toolCallsDone {
		p.applyToolCalls(t)
		if t.Dead() {
			return
		}
	}
	if len(t.Deferred()) > 0 {
		t.ReleaseNow()
		if t.Dead() {
			return
		}
	}
	p.writeRole(t)
	if !m.partsWritten {
		w.Key("parts")
		w.RawString("[]")
	}
	if m.role == "system" {
		p.pendingOkay = true // the dummy model message follows this element
	}
}

// ---- tail ----

func (p *vertexProto) Tail(t *Transformer) {
	w := t.W()
	if !p.messagesSeen {
		t.Bail("no message found in the request body")
		return
	}
	mapped, err := p.opt.MapModel(p.model)
	if err != nil {
		t.Bail(err.Error())
		return
	}
	if len(p.opt.SafetySettings) > 0 { // the buffered field has omitempty: an empty slice is left out
		b, _ := json.Marshal(p.opt.SafetySettings)
		w.Key("safetySettings")
		w.Raw(b)
	}
	cfg := vertexGenerationConfig{Temperature: p.temp, TopP: p.topP, MaxOutputTokens: p.maxTok}
	if p.effort != "" {
		cfg.ThinkingConfig = vertexThinking{IncludeThoughts: true, ThinkingBudget: 1024}
		switch p.effort {
		case "none":
			cfg.ThinkingConfig = vertexThinking{}
		case "low":
			cfg.ThinkingConfig.ThinkingBudget = 1024
		case "medium":
			cfg.ThinkingConfig.ThinkingBudget = 4096
		case "high":
			cfg.ThinkingConfig.ThinkingBudget = 16384
		}
	}
	if p.rfRaw != nil && p.opt.ApplyResponseFormat != nil {
		var rf map[string]any
		_ = json.Unmarshal(p.rfRaw, &rf) // validated in OnValue
		mime, schema, err := p.opt.ApplyResponseFormat(rf, mapped)
		if err != nil {
			t.Bail(err.Error())
			return
		}
		cfg.ResponseMimeType, cfg.ResponseSchema = mime, schema
	}
	b, _ := json.Marshal(cfg)
	w.Key("generationConfig")
	w.Raw(b)
}
