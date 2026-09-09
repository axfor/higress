package streamxform

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
)

// Streaming conversion OpenAI → Amazon Bedrock Converse.
//
// Derived from the buffered bedrock.go buildBedrockTextGenerationRequest and chatMessage2BedrockMessage:
//   - system messages become system[] text blocks; a tool message becomes a toolResult block inside a user message,
//     merged into the previous message when that one is a tool-result user message; a message with tool_calls
//     becomes toolUse blocks, one per call, and its content is ignored; otherwise a string content is one text block
//     and an array content its text and data-URL image parts (anything else, http image URLs included, is dropped);
//   - inferenceConfig carries maxTokens (max_completion_tokens first), temperature and topP; additionalModelRequestFields
//     carries thinking (from claude_thinking, else reasoning_effort) and the provider's configured extras;
//     performanceConfig is always standard latency; outputConfig follows claude_output_config.format; toolConfig
//     is written whenever tools is an array and tool_choice is not "none", with the choice mapped by Bedrock's rules;
//   - prompt cache points need the last user message, so a request that would get them takes the buffered path.
//
// As in the Vertex shape, an assistant message's content waits for tool_calls (bounded, then fallback); for any
// other role the content streams and tool_calls turning up afterwards falls back.

type BedrockOptions struct {
	// MapModel reproduces the buffered mapModel: an error when model is empty or maps to empty.
	MapModel func(model string) (string, error)
	// AdditionalFields mirrors the setting bedrockAdditionalFields, merged into additionalModelRequestFields.
	AdditionalFields map[string]any
	// PromptCacheRetention mirrors the setting promptCacheRetention (the request's own field takes precedence).
	PromptCacheRetention string
	// PromptCacheSupported reproduces isPromptCacheSupportedModel for the mapped model.
	PromptCacheSupported func(mapped string) bool
}

type bedrockInference struct {
	MaxTokens   int     `json:"maxTokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	TopP        float64 `json:"topP,omitempty"`
}

type bedrockToolChoiceOut struct {
	Any  *struct{} `json:"any,omitempty"`
	Auto *struct{} `json:"auto,omitempty"`
	Tool *struct {
		Name string `json:"name"`
	} `json:"tool,omitempty"`
}

type bedrockToolInputSchema struct {
	Json map[string]any `json:"json,omitempty"`
}

type bedrockToolSpecOut struct {
	InputSchema bedrockToolInputSchema `json:"inputSchema,omitempty"`
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
}

type bedrockToolOut struct {
	ToolSpec bedrockToolSpecOut `json:"toolSpec,omitempty"`
}

type bedrockToolConfigOut struct {
	Tools      []bedrockToolOut     `json:"tools,omitempty"`
	ToolChoice bedrockToolChoiceOut `json:"toolChoice,omitempty"`
}

type bedrockToolUseOut struct {
	Input     map[string]any `json:"input"`
	Name      string         `json:"name"`
	ToolUseId string         `json:"toolUseId"`
}

type bedrockSystemBlock struct {
	Text string `json:"text,omitempty"`
}

type bedrockToolResultOut struct {
	ToolUseId string               `json:"toolUseId"`
	Content   []map[string]*string `json:"content"`
}

type bdkTool struct {
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  map[string]any `json:"parameters"`
	} `json:"function"`
}

type bedrockMsg struct {
	role            string
	roleSeen        bool
	roleWritten     bool
	contentWritten  bool
	finalizing      bool
	toolCallsRaw    []byte
	toolCallsSeen   bool
	toolCallsDone   bool
	toolCallsWrote  bool
	streamedContent bool
	toolCallId      string
	toolContent     []byte
	toolContentSeen bool
	sysContentSeen  bool
	part            gemPart
}

type bedrockProto struct {
	opt BedrockOptions

	model         string
	modelSeen     bool
	stream        bool
	streamSeen    bool
	maxTok        int
	maxCompletion int
	temp, topP    float64
	effort        string
	thinkingRaw   []byte
	outputRaw     []byte
	toolsRaw      []byte
	toolsSeen     bool
	toolChoiceRaw []byte
	pcr           string

	messagesSeen   bool
	inputMsgs      int
	msgLevel       int
	openToolResult bool
	system         []bedrockSystemBlock
	m              bedrockMsg
}

// NewBedrock builds the OpenAI → Bedrock Converse transformer.
func NewBedrock(opt BedrockOptions) *Transformer {
	opt.MapModel = strictMapper(opt.MapModel)
	if opt.PromptCacheSupported == nil {
		opt.PromptCacheSupported = func(string) bool { return false }
	}
	p := &bedrockProto{opt: opt}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

func (p *bedrockProto) Prelude() Prelude {
	return Prelude{Model: p.model, ModelSeen: p.modelSeen, Stream: p.stream, StreamSeen: p.streamSeen}
}

// ---- dispatch ----

func (p *bedrockProto) OnKey(t *Transformer) Action {
	switch t.Depth() {
	case 1:
		switch t.Last() {
		case "model":
			return Capture(4 << 10)
		case "messages":
			return Probe()
		case "stream":
			return Capture(16)
		case "max_tokens", "max_completion_tokens", "temperature", "top_p":
			return Capture(64)
		case "reasoning_effort", "prompt_cache_retention":
			return Capture(256)
		case "claude_thinking", "tool_choice":
			return Capture(4 << 10)
		case "claude_output_config":
			return Capture(smallCap)
		case "tools":
			return Capture(toolsCap) // re-marshalled through structs by the buffered path; kept whole
		}
		return Skip() // prompt_cache_key is only warned about; nothing else is read
	case 3:
		m := &p.m
		switch t.Last() {
		case "role":
			return Capture(256)
		case "content":
			if !m.roleSeen {
				return Defer(roleWaitCap)
			}
			switch m.role {
			case "system":
				return Capture(systemCap)
			case "tool":
				return Capture(assistantWaitCap)
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
		case "tool_call_id":
			return Capture(4 << 10)
		case "claude_content_blocks":
			return Bail("claude_content_blocks needs a struct round trip, not reproduced by streaming")
		}
		return Skip()
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
			return Prefix(urlPrefixWin)
		}
		return Skip()
	}
	return Bail("unexpected path: " + t.PathString())
}

func (p *bedrockProto) OnElem(t *Transformer) Action {
	switch t.Depth() {
	case 2, 4:
		return Probe()
	}
	return Bail("unexpected array: " + t.PathString())
}

func (p *bedrockProto) OnStart(t *Transformer, kind ValueKind) Action {
	w := t.W()
	switch t.Depth() {
	case 1: // messages: always written by the buffered path
		if kind != KindArray {
			return Bail("messages is not an array")
		}
		p.msgLevel = w.Level() + 1
		return Enter().As("messages")
	case 2:
		if kind != KindObject {
			return Bail("message is not an object")
		}
		p.m = bedrockMsg{}
		p.inputMsgs++
		return Enter().Lazy() // system messages and merged tool results produce no element of their own
	case 3: // content of an ordinary message, streaming
		switch kind {
		case KindString:
			p.closeToolResultIfOpen(t)
			p.writeRole(t)
			p.m.contentWritten = true
			return Prefix(1) // Text has omitempty: an empty string is [{}]
		case KindArray:
			p.closeToolResultIfOpen(t)
			p.writeRole(t)
			p.m.contentWritten = true
			return Enter().As("content").Lazy() // nil when no part lands: null, written in OnLeave
		}
		return Skip() // object / scalar / null: ParseContent yields nothing → "content":null
	case 4:
		if kind != KindObject {
			return Skip()
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

func (p *bedrockProto) OnValue(t *Transformer, raw []byte) {
	switch t.Depth() {
	case 1:
		p.topValue(t, raw)
	case 3:
		p.msgValue(t, raw)
	case 5:
		p.partValue(t, raw)
	}
}

func (p *bedrockProto) topValue(t *Transformer, raw []byte) {
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
	case "max_tokens", "max_completion_tokens":
		if isNull {
			return
		}
		if !isIntLiteral(raw) {
			t.Bail(t.Last() + " is not an integer")
			return
		}
		if t.Last() == "max_tokens" {
			p.maxTok = atoi(raw)
		} else {
			p.maxCompletion = atoi(raw)
		}
	case "temperature", "top_p":
		if isNull {
			return
		}
		if !isNumLiteral(raw) {
			t.Bail(t.Last() + " is not a number")
			return
		}
		f, _ := parseFloat(raw)
		if t.Last() == "temperature" {
			p.temp = f
		} else {
			p.topP = f
		}
	case "reasoning_effort", "prompt_cache_retention":
		if isNull {
			return
		}
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail(t.Last() + " is not a string")
			return
		}
		if t.Last() == "reasoning_effort" {
			p.effort = s
		} else {
			p.pcr = s
		}
	case "claude_thinking":
		if !isNull {
			p.thinkingRaw = append([]byte(nil), raw...)
		}
	case "claude_output_config":
		if !isNull {
			p.outputRaw = append([]byte(nil), raw...)
		}
	case "tool_choice":
		if !isNull {
			p.toolChoiceRaw = append([]byte(nil), raw...)
		}
	case "tools":
		if isNull {
			return
		}
		var tools []bdkTool
		if err := json.Unmarshal(raw, &tools); err != nil {
			t.Bail("tools failed to decode: " + err.Error())
			return
		}
		p.toolsRaw, p.toolsSeen = append([]byte(nil), raw...), true
	}
}

func (p *bedrockProto) msgValue(t *Transformer, raw []byte) {
	m := &p.m
	switch t.Last() {
	case "role":
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("role is not a string")
			return
		}
		m.role, m.roleSeen = s, true
		if m.toolCallsSeen && !m.toolCallsDone && s != "system" && s != "tool" {
			p.applyToolCalls(t)
			return
		}
		if len(t.Deferred()) > 0 {
			t.Release()
		}
	case "content": // system and tool contents are Captured whole
		switch m.role {
		case "system":
			m.sysContentSeen = true
			p.system = append(p.system, bedrockSystemBlock{Text: stringContent(t, raw)})
		case "tool":
			m.toolContent, m.toolContentSeen = append([]byte(nil), raw...), true
		}
	case "tool_calls":
		m.toolCallsRaw, m.toolCallsSeen = append([]byte(nil), raw...), true
		if m.streamedContent {
			var tcs []oaiToolCall
			if json.Unmarshal(raw, &tcs) == nil && len(tcs) > 0 {
				t.Bail("tool_calls after the content of a non-assistant message, whose content already went out")
			}
			return
		}
		if m.roleSeen && m.role != "system" && m.role != "tool" {
			p.applyToolCalls(t)
		}
	case "tool_call_id":
		s, ok := jsonUnquote(raw)
		if !ok && string(raw) != "null" {
			t.Bail("tool_call_id is not a string")
			return
		}
		m.toolCallId = s
	}
}

func (p *bedrockProto) applyToolCalls(t *Transformer) {
	m := &p.m
	m.toolCallsDone = true
	var tcs []oaiToolCall
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
	p.closeToolResultIfOpen(t)
	p.writeRole(t)
	blocks := make([]map[string]bedrockToolUseOut, 0, len(tcs))
	for _, tc := range tcs {
		input := map[string]any{}
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &input) // the buffered path ignores the error and keeps an empty map
		blocks = append(blocks, map[string]bedrockToolUseOut{"toolUse": {Input: input, Name: tc.Function.Name, ToolUseId: tc.Id}})
	}
	b, _ := json.Marshal(blocks)
	w := t.W()
	w.Key("content")
	w.Raw(b)
	m.contentWritten = true
	m.toolCallsWrote = true
}

func (p *bedrockProto) partValue(t *Transformer, raw []byte) {
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
		pt.dead = true // "type is not supported": skipped
		t.DropDeferred()
	}
}

// OnPrefix: string content, text parts and the head of image_url.url (extractImageType: data:<mime>;base64,).
func (p *bedrockProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	w := t.W()
	switch t.Depth() {
	case 3:
		w.Key("content")
		if complete && len(raw) == 0 {
			w.RawString(`[{}]`)
			return Skip(), 0
		}
		return Pass().Wrap(lit5, lit6), 0
	case 5:
		if complete && len(raw) == 0 {
			w.Open()
			return Skip(), 0
		}
		w.KeyRaw(t.KeyRaw())
		return Pass().Wrap(lit0, lit0), 0
	}
	dec, off := unescapePrefix(raw)
	if !bytes.HasPrefix(dec, lit1) {
		if !complete && len(dec) < len(lit1) {
			return Bail("image URL cut at the window boundary"), 0
		}
		return Skip(), 0 // "image url is not supported": the part is dropped
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
	sub := strings.SplitN(mime, "/", 2)
	if !bytes.HasPrefix(rest, lit4) || len(sub) < 2 {
		return Skip(), 0
	}
	resume := off[semi+1+len(lit4)]
	if !complete && resume >= len(raw) {
		return Bail("data URL payload cut at the window boundary"), 0
	}
	w.Key("image")
	w.RawString(`{"format":`)
	w.JSONString(sub[1])
	if resume >= len(raw) { // nothing after the prefix: Bytes has omitempty
		w.RawString(`,"source":{}}`)
		return Skip(), 0
	}
	w.RawString(`,"source":{"bytes":"`)
	return Pass().Wrap(nil, []byte(`"}}`)), resume
}

// ---- containers closing ----

func (p *bedrockProto) OnLeave(t *Transformer) {
	w := t.W()
	switch t.Depth() {
	case 1:
		p.messagesSeen = true
		if p.inputMsgs == 0 {
			t.Bail("no message found in the request body")
			return
		}
		p.closeToolResultIfOpen(t)
	case 2:
		p.finishMessage(t)
	case 3: // content array: nothing landed means the buffered nil slice, null
		if !w.Opened(w.Level()) {
			w.KeyAt(w.Level()-1, "content")
			w.RawString("null")
		}
	case 4:
		if p.m.part.dead || !p.m.part.typSeen {
			t.DropDeferred()
		}
	}
}

func (p *bedrockProto) closeToolResultIfOpen(t *Transformer) {
	if p.openToolResult {
		t.W().RawString("]}")
		p.openToolResult = false
	}
}

func (p *bedrockProto) writeRole(t *Transformer) {
	m := &p.m
	if m.roleWritten {
		return
	}
	m.roleWritten = true
	w := t.W()
	w.Open()
	w.Key("role") // no omitempty on the buffered Role
	w.JSONString(m.role)
}

// toolResultBlock reproduces chatToolMessage2BedrockToolResultContent for a tool message.
func (p *bedrockProto) toolResultBlock(t *Transformer) []byte {
	m := &p.m
	var content []map[string]*string
	if m.toolContentSeen {
		raw := m.toolContent
		if len(raw) > 0 && raw[0] == '"' {
			s, ok := jsonUnquote(raw)
			if !ok {
				t.Bail("invalid tool content string")
				return nil
			}
			content = []map[string]*string{{"text": &s}}
		} else if len(raw) > 0 && raw[0] == '[' {
			var items []any
			if err := json.Unmarshal(raw, &items); err != nil {
				t.Bail("invalid tool content array")
				return nil
			}
			for _, it := range items {
				mm, ok := it.(map[string]any)
				if !ok || mm["type"] != "text" {
					continue
				}
				if s, ok := mm["text"].(string); ok {
					s := s
					content = append(content, map[string]*string{"text": &s})
				}
			}
		}
	}
	b, _ := json.Marshal(map[string]bedrockToolResultOut{"toolResult": {ToolUseId: m.toolCallId, Content: content}})
	return b
}

func (p *bedrockProto) finishMessage(t *Transformer) {
	m := &p.m
	w := t.W()
	if !m.roleSeen {
		m.role, m.roleSeen = "", true
	}
	m.finalizing = true
	switch m.role {
	case "system":
		if len(t.Deferred()) > 0 {
			t.ReleaseNow()
		}
		if !m.sysContentSeen {
			p.system = append(p.system, bedrockSystemBlock{}) // content missing: StringContent of nil is ""
		}
		return
	case "tool":
		if len(t.Deferred()) > 0 {
			t.ReleaseNow()
			if t.Dead() {
				return
			}
		}
		blk := p.toolResultBlock(t)
		if t.Dead() {
			return
		}
		if p.openToolResult {
			w.Byte(',')
			w.Raw(blk)
			return
		}
		if !w.ElemAt(p.msgLevel) {
			t.Bail("tool message written at an unexpected position")
			return
		}
		w.RawString(`{"role":"user","content":[`)
		w.Raw(blk)
		p.openToolResult = true
		return
	}
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
	p.closeToolResultIfOpen(t)
	p.writeRole(t)
	if !m.contentWritten {
		w.Key("content")
		w.RawString("null")
	}
}

// ---- tail ----

func (p *bedrockProto) Tail(t *Transformer) {
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
	pcr := p.pcr
	if pcr == "" {
		pcr = p.opt.PromptCacheRetention
	}
	if pcr != "" && p.opt.PromptCacheSupported(mapped) {
		t.Bail("prompt cache points are placed on the last user message, which needs the whole conversation")
		return
	}
	if len(p.system) > 0 {
		b, _ := json.Marshal(p.system)
		w.Key("system")
		w.Raw(b)
	}
	maxTokens := p.maxTok
	if p.maxCompletion > 0 {
		maxTokens = p.maxCompletion
	}
	b, _ := json.Marshal(bedrockInference{MaxTokens: maxTokens, Temperature: p.temp, TopP: p.topP})
	w.Key("inferenceConfig")
	w.Raw(b)
	// additionalModelRequestFields: thinking, then the configured extras
	extra := map[string]any{}
	var thinking map[string]any
	if p.thinkingRaw != nil {
		var cfg struct {
			Type         string `json:"type"`
			BudgetTokens int    `json:"budget_tokens"`
			Display      string `json:"display"`
		}
		if err := json.Unmarshal(p.thinkingRaw, &cfg); err != nil {
			t.Bail("claude_thinking failed to decode")
			return
		}
		if cfg.Type != "" && cfg.Type != "disabled" {
			thinking = map[string]any{"type": cfg.Type}
			if cfg.Display != "" {
				thinking["display"] = cfg.Display
			}
			if cfg.Type == "enabled" && cfg.BudgetTokens > 0 {
				thinking["budget_tokens"] = cfg.BudgetTokens
			}
		}
	}
	var output struct {
		Effort string          `json:"effort"`
		Format json.RawMessage `json:"format"`
	}
	if p.outputRaw != nil {
		if err := json.Unmarshal(p.outputRaw, &output); err != nil {
			t.Bail("claude_output_config failed to decode")
			return
		}
	}
	if thinking != nil {
		extra["thinking"] = thinking
		if thinking["type"] == "adaptive" && p.outputRaw != nil && (output.Effort == "low" || output.Effort == "medium" || output.Effort == "high") {
			extra["output_config"] = map[string]any{"effort": output.Effort}
		}
	} else if p.effort != "" {
		budget := 1024
		switch p.effort {
		case "medium":
			budget = 4096
		case "high":
			budget = 16384
		}
		extra["thinking"] = map[string]any{"type": "enabled", "budget_tokens": budget}
	}
	for k, v := range p.opt.AdditionalFields {
		extra[k] = v
	}
	if p.outputRaw != nil && len(output.Format) > 0 && string(output.Format) != "null" {
		w.Key("outputConfig")
		w.RawString(`{"textFormat":`)
		w.Raw(output.Format)
		w.Byte('}')
	}
	if len(extra) > 0 {
		b, _ := json.Marshal(extra)
		w.Key("additionalModelRequestFields")
		w.Raw(b)
	}
	w.Key("performanceConfig")
	w.RawString(`{"latency":"standard"}`)
	// toolConfig
	choiceType, choiceName := p.toolChoice(t)
	if t.Dead() {
		return
	}
	if p.toolsSeen && choiceType != "none" {
		hasThinking := thinking != nil || p.effort != ""
		cfg := bedrockToolConfigOut{}
		cfg.ToolChoice.Auto = &struct{}{}
		switch choiceType {
		case "required", "any":
			if !hasThinking {
				cfg.ToolChoice.Auto = nil
				cfg.ToolChoice.Any = &struct{}{}
			}
		case "function":
			if !hasThinking && choiceName != "" {
				cfg.ToolChoice.Auto = nil
				cfg.ToolChoice.Tool = &struct {
					Name string `json:"name"`
				}{Name: choiceName}
			}
		}
		var tools []bdkTool
		_ = json.Unmarshal(p.toolsRaw, &tools) // validated in OnValue
		cfg.Tools = []bedrockToolOut{}
		for _, tool := range tools {
			var out bedrockToolOut
			out.ToolSpec.InputSchema.Json = tool.Function.Parameters
			out.ToolSpec.Name = tool.Function.Name
			out.ToolSpec.Description = tool.Function.Description
			cfg.Tools = append(cfg.Tools, out)
		}
		b, _ := json.Marshal(cfg)
		w.Key("toolConfig")
		w.Raw(b)
	}
}

// toolChoice reproduces getToolChoiceType / getToolChoiceObject: a string is the type itself, an object gives its
// type and function name, anything else counts as unset.
func (p *bedrockProto) toolChoice(t *Transformer) (typ, name string) {
	if p.toolChoiceRaw == nil {
		return "", ""
	}
	if s, ok := jsonUnquote(p.toolChoiceRaw); ok {
		return s, ""
	}
	var obj struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(p.toolChoiceRaw, &obj); err != nil {
		return "", ""
	}
	return obj.Type, obj.Function.Name
}

var _ = errors.New
