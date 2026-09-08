package streamxform

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Streaming conversion Claude Messages → OpenAI chat, the first stage of the automatic protocol conversion
// (ConvertClaudeRequestToOpenAIWithOptions); the provider's own transformer is the second stage of the pipeline.
//
// Derived from the buffered converter: each Claude message becomes one OpenAI message, or several when it carries
// tool_use / tool_result blocks; the system prompt becomes the first message; tools, tool_choice, thinking and
// the scalars are mapped one to one. A message is taken whole, bounded, and converted in memory -- its OpenAI shape
// depends on blocks that may come last -- so memory is one message, not the conversation. The system prompt has
// to come out first: a messages array that arrives before system is held, bounded, and replayed once system is
// known or the body ends.
//
// The converter also attaches Claude-internal fields (claude_content_blocks, claude_thinking ...) that the
// buffered path strips again before an OpenAI-compatible provider sees them; EmitInternal false leaves them out.

type ClaudeToOpenAIOptions struct {
	// PreserveReasoning mirrors supportsMessageReasoningContent: thinking text becomes reasoning_content.
	PreserveReasoning bool
	// DisableStreamUsageStats mirrors the setting: no stream_options.include_usage.
	DisableStreamUsageStats bool
	// EmitInternal writes the Claude-internal fields the buffered converter adds for bedrock / claude targets.
	EmitInternal bool
}

const c2oMessageCap = 1 << 20

type c2oProto struct {
	opt ClaudeToOpenAIOptions

	model      string
	modelSeen  bool
	stream     bool
	streamSeen bool
	temp, topP []byte
	maxTok     []byte
	stopRaw    []byte
	toolsRaw   []byte
	choiceRaw  []byte
	thinkRaw   []byte
	outputRaw  []byte
	betaRaw    []byte

	systemSeen   bool
	systemRaw    []byte
	messagesSeen bool
	msgsOpen     bool
	msgCount     int
	pending      [][]byte // converted messages waiting for the system prompt, bounded
	pendingBytes int
}

// NewClaudeToOpenAI builds the Claude → OpenAI transformer.
func NewClaudeToOpenAI(opt ClaudeToOpenAIOptions) *Transformer {
	p := &c2oProto{opt: opt}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

func (p *c2oProto) Prelude() Prelude {
	return Prelude{Model: p.model, ModelSeen: p.modelSeen, Stream: p.stream, StreamSeen: p.streamSeen}
}

func (p *c2oProto) OnKey(t *Transformer) Action {
	switch t.Depth() {
	case 1:
		switch t.Last() {
		case "model":
			return Capture(4 << 10)
		case "stream":
			return Capture(16)
		case "temperature", "top_p", "max_tokens":
			return Capture(64)
		case "stop_sequences":
			return Capture(64 << 10)
		case "tools":
			return Capture(toolsCap)
		case "tool_choice", "thinking":
			return Capture(4 << 10)
		case "output_config", "anthropic_beta":
			return Capture(smallCap)
		case "system":
			return Capture(systemCap)
		case "messages":
			return Probe()
		}
		return Skip() // metadata / top_k / service_tier / anthropic_version: no OpenAI counterpart
	}
	return Bail("unexpected path: " + t.PathString())
}

func (p *c2oProto) OnElem(t *Transformer) Action {
	if t.Depth() == 2 {
		return Capture(c2oMessageCap) // converted whole: its OpenAI shape depends on blocks that may come last
	}
	return Bail("unexpected array: " + t.PathString())
}

func (p *c2oProto) OnStart(t *Transformer, kind ValueKind) Action {
	if t.Depth() == 1 && t.Last() == "messages" {
		if kind != KindArray {
			return Bail("messages is not an array, the buffered decode fails")
		}
		p.messagesSeen = true
		if p.systemSeen {
			p.openMessages(t) // otherwise the converted messages wait for the system prompt, which has to come first
		}
		return Enter().Flat()
	}
	return Bail("unexpected Probe: " + t.PathString())
}

// openMessages starts the output messages array with the system message, when there is one.
func (p *c2oProto) openMessages(t *Transformer) {
	if p.msgsOpen {
		return
	}
	p.msgsOpen = true
	w := t.W()
	w.PushArr("messages")
	if p.systemSeen {
		p.writeSystem(t)
	}
	for _, b := range p.pending {
		w.Elem()
		w.Raw(b)
		p.msgCount++
	}
	p.pending, p.pendingBytes = nil, 0
}

func (p *c2oProto) OnValue(t *Transformer, raw []byte) {
	switch t.Depth() {
	case 1:
		p.topValue(t, raw)
	case 2:
		p.convertMessage(t, raw)
	}
}

func (p *c2oProto) topValue(t *Transformer, raw []byte) {
	isNull := string(raw) == "null"
	keep := func() []byte {
		if isNull {
			return nil
		}
		return append([]byte(nil), raw...)
	}
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
		if !isNull && !isNumLiteral(raw) {
			t.Bail(t.Last() + " is not a number")
			return
		}
		if t.Last() == "temperature" {
			p.temp = keep()
		} else {
			p.topP = keep()
		}
	case "max_tokens":
		if !isNull && !isIntLiteral(raw) {
			t.Bail("max_tokens is not an integer")
			return
		}
		p.maxTok = keep()
	case "stop_sequences":
		var ss []string
		if !isNull && json.Unmarshal(raw, &ss) != nil {
			t.Bail("stop_sequences is not an array of strings")
			return
		}
		p.stopRaw = keep()
	case "tools":
		p.toolsRaw = keep()
	case "tool_choice":
		p.choiceRaw = keep()
	case "thinking":
		p.thinkRaw = keep()
	case "output_config":
		p.outputRaw = keep()
	case "anthropic_beta":
		p.betaRaw = keep()
	case "system":
		p.systemSeen = true
		p.systemRaw = keep()
		if p.messagesSeen {
			p.openMessages(t) // writes the system message, then the messages that were waiting for it
		}
	}
}

func (p *c2oProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	return Bail("unexpected prefix: " + t.PathString()), 0
}

func (p *c2oProto) OnLeave(t *Transformer) {}

// ---- message conversion (in memory, one message at a time) ----

type c2oBlock struct {
	Type         string          `json:"type"`
	Text         string          `json:"text,omitempty"`
	Data         string          `json:"data,omitempty"`
	Source       *c2oSource      `json:"source,omitempty"`
	CacheControl map[string]any  `json:"cache_control,omitempty"`
	Id           string          `json:"id,omitempty"`
	Name         string          `json:"name,omitempty"`
	Input        *map[string]any `json:"input,omitempty"`
	ToolUseId    string          `json:"tool_use_id,omitempty"`
	Content      json.RawMessage `json:"content,omitempty"`
	IsError      bool            `json:"is_error,omitempty"`
	Signature    string          `json:"signature,omitempty"`
	Thinking     string          `json:"thinking,omitempty"`
}

type c2oSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	Url       string `json:"url,omitempty"`
}

type c2oPart struct {
	CacheControl map[string]any `json:"cache_control,omitempty"`
	Type         string         `json:"type,omitempty"`
	Text         string         `json:"text"`
	ImageUrl     *c2oImageURL   `json:"image_url,omitempty"`
}

type c2oImageURL struct {
	Url string `json:"url,omitempty"`
}

type c2oToolCall struct {
	Index    int    `json:"index"`
	Id       string `json:"id,omitempty"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type c2oMessage struct {
	Role                string        `json:"role,omitempty"`
	Content             any           `json:"content,omitempty"`
	ToolCalls           []c2oToolCall `json:"tool_calls,omitempty"`
	ToolCallId          string        `json:"tool_call_id,omitempty"`
	ReasoningContent    string        `json:"reasoning_content,omitempty"`
	ClaudeContentBlocks []c2oBlock    `json:"claude_content_blocks,omitempty"`
}

type c2oConversion struct {
	textParts, reasoningParts []string
	toolCalls                 []c2oToolCall
	toolResults               []c2oBlock
	parts                     []c2oPart
	hasReasoning              bool
	blocks                    []c2oBlock
}

// convertBlocks reproduces convertContentArray.
func convertBlocks(blocks []c2oBlock) *c2oConversion {
	r := &c2oConversion{textParts: []string{}, reasoningParts: []string{}, toolCalls: []c2oToolCall{}, toolResults: []c2oBlock{}, parts: []c2oPart{}}
	preserved := make([]c2oBlock, 0, len(blocks))
	keep := false
	for _, b := range blocks {
		pb := b
		switch b.Type {
		case "text":
			if b.Text != "" {
				txt := stripCchFromBillingHeader(b.Text)
				pb.Text = txt
				r.textParts = append(r.textParts, txt)
				r.parts = append(r.parts, c2oPart{Type: "text", Text: txt, CacheControl: b.CacheControl})
			}
		case "thinking":
			r.hasReasoning, keep = true, true
			if b.Thinking != "" {
				r.reasoningParts = append(r.reasoningParts, b.Thinking)
			}
		case "redacted_thinking":
			r.hasReasoning, keep = true, true
		case "image":
			if b.Source != nil {
				switch b.Source.Type {
				case "base64":
					r.parts = append(r.parts, c2oPart{Type: "image_url", ImageUrl: &c2oImageURL{Url: fmt.Sprintf("data:%s;base64,%s", b.Source.MediaType, b.Source.Data)}})
				case "url":
					r.parts = append(r.parts, c2oPart{Type: "image_url", ImageUrl: &c2oImageURL{Url: b.Source.Url}})
				}
			}
		case "tool_use":
			keep = true
			if b.Id != "" && b.Name != "" {
				var tc c2oToolCall
				tc.Id, tc.Type = b.Id, "function"
				tc.Function.Name = b.Name
				if b.Input != nil {
					if ab, err := json.Marshal(b.Input); err == nil {
						tc.Function.Arguments = string(ab)
					}
				}
				r.toolCalls = append(r.toolCalls, tc)
			}
		case "tool_result":
			keep = true
			r.toolResults = append(r.toolResults, b)
		}
		preserved = append(preserved, pb)
	}
	if keep {
		r.blocks = preserved
	}
	return r
}

// stripCchFromBillingHeader reproduces the buffered helper: the dynamic "; cch=..." field of the billing header line.
func stripCchFromBillingHeader(text string) string {
	const prefix = "x-anthropic-billing-header:"
	if !strings.HasPrefix(text, prefix) {
		return text
	}
	result := text
	for {
		i := strings.Index(result, "; cch=")
		if i == -1 {
			break
		}
		end := strings.Index(result[i+len("; cch="):], ";")
		if end == -1 {
			result = result[:i]
			break
		}
		result = result[:i] + result[i+len("; cch=")+end:]
	}
	return result
}

// toolResultText reproduces GetStringValue on a tool_result's content: a string as is, an array as its text
// blocks joined, anything else empty.
func c2oToolResultText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	if raw[0] == '"' {
		var s string
		_ = json.Unmarshal(raw, &s)
		return s
	}
	if raw[0] == '[' {
		var items []c2oBlock
		if json.Unmarshal(raw, &items) != nil {
			return ""
		}
		var texts []string
		for _, it := range items {
			if it.Type == "text" {
				texts = append(texts, it.Text)
			}
		}
		return strings.Join(texts, "\n")
	}
	return ""
}

func (p *c2oProto) applyReasoning(m *c2oMessage, r *c2oConversion) {
	if p.opt.PreserveReasoning && len(r.reasoningParts) > 0 {
		m.ReasoningContent = strings.Join(r.reasoningParts, "\n\n")
	}
	if p.opt.EmitInternal {
		m.ClaudeContentBlocks = r.blocks
	}
}

func (p *c2oProto) writeMessage(t *Transformer, m c2oMessage) {
	b, err := json.Marshal(m)
	if err != nil {
		t.Bail("message failed to encode: " + err.Error())
		return
	}
	if !p.msgsOpen {
		// the system prompt has not been seen yet and has to come first: hold the converted message, bounded
		p.pendingBytes += len(b)
		if p.pendingBytes > c2oMessageCap {
			t.Bail("messages ahead of the system prompt exceed the hold cap")
			return
		}
		p.pending = append(p.pending, b)
		return
	}
	w := t.W()
	w.Elem()
	w.Raw(b)
	p.msgCount++
}

// convertMessage reproduces one iteration of the buffered converter's message loop.
func (p *c2oProto) convertMessage(t *Transformer, raw []byte) {
	var msg struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &msg); err != nil {
		t.Bail("message failed to decode, the buffered decode fails: " + err.Error())
		return
	}
	c := msg.Content
	if string(c) == "null" { // the buffered content wrapper decodes null as the empty string
		p.writeMessage(t, c2oMessage{Role: msg.Role, Content: ""})
		return
	}
	if len(c) > 0 && c[0] == '"' {
		var s string
		if err := json.Unmarshal(c, &s); err != nil {
			t.Bail("invalid content string")
			return
		}
		p.writeMessage(t, c2oMessage{Role: msg.Role, Content: s})
		return
	}
	var blocks []c2oBlock
	if len(c) > 0 && string(c) != "null" {
		if err := json.Unmarshal(c, &blocks); err != nil {
			t.Bail("content is neither a string nor an array of blocks, the buffered decode fails")
			return
		}
	}
	r := convertBlocks(blocks)
	if len(r.toolCalls) > 0 {
		m := c2oMessage{Role: msg.Role, ToolCalls: r.toolCalls}
		p.applyReasoning(&m, r)
		if len(r.textParts) > 0 {
			m.Content = strings.Join(r.textParts, "\n\n")
		}
		p.writeMessage(t, m)
	}
	if len(r.toolResults) > 0 {
		for _, tr := range r.toolResults {
			m := c2oMessage{Role: "tool", Content: c2oToolResultText(tr.Content), ToolCallId: tr.ToolUseId}
			if p.opt.EmitInternal {
				m.ClaudeContentBlocks = []c2oBlock{tr}
			}
			p.writeMessage(t, m)
		}
		if len(r.textParts) > 0 {
			p.writeMessage(t, c2oMessage{Role: msg.Role, Content: strings.Join(r.textParts, "\n\n")})
		}
	}
	if len(r.toolCalls) == 0 && len(r.toolResults) == 0 {
		m := c2oMessage{Role: msg.Role}
		if len(r.parts) > 0 {
			m.Content = r.parts
		}
		p.applyReasoning(&m, r)
		if m.Content == nil && m.ReasoningContent == "" && r.hasReasoning {
			m.Content = ""
		}
		p.writeMessage(t, m)
	}
}

// writeSystem reproduces the system message: a string with the billing header cleaned, or the parts of an array.
func (p *c2oProto) writeSystem(t *Transformer) {
	raw := p.systemRaw
	if len(raw) == 0 || string(raw) == "null" {
		return
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			t.Bail("invalid system string")
			return
		}
		p.writeMessage(t, c2oMessage{Role: "system", Content: stripCchFromBillingHeader(s)})
		return
	}
	var blocks []c2oBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		t.Bail("system is neither a string nor an array of blocks, the buffered decode fails")
		return
	}
	p.writeMessage(t, c2oMessage{Role: "system", Content: convertBlocks(blocks).parts})
}

// ---- tail ----

func (p *c2oProto) Tail(t *Transformer) {
	w := t.W()
	if p.msgsOpen || len(p.pending) > 0 || (p.systemSeen && p.systemRaw != nil && string(p.systemRaw) != "null") {
		p.openMessages(t) // messages that never saw a system prompt, or a system prompt alone
	} else {
		w.Key("messages")
		w.RawString("null") // no message came out: the buffered nil slice
	}
	if p.msgsOpen {
		w.Open()
		w.Pop()
	}
	w.Key("model")
	w.JSONString(p.model)
	if p.stream {
		w.Key("stream")
		w.RawString("true")
		if !p.opt.DisableStreamUsageStats {
			w.Key("stream_options")
			w.RawString(`{"include_usage":true}`)
		}
	}
	for _, kv := range []struct {
		k string
		v []byte
	}{{"temperature", p.temp}, {"top_p", p.topP}, {"max_tokens", p.maxTok}} {
		if kv.v != nil && !isZeroNum(kv.v) {
			w.Key(kv.k)
			w.Raw(kv.v)
		}
	}
	if p.stopRaw != nil {
		var ss []string
		_ = json.Unmarshal(p.stopRaw, &ss)
		if len(ss) > 0 {
			b, _ := json.Marshal(ss)
			w.Key("stop")
			w.Raw(b)
		}
	}
	if p.toolsRaw != nil {
		var tools []struct {
			Name        string         `json:"name"`
			Description string         `json:"description"`
			InputSchema map[string]any `json:"input_schema"`
		}
		if err := json.Unmarshal(p.toolsRaw, &tools); err != nil {
			t.Bail("tools failed to decode, the buffered decode fails")
			return
		}
		if len(tools) > 0 {
			type fn struct {
				Description string         `json:"description,omitempty"`
				Name        string         `json:"name"`
				Parameters  map[string]any `json:"parameters,omitempty"`
			}
			type tool struct {
				Type     string `json:"type"`
				Function fn     `json:"function"`
			}
			out := make([]tool, 0, len(tools))
			for _, ct := range tools {
				out = append(out, tool{Type: "function", Function: fn{Name: ct.Name, Description: ct.Description, Parameters: ct.InputSchema}})
			}
			b, _ := json.Marshal(out)
			w.Key("tools")
			w.Raw(b)
		}
	}
	if p.choiceRaw != nil {
		var tc struct {
			Type                   string `json:"type"`
			Name                   string `json:"name"`
			DisableParallelToolUse bool   `json:"disable_parallel_tool_use"`
		}
		if err := json.Unmarshal(p.choiceRaw, &tc); err != nil {
			t.Bail("tool_choice failed to decode, the buffered decode fails")
			return
		}
		w.Key("tool_choice")
		switch {
		case tc.Type == "tool" && tc.Name != "":
			b, _ := json.Marshal(map[string]any{"type": "function", "function": map[string]any{"name": tc.Name}})
			w.Raw(b)
		case tc.Type == "any":
			w.RawString(`"required"`)
		default:
			w.JSONString(tc.Type)
		}
		w.Key("parallel_tool_calls")
		if tc.DisableParallelToolUse {
			w.RawString("false")
		} else {
			w.RawString("true")
		}
	}
	if p.thinkRaw != nil {
		var th struct {
			Type         string `json:"type"`
			BudgetTokens int    `json:"budget_tokens"`
		}
		if err := json.Unmarshal(p.thinkRaw, &th); err != nil {
			t.Bail("thinking failed to decode, the buffered decode fails")
			return
		}
		if p.opt.EmitInternal {
			w.Key("claude_thinking")
			w.Raw(p.thinkRaw)
		}
		if th.Type == "enabled" {
			effort := "high"
			if th.BudgetTokens < 4096 {
				effort = "low"
			} else if th.BudgetTokens < 16384 {
				effort = "medium"
			}
			w.Key("reasoning_effort")
			w.JSONString(effort)
		}
	}
	if p.opt.EmitInternal {
		if p.outputRaw != nil {
			w.Key("claude_output_config")
			w.Raw(p.outputRaw)
		}
		if p.betaRaw != nil {
			w.Key("claude_anthropic_beta")
			w.Raw(p.betaRaw)
		}
	}
}
