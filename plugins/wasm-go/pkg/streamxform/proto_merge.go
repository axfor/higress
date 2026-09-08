package streamxform

import (
	"bytes"
	"encoding/json"
)

// The mergeConsecutiveMessages stage: the streaming form of the buffered helper of the same name, which runs on
// the OpenAI-shaped chat body before the provider's conversion. Consecutive messages with the same role, user or
// assistant, become one: string contents joined by a blank line, anything else concatenated as content parts
// (ParseContent's normalised form), reasoning_content joined by a blank line. Assistant turns that carry
// tool_calls or function_call on either side stay apart.
//
// A message is held whole (bounded) until the next one's role is known, so the output lags the input by one
// message: message granularity, as the Claude-input conversion. A message never merged goes out byte for byte;
// a merged one is rebuilt from its keys. The buffered helper, when it merged anything, re-serialises the whole
// request through its struct on the way out; that incidental round trip is not reproduced.
//
// The buffered helper logs and keeps the body unchanged when it cannot decode the request; here such a body is
// reported as unsupported, and the buffered path reproduces that outcome.

const mergeMsgCap = 1 << 20

type mergeMsg struct {
	raw       []byte
	keys      []string // in order of appearance, content and reasoning_content excluded
	fields    map[string]json.RawMessage
	role      string
	content   json.RawMessage // nil when absent
	parts     bool            // content is the result of a parts merge: a []chatMessageContent on the buffered path
	reasoning string
	toolish   bool // tool_calls or function_call present
	merged    bool
}

type mergeProto struct {
	held    *mergeMsg
	entered bool
}

// NewMergeConsecutive builds the mergeConsecutiveMessages stage.
func NewMergeConsecutive() *Transformer {
	t := NewTransformer(&mergeProto{})
	t.DupKeyBail = true
	return t
}

func (p *mergeProto) OnKey(t *Transformer) Action {
	if t.Depth() == 1 && t.Last() == "messages" && !p.entered {
		return Probe()
	}
	return Pass()
}

func (p *mergeProto) OnStart(t *Transformer, kind ValueKind) Action {
	if kind != KindArray {
		return Pass() // null decodes to no messages, anything else fails the decode: the body stays as it is
	}
	p.entered = true
	return Enter()
}

func (p *mergeProto) OnElem(t *Transformer) Action {
	return Capture(mergeMsgCap)
}

func (p *mergeProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	return Bail("unexpected prefix: " + t.PathString()), 0
}

// decodeMergeMsg reads what the merge looks at, keeping the rest raw for the rebuild.
func decodeMergeMsg(raw []byte) (*mergeMsg, string) {
	m := &mergeMsg{raw: append([]byte(nil), raw...), fields: map[string]json.RawMessage{}}
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return nil, "message is not an object, the buffered decode fails"
	}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, "message failed to decode: " + err.Error()
		}
		key := kt.(string)
		var val json.RawMessage
		if err := dec.Decode(&val); err != nil {
			return nil, "message failed to decode: " + err.Error()
		}
		switch key {
		case "content":
			if string(val) != "null" {
				m.content = val
			}
			continue
		case "reasoning_content":
			if string(val) == "null" {
				continue
			}
			s, ok := jsonUnquote(val)
			if !ok {
				return nil, "reasoning_content is not a string, the buffered decode fails"
			}
			m.reasoning = s
			continue
		case "role":
			if string(val) != "null" {
				s, ok := jsonUnquote(val)
				if !ok {
					return nil, "role is not a string, the buffered decode fails"
				}
				m.role = s
			}
		case "tool_calls":
			var calls []json.RawMessage
			if string(val) != "null" {
				if err := json.Unmarshal(val, &calls); err != nil {
					return nil, "tool_calls failed to decode: " + err.Error()
				}
			}
			if len(calls) > 0 {
				m.toolish = true
			}
		case "function_call":
			if string(val) != "null" {
				if len(val) == 0 || val[0] != '{' {
					return nil, "function_call is not an object, the buffered decode fails"
				}
				m.toolish = true
			}
		}
		m.keys = append(m.keys, key)
		m.fields[key] = val
	}
	return m, ""
}

func (p *mergeProto) OnValue(t *Transformer, raw []byte) {
	if t.Depth() != 2 {
		return
	}
	m, why := decodeMergeMsg(raw)
	if why != "" {
		t.Bail(why)
		return
	}
	if p.held != nil && p.held.role == m.role && (m.role == "user" || m.role == "assistant") &&
		!(m.role == "assistant" && (p.held.toolish || m.toolish)) {
		content, parts, why := mergeContent(p.held.content, p.held.parts, m.content)
		if why != "" {
			t.Bail(why)
			return
		}
		p.held.content, p.held.parts = content, parts
		p.held.reasoning = mergeReasoning(p.held.reasoning, m.reasoning)
		p.held.merged = true
		return
	}
	p.flush(t)
	p.held = m
}

// flush writes the held message: verbatim unless it absorbed another one.
func (p *mergeProto) flush(t *Transformer) {
	m := p.held
	if m == nil {
		return
	}
	p.held = nil
	w := t.W()
	w.Elem()
	if !m.merged {
		w.Raw(m.raw)
		return
	}
	w.Byte('{')
	first := true
	sep := func() {
		if !first {
			w.Byte(',')
		}
		first = false
	}
	for _, k := range m.keys {
		sep()
		w.JSONString(k)
		w.Byte(':')
		w.Raw(m.fields[k])
	}
	if m.content != nil { // Content has omitempty: a nil stays out
		sep()
		w.RawString(`"content":`)
		w.Raw(m.content)
	}
	if m.reasoning != "" {
		sep()
		w.RawString(`"reasoning_content":`)
		w.JSONString(m.reasoning)
	}
	w.Byte('}')
}

func (p *mergeProto) OnLeave(t *Transformer) {
	if t.Depth() == 1 {
		p.flush(t)
	}
}

func (p *mergeProto) Tail(t *Transformer) {}

// mergeReasoning reproduces the buffered mergeReasoningContent.
func mergeReasoning(prev, curr string) string {
	switch {
	case prev == "":
		return curr
	case curr == "":
		return prev
	default:
		return prev + "\n\n" + curr
	}
}

// mergeContent reproduces mergeMessageContent: two strings joined by a blank line, otherwise both parsed into
// content parts and concatenated. Once a merge has produced parts, the buffered value is a []chatMessageContent,
// which ParseContent does not recognise (it looks for []any): the next merge keeps only the newer message's
// parts. Reproduced through prevParts.
func mergeContent(prev json.RawMessage, prevParts bool, curr json.RawMessage) (json.RawMessage, bool, string) {
	if !prevParts {
		ps, pok := jsonUnquote(prev)
		cs, cok := jsonUnquote(curr)
		if pok && cok {
			return appendJSONString(nil, ps+"\n\n"+cs), false, ""
		}
	}
	var pp []contentPart
	if !prevParts {
		var why string
		if pp, why = parseContentParts(prev); why != "" {
			return nil, false, why
		}
	}
	cp, why := parseContentParts(curr)
	if why != "" {
		return nil, false, why
	}
	parts := append(pp, cp...)
	if parts == nil {
		return json.RawMessage("null"), true, "" // a nil []chatMessageContent marshals as null and is not omitted
	}
	b, _ := json.Marshal(parts)
	return b, true, ""
}

// contentPart is the buffered chatMessageContent as json.Marshal writes it (Text has no omitempty).
type contentPart struct {
	Type       string             `json:"type,omitempty"`
	Text       string             `json:"text"`
	ImageURL   *contentImageURL   `json:"image_url,omitempty"`
	File       *contentFile       `json:"file,omitempty"`
	InputAudio *contentInputAudio `json:"input_audio,omitempty"`
}

type contentImageURL struct {
	URL    string `json:"url,omitempty"`
	Detail string `json:"detail,omitempty"`
}

type contentFile struct {
	FileID string `json:"file_id,omitempty"`
}

type contentInputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

// parseContentParts reproduces ParseContent: a string is one text part; an array yields a part per element of a
// known type, the others are skipped; anything else yields nothing. Where the buffered code asserts a string
// without checking (image_url.url, input_audio.data / format, file.file_id) it panics on a miss: unsupported here.
func parseContentParts(raw json.RawMessage) ([]contentPart, string) {
	if raw == nil {
		return nil, ""
	}
	if s, ok := jsonUnquote(raw); ok {
		return []contentPart{{Type: "text", Text: s}}, ""
	}
	var els []json.RawMessage
	if err := json.Unmarshal(raw, &els); err != nil {
		return nil, "" // not an array: ParseContent yields nothing
	}
	var out []contentPart
	for _, el := range els {
		var m map[string]json.RawMessage
		if err := json.Unmarshal(el, &m); err != nil || m == nil {
			continue
		}
		typ, _ := jsonUnquote(m["type"])
		switch typ {
		case "text":
			if s, ok := jsonUnquote(m["text"]); ok {
				out = append(out, contentPart{Type: "text", Text: s})
			}
		case "image_url":
			var sub map[string]json.RawMessage
			if err := json.Unmarshal(m["image_url"], &sub); err != nil || sub == nil {
				continue
			}
			u, ok := jsonUnquote(sub["url"])
			if !ok {
				return nil, "image_url.url is not a string, the buffered ParseContent panics"
			}
			part := contentPart{Type: "image_url", ImageURL: &contentImageURL{URL: u}}
			if d, ok := jsonUnquote(sub["detail"]); ok {
				part.ImageURL.Detail = d
			}
			out = append(out, part)
		case "input_audio":
			var sub map[string]json.RawMessage
			if err := json.Unmarshal(m["input_audio"], &sub); err != nil || sub == nil {
				continue
			}
			d, dok := jsonUnquote(sub["data"])
			f, fok := jsonUnquote(sub["format"])
			if !dok || !fok {
				return nil, "input_audio.data / format is not a string, the buffered ParseContent panics"
			}
			out = append(out, contentPart{Type: "input_audio", InputAudio: &contentInputAudio{Data: d, Format: f}})
		case "file":
			var sub map[string]json.RawMessage
			if err := json.Unmarshal(m["file"], &sub); err != nil || sub == nil {
				continue
			}
			id, ok := jsonUnquote(sub["file_id"])
			if !ok {
				return nil, "file.file_id is not a string, the buffered ParseContent panics"
			}
			out = append(out, contentPart{Type: "file", File: &contentFile{FileID: id}})
		}
	}
	return out, ""
}
