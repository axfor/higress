package streamxform

// Three conversions that rebuild the request from the messages' text and a few scalars, each message taken whole
// (StringContent), bounded by the same cap as the buffered body: MiniMax Pro, Dify and Triton. They share the
// message walk below; what differs is what each writes for a message and in Tail.

import (
	"encoding/json"
	"errors"

	"github.com/axfor/ason"
)

// jsonEscapeBody returns the JSON encoding of s without the surrounding quotes, so pieces can be concatenated into
// one string value written across several writes.
func jsonEscapeBody(s string) []byte {
	b := ason.AppendJSONString(nil, s)
	return b[1 : len(b)-1]
}

type collectMsg struct {
	role, name, id string
	content        string
	roleSeen       bool
}

// collectProto is the shared skeleton: top-level scalars are captured by the concrete protocol's keys, every message
// is entered lazily with role / name / id / content captured, and the concrete protocol is told when it closes.
type collectProto struct {
	topKeys map[string]int // key → capture cap
	msgs    int
	m       collectMsg
	onTop   func(t *Transformer, key string, raw []byte)
	onMsg   func(t *Transformer, m *collectMsg)
	msgKeys map[string]int
}

func (p *collectProto) OnKey(t *Transformer) Action {
	switch t.Depth() {
	case 1:
		if t.Last() == "messages" {
			return Probe()
		}
		if n, ok := p.topKeys[t.Last()]; ok {
			return Capture(n)
		}
		return Skip()
	case 3:
		if n, ok := p.msgKeys[t.Last()]; ok {
			return Capture(n)
		}
		return Skip()
	}
	return Bail("unexpected path: " + t.PathString())
}

func (p *collectProto) OnElem(t *Transformer) Action {
	if t.Depth() == 2 {
		return Probe()
	}
	return Bail("unexpected array: " + t.PathString())
}

func (p *collectProto) OnStart(t *Transformer, kind ValueKind) Action {
	switch t.Depth() {
	case 1:
		if kind != KindArray {
			return Bail("messages is not an array, the buffered struct decoding fails")
		}
		return Enter().Lazy()
	case 2:
		if kind != KindObject {
			return Bail("message is not an object, the buffered struct decoding fails")
		}
		p.msgs++
		p.m = collectMsg{}
		return Enter().Lazy()
	}
	return Bail("unexpected Probe: " + t.PathString())
}

func (p *collectProto) OnValue(t *Transformer, raw []byte) {
	switch t.Depth() {
	case 1:
		p.onTop(t, t.Last(), raw)
	case 3:
		switch t.Last() {
		case "role", "name", "id":
			s, ok := jsonUnquote(raw)
			if !ok && string(raw) != "null" {
				t.Bail(t.Last() + " is not a string, the buffered struct decoding fails")
				return
			}
			switch t.Last() {
			case "role":
				p.m.role, p.m.roleSeen = s, true
			case "name":
				p.m.name = s
			default:
				p.m.id = s
			}
		case "content":
			p.m.content = stringContent(t, raw)
		}
	}
}

func (p *collectProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	return Bail("unexpected prefix: " + t.PathString()), 0
}

func (p *collectProto) OnLeave(t *Transformer) {
	if t.Depth() == 2 {
		p.onMsg(t, &p.m)
	}
}

func (p *collectProto) requireMessages(t *Transformer) bool {
	if p.msgs == 0 {
		t.Bail("no message found in the request body") // decodeChatCompletionRequest rejects an empty or missing messages
		return false
	}
	return true
}

var collectMsgKeys = map[string]int{"role": 256, "name": 4 << 10, "id": 4 << 10, "content": systemCap}

// strictMapper wraps a strict MapModel with the "missing model" default when nil.
func strictMapper(f func(string) (string, error)) func(string) (string, error) {
	if f != nil {
		return f
	}
	return func(m string) (string, error) {
		if m == "" {
			return "", errors.New("missing model in request")
		}
		return m, nil
	}
}

// ---- MiniMax Pro (chatcompletion_pro) ----
//
// Buffered buildMinimaxChatCompletionProRequest: system messages become bot_setting entries (name or the default bot
// name), assistant and user messages become {sender_type, sender_name, text}, other roles are dropped;
// reply_constraints names the last system message's bot; without any bot_setting a default one is added. model is
// mapped leniently and always written, mask_sensitive_info is always true.

type MiniMaxProOptions struct {
	// MapModel reproduces getMappedModel: returns the input unchanged when no mapping matches, never fails.
	MapModel func(model string) string
	// The provider's constants: default bot / sender names, the default bot setting content, the two sender types.
	DefaultBotName, DefaultSenderName, DefaultBotSettingContent string
	SenderTypeBot, SenderTypeUser                               string
}

type minimaxBotSettingOut struct {
	BotName string `json:"bot_name"`
	Content string `json:"content"`
}

type minimaxProProto struct {
	collectProto
	opt MiniMaxProOptions

	model      string
	modelSeen  bool
	stream     bool
	streamSeen bool
	maxTok     int
	temp, topP float64
	botName    string
	bots       []minimaxBotSettingOut
	written    int
}

// NewMiniMaxPro builds the OpenAI → MiniMax Pro transformer.
func NewMiniMaxPro(opt MiniMaxProOptions) *Transformer {
	if opt.MapModel == nil {
		opt.MapModel = func(m string) string { return m }
	}
	p := &minimaxProProto{opt: opt}
	p.collectProto = collectProto{
		topKeys: map[string]int{"model": 4 << 10, "stream": 16, "max_tokens": 64, "temperature": 64, "top_p": 64},
		msgKeys: collectMsgKeys,
		onTop:   p.top,
		onMsg:   p.msg,
	}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

func (p *minimaxProProto) Prelude() Prelude {
	return Prelude{Model: p.model, ModelSeen: p.modelSeen, Stream: p.stream, StreamSeen: p.streamSeen}
}

func (p *minimaxProProto) OnStart(t *Transformer, kind ValueKind) Action {
	if t.Depth() == 1 {
		if kind != KindArray {
			return Bail("messages is not an array, the buffered struct decoding fails")
		}
		return Enter().As("messages").Lazy() // nil slice on the buffered path when nothing lands: written as null in Tail
	}
	return p.collectProto.OnStart(t, kind)
}

func (p *minimaxProProto) top(t *Transformer, key string, raw []byte) {
	isNull := string(raw) == "null"
	switch key {
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
	case "max_tokens":
		if isNull {
			return
		}
		if !isIntLiteral(raw) {
			t.Bail("max_tokens is not an integer")
			return
		}
		p.maxTok = atoi(raw)
	case "temperature", "top_p":
		if isNull {
			return
		}
		if !isNumLiteral(raw) {
			t.Bail(key + " is not a number")
			return
		}
		f, _ := parseFloat(raw)
		if key == "temperature" {
			p.temp = f
		} else {
			p.topP = f
		}
	}
}

func (p *minimaxProProto) msg(t *Transformer, m *collectMsg) {
	name := func(n, def string) string {
		if n != "" {
			return n
		}
		return def
	}
	w := t.W()
	switch m.role {
	case "system":
		p.botName = name(m.name, p.opt.DefaultBotName)
		p.bots = append(p.bots, minimaxBotSettingOut{BotName: p.botName, Content: m.content})
	case "assistant":
		p.written++
		w.Key("sender_type")
		w.JSONString(p.opt.SenderTypeBot)
		w.Key("sender_name")
		w.JSONString(name(m.name, p.opt.DefaultBotName))
		w.Key("text")
		w.JSONString(m.content)
	case "user":
		p.written++
		w.Key("sender_type")
		w.JSONString(p.opt.SenderTypeUser)
		w.Key("sender_name")
		w.JSONString(name(m.name, p.opt.DefaultSenderName))
		w.Key("text")
		w.JSONString(m.content)
	}
}

func (p *minimaxProProto) Tail(t *Transformer) {
	if !p.requireMessages(t) {
		return
	}
	w := t.W()
	w.Key("model")
	w.JSONString(p.opt.MapModel(p.model))
	if p.stream {
		w.Key("stream")
		w.RawString("true")
	}
	if p.maxTok != 0 {
		w.Key("tokens_to_generate")
		w.Int(p.maxTok)
	}
	for _, kv := range []struct {
		k string
		v float64
	}{{"temperature", p.temp}, {"top_p", p.topP}} {
		if kv.v != 0 {
			b, _ := json.Marshal(kv.v)
			w.Key(kv.k)
			w.Raw(b)
		}
	}
	w.Key("mask_sensitive_info")
	w.RawString("true")
	if p.written == 0 {
		w.Key("messages")
		w.RawString("null")
	}
	bots := p.bots
	if len(bots) == 0 {
		bots = []minimaxBotSettingOut{{BotName: p.opt.DefaultBotName, Content: p.opt.DefaultBotSettingContent}}
	}
	b, _ := json.Marshal(bots)
	w.Key("bot_setting")
	w.Raw(b)
	botName := p.botName
	if botName == "" {
		botName = p.opt.DefaultBotName
	}
	rc, _ := json.Marshal(map[string]string{"sender_type": p.opt.SenderTypeBot, "sender_name": botName})
	w.Key("reply_constraints")
	w.Raw(rc)
}

// ---- Dify ----
//
// Buffered difyChatGenRequest: every message's StringContent is appended to one text under a "SYSTEM: " /
// "ASSISTANT: " / "USER: " heading (developer counts as system: handleRequestBody converts it first); where that
// text lands depends on the bot type. response_mode follows stream, user defaults to "api-user", conversation_id
// comes from the ConversationId request header. The text is written as one JSON string across the messages, so a
// long conversation never has to be held.

type DifyOptions struct {
	// MapModel reproduces the buffered mapModel: an error when model is empty or maps to empty.
	MapModel func(model string) (string, error)
	// BotType mirrors the provider setting botType (Chat / Agent / Completion / Workflow; anything else writes an empty request).
	BotType string
	// InputVariable mirrors the provider setting inputVariable (Workflow).
	InputVariable string
	// ConversationId is the ConversationId request header (Chat / Agent).
	ConversationId string
}

type difyProto struct {
	collectProto
	opt DifyOptions

	model      string
	modelSeen  bool
	stream     bool
	streamSeen bool
	user       string
	opened     bool
}

// NewDify builds the OpenAI → Dify transformer.
func NewDify(opt DifyOptions) *Transformer {
	opt.MapModel = strictMapper(opt.MapModel)
	p := &difyProto{opt: opt}
	p.collectProto = collectProto{
		topKeys: map[string]int{"model": 4 << 10, "stream": 16, "user": 4 << 10},
		msgKeys: collectMsgKeys,
		onTop:   p.top,
		onMsg:   p.msg,
	}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

func (p *difyProto) Prelude() Prelude {
	return Prelude{Model: p.model, ModelSeen: p.modelSeen, Stream: p.stream, StreamSeen: p.streamSeen}
}

func (p *difyProto) textKind() (usesText bool, inInputs bool, key string) {
	switch p.opt.BotType {
	case "Chat", "Agent":
		return true, false, "query"
	case "Completion":
		return true, true, "query"
	case "Workflow":
		return true, true, p.opt.InputVariable
	}
	return false, false, ""
}

func (p *difyProto) OnStart(t *Transformer, kind ValueKind) Action {
	if t.Depth() == 1 {
		if kind != KindArray {
			return Bail("messages is not an array, the buffered struct decoding fails")
		}
		uses, inInputs, key := p.textKind()
		if uses {
			w := t.W()
			if inInputs {
				w.Key("inputs")
				w.RawString(`{`)
				w.Raw(ason.AppendJSONString(nil, key))
				w.RawString(`:"`)
			} else {
				w.Key(key)
				w.Byte('"')
			}
			p.opened = true
		}
		return Enter().Lazy()
	}
	return p.collectProto.OnStart(t, kind)
}

func (p *difyProto) top(t *Transformer, key string, raw []byte) {
	isNull := string(raw) == "null"
	switch key {
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
	case "user":
		s, ok := jsonUnquote(raw)
		if !ok && !isNull {
			t.Bail("user is not a string")
			return
		}
		p.user = s
	}
}

func (p *difyProto) msg(t *Transformer, m *collectMsg) {
	if !p.opened {
		return
	}
	heading := "USER: \n"
	switch m.role {
	case "system", "developer":
		heading = "SYSTEM: \n"
	case "assistant":
		heading = "ASSISTANT: \n"
	}
	t.W().Raw(jsonEscapeBody(heading + m.content + "\n"))
}

func (p *difyProto) OnLeave(t *Transformer) {
	if t.Depth() == 1 && p.opened {
		_, inInputs, _ := p.textKind()
		if inInputs {
			t.W().RawString(`"}`)
		} else {
			t.W().Byte('"')
		}
		return
	}
	p.collectProto.OnLeave(t)
}

func (p *difyProto) Tail(t *Transformer) {
	if !p.requireMessages(t) {
		return
	}
	if _, err := p.opt.MapModel(p.model); err != nil {
		t.Bail(err.Error())
		return
	}
	w := t.W()
	uses, inInputs, _ := p.textKind()
	if !uses { // unknown bot type: an empty DifyChatRequest
		for _, kv := range [][2]string{{"inputs", "null"}, {"query", `""`}, {"response_mode", `""`}, {"user", `""`}, {"auto_generate_name", "false"}, {"conversation_id", `""`}} {
			w.Key(kv[0])
			w.RawString(kv[1])
		}
		return
	}
	mode := "blocking"
	if p.stream {
		mode = "streaming"
	}
	user := p.user
	if user == "" {
		user = "api-user"
	}
	if inInputs {
		w.Key("query")
		w.RawString(`""`)
	} else {
		w.Key("inputs")
		w.RawString("{}")
	}
	w.Key("response_mode")
	w.JSONString(mode)
	w.Key("user")
	w.JSONString(user)
	w.Key("auto_generate_name")
	w.RawString("false")
	w.Key("conversation_id")
	if inInputs {
		w.RawString(`""`)
	} else {
		w.JSONString(p.opt.ConversationId)
	}
}

// ---- Triton ----
//
// Buffered BuildTritonTexGenRequest: id and text_input come from the last message (its id and StringContent, so a
// message without content leaves text_input empty); parameters carries stream and temperature, both always written.
// Only the last message is kept, so memory is one message, not the conversation.

type TritonOptions struct {
	// MapModel reproduces the buffered mapModel: an error when model is empty or maps to empty.
	MapModel func(model string) (string, error)
}

type tritonProto struct {
	collectProto
	opt TritonOptions

	model      string
	modelSeen  bool
	stream     bool
	streamSeen bool
	temp       float64
	lastID     string
	lastText   string
}

// NewTriton builds the OpenAI → Triton generate transformer.
func NewTriton(opt TritonOptions) *Transformer {
	opt.MapModel = strictMapper(opt.MapModel)
	p := &tritonProto{opt: opt}
	p.collectProto = collectProto{
		topKeys: map[string]int{"model": 4 << 10, "stream": 16, "temperature": 64},
		msgKeys: collectMsgKeys,
		onTop:   p.top,
		onMsg:   p.msg,
	}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

func (p *tritonProto) Prelude() Prelude {
	return Prelude{Model: p.model, ModelSeen: p.modelSeen, Stream: p.stream, StreamSeen: p.streamSeen}
}

func (p *tritonProto) top(t *Transformer, key string, raw []byte) {
	isNull := string(raw) == "null"
	switch key {
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
	case "temperature":
		if isNull {
			return
		}
		if !isNumLiteral(raw) {
			t.Bail("temperature is not a number")
			return
		}
		p.temp, _ = parseFloat(raw)
	}
}

func (p *tritonProto) msg(t *Transformer, m *collectMsg) {
	p.lastID, p.lastText = m.id, m.content
}

func (p *tritonProto) Tail(t *Transformer) {
	if !p.requireMessages(t) {
		return
	}
	if _, err := p.opt.MapModel(p.model); err != nil {
		t.Bail(err.Error())
		return
	}
	w := t.W()
	w.Key("id")
	w.JSONString(p.lastID)
	w.Key("text_input")
	w.JSONString(p.lastText)
	temp, _ := json.Marshal(p.temp)
	w.Key("parameters")
	w.RawString(`{"stream":`)
	if p.stream {
		w.RawString("true")
	} else {
		w.RawString("false")
	}
	w.RawString(`,"temperature":`)
	w.Raw(temp)
	w.Byte('}')
}
