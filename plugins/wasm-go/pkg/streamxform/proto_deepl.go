package streamxform

import (
	"encoding/json"
)

// Streaming conversion OpenAI → DeepL text translation.
//
// Derived from the buffered deepl.go deeplTextGenRequest: every non-system message's StringContent becomes one
// entry of text, the (last) system message's StringContent becomes context, target_lang comes from the provider
// setting. Nothing else of the request is read. model is not mapped; it only selects the host and must be
// "Free" or "Pro", anything else fails on the buffered path and falls back here.

type DeepLOptions struct {
	// TargetLang mirrors the provider setting targetLang.
	TargetLang string
}

type deeplMsg struct {
	role    string
	content string
}

type deeplProto struct {
	opt DeepLOptions

	model     string
	modelSeen bool
	msgs      int
	texts     []string
	context   string
	m         deeplMsg
}

// NewDeepL builds the OpenAI → DeepL transformer.
func NewDeepL(opt DeepLOptions) *Transformer {
	p := &deeplProto{opt: opt, texts: []string{}}
	t := NewTransformer(p)
	t.DupKeyBail = true // buffered struct decoding is last-wins, which streaming cannot reproduce: fall back
	return t
}

func (p *deeplProto) Prelude() Prelude {
	return Prelude{Model: p.model, ModelSeen: p.modelSeen}
}

func (p *deeplProto) OnKey(t *Transformer) Action {
	switch t.Depth() {
	case 1:
		switch t.Last() {
		case "model":
			return Capture(4 << 10)
		case "messages":
			return Probe()
		}
		return Skip() // deeplTextGenRequest reads nothing else
	case 3:
		switch t.Last() {
		case "role":
			return Capture(256)
		case "content":
			return Capture(systemCap) // needed whole; the cap matches the buffered body limit
		}
		return Skip()
	}
	return Bail("unexpected path: " + t.PathString())
}

func (p *deeplProto) OnElem(t *Transformer) Action {
	if t.Depth() == 2 {
		return Probe()
	}
	return Bail("unexpected array: " + t.PathString())
}

func (p *deeplProto) OnStart(t *Transformer, kind ValueKind) Action {
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
		p.m = deeplMsg{}
		return Enter().Lazy()
	}
	return Bail("unexpected Probe: " + t.PathString())
}

func (p *deeplProto) OnValue(t *Transformer, raw []byte) {
	switch t.Depth() {
	case 1: // model
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("model is not a string")
			return
		}
		p.model, p.modelSeen = s, true
		if s != "Free" && s != "Pro" {
			t.Bail(`deepl model should be "Free" or "Pro"`) // the buffered path fails the request the same way
		}
	case 3:
		switch t.Last() {
		case "role":
			s, ok := jsonUnquote(raw)
			if !ok && string(raw) != "null" {
				t.Bail("role is not a string")
				return
			}
			p.m.role = s
		case "content":
			p.m.content = stringContent(t, raw)
		}
	}
}

func (p *deeplProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	return Bail("unexpected prefix: " + t.PathString()), 0
}

func (p *deeplProto) OnLeave(t *Transformer) {
	if t.Depth() == 2 {
		if p.m.role == "system" || p.m.role == "developer" { // handleRequestBody turns developer into system before the decode
			p.context = p.m.content
		} else {
			p.texts = append(p.texts, p.m.content)
		}
	}
}

func (p *deeplProto) Tail(t *Transformer) {
	if p.msgs == 0 {
		t.Bail("no message found in the request body") // decodeChatCompletionRequest rejects an empty or missing messages
		return
	}
	if !p.modelSeen {
		t.Bail(`deepl model should be "Free" or "Pro"`) // overwriteRequestHost fails on an empty model
		return
	}
	w := t.W()
	b, _ := json.Marshal(p.texts) // never nil: the buffered path starts from an empty slice, so no messages gives []
	w.Key("text")
	w.Raw(b)
	w.Key("target_lang")
	w.JSONString(p.opt.TargetLang)
	if p.context != "" {
		w.Key("context")
		w.JSONString(p.context)
	}
}
