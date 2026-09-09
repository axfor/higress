package streamxform

import (
	"encoding/json"
	"errors"
)

// Streaming conversion OpenAI → Cohere v1 chat.
//
// Derived from the buffered cohere.go buildCohereRequest: the request is rebuilt from scratch with message = the
// StringContent of the FIRST message (a string as is, an array joins its text parts each followed by "\n"), the
// mapped model, and the scalars under Cohere's names (n → k, top_p → p, stop → stop_sequences). Every other field
// of the request, the remaining messages included, is dropped. No messages at all makes the buffered path send
// the literal null; that is left to it.

type CohereOptions struct {
	// MapModel reproduces the buffered mapModel: an error when model is empty or maps to empty.
	MapModel func(model string) (string, error)
}

type cohereProto struct {
	opt CohereOptions

	model      string
	modelSeen  bool
	stream     bool
	streamSeen bool
	maxTok     int
	n, seed    int
	temp, topP float64
	freq, pres float64
	stop       []string

	messagesSeen bool
	msgs         int
	message      string
}

// NewCohere builds the OpenAI → Cohere chat transformer.
func NewCohere(opt CohereOptions) *Transformer {
	if opt.MapModel == nil {
		opt.MapModel = func(m string) (string, error) {
			if m == "" {
				return "", errors.New("missing model in request")
			}
			return m, nil
		}
	}
	p := &cohereProto{opt: opt}
	t := NewTransformer(p)
	t.DupKeyBail = true // buffered struct decoding is last-wins, which streaming cannot reproduce: fall back
	return t
}

func (p *cohereProto) Prelude() Prelude {
	return Prelude{Model: p.model, ModelSeen: p.modelSeen, Stream: p.stream, StreamSeen: p.streamSeen}
}

func (p *cohereProto) OnKey(t *Transformer) Action {
	switch t.Depth() {
	case 1:
		switch t.Last() {
		case "model":
			return Capture(4 << 10)
		case "stream":
			return Capture(16)
		case "max_tokens", "n", "seed", "temperature", "top_p", "frequency_penalty", "presence_penalty":
			return Capture(64)
		case "stop":
			return Capture(64 << 10)
		case "messages":
			return Probe()
		}
		return Skip() // buildCohereRequest reads nothing else
	case 3:
		if t.Last() == "content" && p.msgs == 1 {
			return Capture(systemCap) // the first message's content is needed whole; the cap matches the buffered body limit
		}
		return Skip()
	}
	return Bail("unexpected path: " + t.PathString())
}

func (p *cohereProto) OnElem(t *Transformer) Action {
	if t.Depth() == 2 {
		return Probe()
	}
	return Bail("unexpected array: " + t.PathString())
}

func (p *cohereProto) OnStart(t *Transformer, kind ValueKind) Action {
	switch t.Depth() {
	case 1: // messages
		if kind != KindArray {
			return Bail("messages is not an array, the buffered struct decoding fails")
		}
		p.messagesSeen = true
		return Enter().Lazy() // nothing of the input array reaches the output
	case 2:
		if kind != KindObject {
			return Bail("message is not an object, the buffered struct decoding fails")
		}
		p.msgs++
		if p.msgs == 1 {
			return Enter().Lazy()
		}
		return Skip() // only the first message is read
	}
	return Bail("unexpected Probe: " + t.PathString())
}

func (p *cohereProto) OnValue(t *Transformer, raw []byte) {
	isNull := string(raw) == "null"
	switch t.Depth() {
	case 1:
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
		case "max_tokens", "n", "seed":
			if isNull {
				return
			}
			if !isIntLiteral(raw) {
				t.Bail(t.Last() + " is not an integer")
				return
			}
			v := atoi(raw)
			switch t.Last() {
			case "max_tokens":
				p.maxTok = v
			case "n":
				p.n = v
			default:
				p.seed = v
			}
		case "temperature", "top_p", "frequency_penalty", "presence_penalty":
			if isNull {
				return
			}
			if !isNumLiteral(raw) {
				t.Bail(t.Last() + " is not a number")
				return
			}
			f, _ := parseFloat(raw)
			switch t.Last() {
			case "temperature":
				p.temp = f
			case "top_p":
				p.topP = f
			case "frequency_penalty":
				p.freq = f
			default:
				p.pres = f
			}
		case "stop":
			if isNull {
				return
			}
			if err := json.Unmarshal(raw, &p.stop); err != nil {
				t.Bail("stop is not an array of strings")
			}
		}
	case 3: // the first message's content
		p.message = stringContent(t, raw)
	}
}

func (p *cohereProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	return Bail("unexpected prefix: " + t.PathString()), 0
}

func (p *cohereProto) OnLeave(t *Transformer) {}

func (p *cohereProto) Tail(t *Transformer) {
	if !p.messagesSeen || p.msgs == 0 {
		t.Bail("no message found in the request body") // the buffered path sends the literal null
		return
	}
	mapped, err := p.opt.MapModel(p.model)
	if err != nil {
		t.Bail(err.Error())
		return
	}
	// omitempty throughout, as the buffered cohereTextGenRequest
	w := t.W()
	if p.message != "" {
		w.Key("message")
		w.JSONString(p.message)
	}
	if mapped != "" {
		w.Key("model")
		w.JSONString(mapped)
	}
	if p.stream {
		w.Key("stream")
		w.RawString("true")
	}
	if p.maxTok != 0 {
		w.Key("max_tokens")
		w.Int(p.maxTok)
	}
	num := func(key string, f float64) {
		if f != 0 {
			b, _ := json.Marshal(f)
			w.Key(key)
			w.Raw(b)
		}
	}
	num("temperature", p.temp)
	if p.n != 0 {
		w.Key("k")
		w.Int(p.n)
	}
	num("p", p.topP)
	if p.seed != 0 {
		w.Key("seed")
		w.Int(p.seed)
	}
	if len(p.stop) > 0 {
		b, _ := json.Marshal(p.stop)
		w.Key("stop_sequences")
		w.Raw(b)
	}
	num("frequency_penalty", p.freq)
	num("presence_penalty", p.pres)
}
