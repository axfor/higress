package streamxform

import (
	"encoding/json"
)

// Streaming rewrites of the small non-chat endpoints: embeddings for Gemini, Vertex and native Qwen, and Gemini
// image generation. The buffered builders read model and input (or prompt / n) and nothing else.
//
// input is a string or an array of strings (ParseInput drops other elements; native Qwen rejects them). An array
// streams element by element; a string is captured whole, bounded. Gemini names the model inside every element,
// so an array that arrives before model is held (bounded) and replayed once model is known.

type EmbeddingsOptions struct {
	// MapModel reproduces the buffered mapModel: an error when model is empty or maps to empty.
	MapModel func(model string) (string, error)
}

type embeddingsShape int

const (
	embGemini embeddingsShape = iota // requests:[{model:"models/<m>",content:{parts:[{text}]}}]
	embVertex                        // instances:[{task_type:"",content}]
	embQwen                          // model, input:{texts:[...]}, parameters:{}
)

const embeddingsInputCap = 1 << 20

type embeddingsProto struct {
	shape embeddingsShape
	opt   EmbeddingsOptions

	model      string
	modelSeen  bool
	mapped     string
	inputArray bool
	strSeen    bool
	str        string
}

func newEmbeddings(shape embeddingsShape, opt EmbeddingsOptions) *Transformer {
	opt.MapModel = strictMapper(opt.MapModel)
	p := &embeddingsProto{shape: shape, opt: opt}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

// NewGeminiEmbeddings builds the OpenAI → Gemini batchEmbedContents transformer.
func NewGeminiEmbeddings(opt EmbeddingsOptions) *Transformer { return newEmbeddings(embGemini, opt) }

// NewVertexEmbeddings builds the OpenAI → Vertex predict (embeddings) transformer.
func NewVertexEmbeddings(opt EmbeddingsOptions) *Transformer { return newEmbeddings(embVertex, opt) }

// NewQwenEmbeddings builds the OpenAI → native DashScope text-embedding transformer.
func NewQwenEmbeddings(opt EmbeddingsOptions) *Transformer { return newEmbeddings(embQwen, opt) }

func (p *embeddingsProto) Prelude() Prelude { return Prelude{Model: p.model, ModelSeen: p.modelSeen} }

func (p *embeddingsProto) OnKey(t *Transformer) Action {
	if t.Depth() != 1 {
		return Bail("unexpected path: " + t.PathString())
	}
	switch t.Last() {
	case "model":
		return Capture(4 << 10)
	case "input":
		return Probe()
	}
	return Skip() // encoding_format / dimensions / user: not read by the buffered builders
}

func (p *embeddingsProto) OnElem(t *Transformer) Action {
	if t.Depth() == 2 {
		if p.shape == embQwen {
			return Probe() // strings pass verbatim, anything else fails the buffered build
		}
		return Probe()
	}
	return Bail("unexpected array: " + t.PathString())
}

func (p *embeddingsProto) OnStart(t *Transformer, kind ValueKind) Action {
	w := t.W()
	switch t.Depth() {
	case 1: // input
		switch kind {
		case KindString:
			return Capture(embeddingsInputCap)
		case KindArray:
			if p.shape == embGemini && !p.modelSeen {
				return Defer(embeddingsInputCap) // every element names the model
			}
			p.inputArray = true
			switch p.shape {
			case embGemini:
				return Enter().As("requests")
			case embVertex:
				return Enter().As("instances")
			default:
				w.PushObj("input")
				w.PushArr("texts")
				return Enter().Flat()
			}
		}
		if p.shape == embQwen {
			return Bail("input is neither a string nor an array, the buffered build fails")
		}
		return Skip() // ParseInput yields nothing: an empty array
	case 2: // input[i]
		if kind != KindString {
			if p.shape == embQwen {
				return Bail("input element is not a string, the buffered build fails")
			}
			return Skip() // ParseInput drops it
		}
		if p.shape == embQwen {
			return Pass()
		}
		return Prefix(1)
	}
	return Bail("unexpected Probe: " + t.PathString())
}

func (p *embeddingsProto) OnValue(t *Transformer, raw []byte) {
	if t.Depth() != 1 {
		return
	}
	switch t.Last() {
	case "model":
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("model is not a string")
			return
		}
		p.model, p.modelSeen = s, true
		mapped, err := p.opt.MapModel(s)
		if err != nil {
			t.Bail(err.Error())
			return
		}
		p.mapped = mapped
		if len(t.Deferred()) > 0 {
			t.Release()
		}
	case "input":
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("invalid input string")
			return
		}
		p.str, p.strSeen = s, true
	}
}

// OnPrefix: one string element of the input array (Gemini, Vertex), wrapped in its element shape.
func (p *embeddingsProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	w := t.W()
	w.Elem()
	switch p.shape {
	case embGemini:
		w.RawString(`{"model":`)
		w.JSONString("models/" + p.mapped)
		w.RawString(`,"content":{"parts":[`)
		if complete && len(raw) == 0 {
			w.RawString(`{}]}}`) // Text has omitempty
			return Skip(), 0
		}
		w.RawString(`{"text":"`)
		return Pass().Wrap(nil, []byte(`"}]}}`)), 0
	default:
		w.RawString(`{"task_type":"","content":"`)
		return Pass().Wrap(nil, []byte(`"}`)), 0
	}
}

func (p *embeddingsProto) OnLeave(t *Transformer) {
	if t.Depth() == 1 && p.inputArray && p.shape == embQwen {
		w := t.W()
		w.Open() // an empty texts array is still written
		w.Pop()
		w.Pop()
	}
}

func (p *embeddingsProto) Tail(t *Transformer) {
	w := t.W()
	if !p.modelSeen {
		if _, err := p.opt.MapModel(""); err != nil {
			t.Bail(err.Error())
			return
		}
	}
	switch p.shape {
	case embGemini:
		if p.strSeen {
			part := map[string]string{}
			if p.str != "" {
				part["text"] = p.str
			}
			b, _ := json.Marshal([]map[string]any{{"model": "models/" + p.mapped, "content": map[string]any{"parts": []map[string]string{part}}}})
			w.Key("requests")
			w.Raw(b)
		} else if !p.inputArray {
			w.Key("requests")
			w.RawString("[]")
		}
	case embVertex:
		if p.strSeen {
			b, _ := json.Marshal([]map[string]string{{"task_type": "", "content": p.str}})
			w.Key("instances")
			w.Raw(b)
		} else if !p.inputArray {
			w.Key("instances")
			w.RawString("[]")
		}
	default:
		w.Key("model")
		w.JSONString(p.mapped)
		if p.strSeen {
			b, _ := json.Marshal([]string{p.str})
			w.Key("input")
			w.RawString(`{"texts":`)
			w.Raw(b)
			w.Byte('}')
		} else if !p.inputArray {
			t.Bail("input missing, the buffered build fails")
			return
		}
		w.Key("parameters")
		w.RawString("{}")
	}
}

// ---- Gemini image generation ----
//
// Buffered buildGeminiImageGenerationRequest: instances:[{prompt}], parameters:{sampleCount:n}.

type geminiImageProto struct {
	opt       EmbeddingsOptions
	model     string
	modelSeen bool
	mapped    string
	prompt    string
	n         int
}

// NewGeminiImage builds the OpenAI images/generations → Gemini predict transformer.
func NewGeminiImage(opt EmbeddingsOptions) *Transformer {
	opt.MapModel = strictMapper(opt.MapModel)
	p := &geminiImageProto{opt: opt}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

func (p *geminiImageProto) Prelude() Prelude { return Prelude{Model: p.model, ModelSeen: p.modelSeen} }

func (p *geminiImageProto) OnKey(t *Transformer) Action {
	if t.Depth() != 1 {
		return Bail("unexpected path: " + t.PathString())
	}
	switch t.Last() {
	case "model":
		return Capture(4 << 10)
	case "prompt":
		return Capture(embeddingsInputCap)
	case "n":
		return Capture(64)
	}
	return Skip()
}

func (p *geminiImageProto) OnElem(t *Transformer) Action {
	return Bail("unexpected array: " + t.PathString())
}
func (p *geminiImageProto) OnStart(t *Transformer, kind ValueKind) Action {
	return Bail("unexpected Probe: " + t.PathString())
}
func (p *geminiImageProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	return Bail("unexpected prefix: " + t.PathString()), 0
}
func (p *geminiImageProto) OnLeave(t *Transformer) {}

func (p *geminiImageProto) OnValue(t *Transformer, raw []byte) {
	isNull := string(raw) == "null"
	switch t.Last() {
	case "model":
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("model is not a string")
			return
		}
		p.model, p.modelSeen = s, true
		mapped, err := p.opt.MapModel(s)
		if err != nil {
			t.Bail(err.Error())
			return
		}
		p.mapped = mapped
	case "prompt":
		if isNull {
			return
		}
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("prompt is not a string")
			return
		}
		p.prompt = s
	case "n":
		if isNull {
			return
		}
		if !isIntLiteral(raw) {
			t.Bail("n is not an integer")
			return
		}
		p.n = atoi(raw)
	}
}

func (p *geminiImageProto) Tail(t *Transformer) {
	if !p.modelSeen {
		if _, err := p.opt.MapModel(""); err != nil {
			t.Bail(err.Error())
			return
		}
	}
	w := t.W()
	b, _ := json.Marshal([]map[string]string{{"prompt": p.prompt}})
	w.Key("instances")
	w.Raw(b)
	w.Key("parameters")
	if p.n != 0 {
		w.RawString(`{"sampleCount":`)
		w.Int(p.n)
		w.Byte('}')
	} else {
		w.RawString("{}")
	}
}
