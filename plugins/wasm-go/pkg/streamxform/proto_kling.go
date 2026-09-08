package streamxform

// Streaming rewrite OpenAI /v1/videos → Kling video creation.
//
// Derived from the buffered kling.go transformOpenAIVideoRequest: the body passes through; model (or, failing
// that, model_name) is read, mapped and written back as model_name, and model itself is removed. Whether the
// request is image-to-video -- which decides the upstream path -- is the presence of any of Kling's image input
// keys at the top level; the integration layer asks after the whole body has been seen, or falls back when the
// body outgrows the window before that.

type KlingOptions struct {
	// MapModel reproduces getMappedModel: returns the input unchanged when no mapping matches, never fails.
	MapModel func(model string) string
}

// klingImageFields mirrors isImageToVideoRequest; keep in sync with kling.go.
var klingImageFields = map[string]bool{
	"image": true, "image_url": true, "image_urls": true, "images": true, "image_tail": true,
	"image_tail_url": true, "input_image": true, "first_frame_image": true, "last_frame_image": true,
}

type klingProto struct {
	opt KlingOptions

	model     string
	modelSeen bool
	name      string
	nameSeen  bool
	imageSeen bool
}

// NewKling builds the OpenAI videos → Kling transformer.
func NewKling(opt KlingOptions) *Transformer {
	if opt.MapModel == nil {
		opt.MapModel = func(m string) string { return m }
	}
	p := &klingProto{opt: opt}
	t := NewTransformer(p)
	t.DupKeyBail = true // gjson reads the first key, sjson rewrites the first: a duplicate model_name would end up in a different place
	return t
}

func (p *klingProto) Prelude() Prelude {
	switch {
	case p.modelSeen:
		return Prelude{Model: p.model, ModelSeen: true}
	case p.nameSeen:
		return Prelude{Model: p.name, ModelSeen: true}
	}
	return Prelude{}
}

// ImageToVideo reports whether an image input key was seen at the top level (isImageToVideoRequest).
func (p *klingProto) ImageToVideo() bool { return p.imageSeen }

func (p *klingProto) OnKey(t *Transformer) Action {
	if t.Depth() == 1 {
		switch k := t.Last(); {
		case k == "model", k == "model_name":
			return Capture(4 << 10) // dropped here, model_name is rewritten in Tail
		case klingImageFields[k]:
			p.imageSeen = true
		}
	}
	return Pass()
}

func (p *klingProto) OnElem(t *Transformer) Action                  { return Pass() }
func (p *klingProto) OnStart(t *Transformer, kind ValueKind) Action { return Pass() }
func (p *klingProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	return Bail("unexpected prefix: " + t.PathString()), 0
}
func (p *klingProto) OnLeave(t *Transformer) {}

func (p *klingProto) OnValue(t *Transformer, raw []byte) {
	if t.Depth() != 1 {
		return
	}
	s, ok := jsonUnquote(raw)
	if !ok {
		t.Bail(t.Last() + " is not a string") // gjson's String() would render the literal; rare enough to leave to the buffered path
		return
	}
	if t.Last() == "model" {
		p.model, p.modelSeen = s, true
	} else {
		p.name, p.nameSeen = s, true
	}
}

func (p *klingProto) Tail(t *Transformer) {
	pre := p.Prelude()
	if !pre.ModelSeen {
		return // neither key: the body goes through untouched
	}
	w := t.W()
	w.Key("model_name")
	w.JSONString(p.opt.MapModel(pre.Model))
}
