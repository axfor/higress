package streamxform

import (
	"bytes"
	"encoding/json"
	"strings"
)

// OpenAI images (generations / edits / variations) → Vertex generateContent: the streaming form of
// buildVertexImageRequest.
//
// The buffered path reads prompt, size, output_format and (for edits and variations) the image inputs, and rebuilds
// the request as one user turn whose parts are the images followed by the prompt text, plus the configured safety
// settings and a fixed generationConfig. The image inputs are the large part -- a data URL carries the whole
// image in base64 -- so this protocol streams them: the head of each URL is inspected in a Prefix window and the
// payload goes out as inlineData.data while it is still arriving. The prompt is held (bounded) and written at
// the end, after the last image, which is where the buffered path puts it.
//
//   - image inputs come from images[] (each element), image and image_url, in the order they arrive; the buffered
//     path orders them images[] then image then image_url, so a request naming both image and images with image
//     first comes out with the parts in the other order
//   - an input is a string or an object {url} / {image_url:{url}}; the buffered path prefers image_url.url when
//     both are set and non-empty, this protocol rejects that shape (streaming the first one cannot be undone)
//   - an http(s) URL becomes fileData with the mime type from its extension (the whole URL must fit the window);
//     a data:<mime>;base64,<data> URL becomes inlineData; anything else fails the request as it does buffered
//   - edits reject a non-empty mask (the buffered path does not support it either); variations default the prompt
//   - generations ignore image inputs (the buffered request struct has none)
type VertexImageOptions struct {
	// Kind selects which OpenAI endpoint's rules apply.
	Kind VertexImageKind
	// MapModel reproduces the buffered mapModel: an error when model is empty or maps to empty.
	MapModel func(model string) (string, error)
	// SafetySettings mirrors the setting geminiSafetySetting (the buffered path iterates a map, so the order was never fixed).
	SafetySettings []GeminiSafetySetting
	// DetectMime reproduces detectMimeTypeFromURL for http(s) image URLs.
	DetectMime func(url string) string
	// ParseSize reproduces parseImageSize: the aspect ratio and image size for an OpenAI size string.
	ParseSize func(size string) (aspectRatio, imageSize string)
}

type VertexImageKind int

const (
	VertexImageGeneration VertexImageKind = iota
	VertexImageEdit
	VertexImageVariation
)

const vertexImageVariationDefaultPrompt = "Create variations of the provided image."

// aligned with the buffered vertexChatGenerationConfig as buildVertexImageRequest fills it
type vertexImageGenerationConfig struct {
	Temperature        float64            `json:"temperature,omitempty"`
	MaxOutputTokens    int                `json:"maxOutputTokens,omitempty"`
	ThinkingConfig     vertexThinking     `json:"thinkingConfig,omitempty"` // a value, never omitted
	ResponseModalities []string           `json:"responseModalities,omitempty"`
	ImageConfig        *vertexImageConfig `json:"imageConfig,omitempty"`
}

type vertexImageConfig struct {
	AspectRatio        string                    `json:"aspectRatio,omitempty"`
	ImageSize          string                    `json:"imageSize,omitempty"`
	ImageOutputOptions *vertexImageOutputOptions `json:"imageOutputOptions,omitempty"`
	PersonGeneration   string                    `json:"personGeneration,omitempty"`
}

type vertexImageOutputOptions struct {
	MimeType string `json:"mimeType,omitempty"`
}

type vertexImageProto struct {
	opt VertexImageOptions

	model     string
	modelSeen bool
	prompt    string
	size      string
	format    string

	partsOpen bool // the contents/parts levels are pushed
	parts     int  // parts written so far

	// the image input being read: emit writes a part, check (mask) only rejects a non-empty URL
	check   bool
	objBase int  // depth of the input object ({url} / {image_url:{url}}), 0 for a bare string
	urlDone bool // a non-empty URL was taken from the current object
}

// NewVertexImage builds the OpenAI images → Vertex generateContent transformer.
func NewVertexImage(opt VertexImageOptions) *Transformer {
	opt.MapModel = strictMapper(opt.MapModel)
	if opt.DetectMime == nil {
		opt.DetectMime = func(string) string { return "" }
	}
	if opt.ParseSize == nil {
		opt.ParseSize = func(string) (string, string) { return "1:1", "1k" }
	}
	p := &vertexImageProto{opt: opt}
	t := NewTransformer(p)
	t.DupKeyBail = true
	return t
}

func (p *vertexImageProto) Prelude() Prelude { return Prelude{Model: p.model, ModelSeen: p.modelSeen} }

func (p *vertexImageProto) imageInputKey(key string) bool {
	if p.opt.Kind == VertexImageGeneration {
		return false // imageGenerationRequest has no image fields: whatever comes is ignored
	}
	return key == "image" || key == "images" || key == "image_url"
}

func (p *vertexImageProto) maskKey(key string) bool {
	return p.opt.Kind == VertexImageEdit && (key == "mask" || key == "mask_url")
}

func (p *vertexImageProto) OnKey(t *Transformer) Action {
	d := t.Depth()
	if d == 1 {
		switch t.Last() {
		case "model":
			return Capture(4 << 10)
		case "prompt":
			return Capture(embeddingsInputCap)
		case "size":
			return Capture(256)
		case "output_format":
			return Capture(64)
		}
		if p.imageInputKey(t.Last()) {
			p.check = false
			return Probe()
		}
		if p.maskKey(t.Last()) {
			p.check = true
			return Probe()
		}
		return Skip()
	}
	if p.objBase == 0 {
		return Bail("unexpected path: " + t.PathString())
	}
	switch d - p.objBase {
	case 1: // {url} / {image_url}
		switch t.Last() {
		case "url", "image_url":
			return Probe()
		}
		return Skip()
	case 2: // image_url:{url,detail}
		switch t.Last() {
		case "url", "detail":
			return Probe()
		}
		return Skip()
	}
	return Bail("unexpected path: " + t.PathString())
}

func (p *vertexImageProto) OnElem(t *Transformer) Action {
	if t.Depth() == 2 && p.objBase == 0 { // images[]
		return Probe()
	}
	return Bail("unexpected array: " + t.PathString())
}

func (p *vertexImageProto) OnStart(t *Transformer, kind ValueKind) Action {
	d := t.Depth()
	if d == 1 && t.Last() == "images" && !p.check {
		switch kind {
		case KindNull:
			return Skip()
		case KindArray:
			return Enter().Flat()
		}
		return Bail("images is not an array")
	}
	inURL := p.objBase != 0 && d == p.objBase+1 && t.Last() == "image_url"
	if inURL {
		switch kind {
		case KindNull:
			return Skip()
		case KindObject:
			return Enter().Flat()
		}
		return Bail("image_url is not an object")
	}
	if p.objBase != 0 && d == p.objBase+2 && t.Last() == "detail" {
		switch kind {
		case KindNull, KindString:
			return Skip()
		}
		return Bail("detail is not a string")
	}
	// an image input (image / image_url / images[i] / mask) or a url inside one
	switch kind {
	case KindNull:
		return Skip()
	case KindString:
		if p.check {
			return Prefix(1)
		}
		return Prefix(vxURLWin)
	case KindObject:
		if p.objBase != 0 {
			return Bail("url is not a string")
		}
		p.objBase, p.urlDone = d, false
		return Enter().Flat()
	}
	if p.objBase != 0 {
		return Bail("url is not a string")
	}
	return Bail("image input is neither a string nor an object")
}

func (p *vertexImageProto) OnValue(t *Transformer, raw []byte) {
	if t.Depth() != 1 {
		return
	}
	isNull := string(raw) == "null"
	switch t.Last() {
	case "model":
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail("model is not a string")
			return
		}
		p.model, p.modelSeen = s, true
		if _, err := p.opt.MapModel(s); err != nil {
			t.Bail(err.Error())
		}
	case "prompt", "size", "output_format":
		if isNull {
			return
		}
		s, ok := jsonUnquote(raw)
		if !ok {
			t.Bail(t.Last() + " is not a string")
			return
		}
		switch t.Last() {
		case "prompt":
			p.prompt = s
		case "size":
			p.size = s
		case "output_format":
			p.format = s
		}
	}
}

// OnPrefix: the head of an image URL (or of a mask URL, which only has to be empty).
func (p *vertexImageProto) OnPrefix(t *Transformer, raw []byte, complete bool) (Action, int) {
	if complete && len(raw) == 0 {
		return Skip(), 0 // an empty URL is no input at all
	}
	if p.check {
		return Bail("mask is not supported for vertex image edits yet"), 0
	}
	if p.objBase != 0 {
		if p.urlDone {
			return Bail("image input sets both url and image_url, the buffered path takes image_url"), 0
		}
		p.urlDone = true
	}
	w := t.W()
	dec, off := unescapePrefix(raw)
	if bytes.HasPrefix(dec, []byte("http")) { // strings.HasPrefix(mediaUrl, "http"), case-sensitive as on the buffered path
		if !complete {
			return Bail("http image URL exceeds the prefix window, the mime type needs its end"), 0
		}
		u := string(dec)
		p.openPart(w)
		w.RawString(`{"fileData":{"mimeType":`)
		w.JSONString(p.opt.DetectMime(u))
		w.RawString(`,"fileUri":`)
		w.JSONString(u)
		w.RawString("}}")
		return Skip(), 0
	}
	// data:<mime>;base64,<data>; anything else fails the request on the buffered path
	if !bytes.HasPrefix(dec, lit1) {
		if !complete && len(dec) < len(lit1) {
			return Bail("image URL cut at the window boundary"), 0
		}
		return Bail("invalid base64 format, expected data:<mimeType>;base64,<data>"), 0
	}
	semi := bytes.IndexByte(dec, ';')
	if semi < 0 {
		if !complete {
			return Bail("data URL header exceeds the prefix window"), 0
		}
		return Bail("invalid base64 format, expected data:<mimeType>;base64,<data>"), 0
	}
	mime := string(dec[5:semi])
	rest := dec[semi+1:]
	if !complete && len(rest) < len(lit4) {
		return Bail("data URL header cut at the window boundary"), 0
	}
	if mime == "" || !bytes.HasPrefix(rest, lit4) {
		return Bail("invalid base64 format, expected data:<mimeType>;base64,<data>"), 0
	}
	if len(strings.Split(mime, "/")) < 2 {
		return Bail("invalid mimeType: " + mime), 0
	}
	resume := off[semi+1+len(lit4)]
	if !complete && resume >= len(raw) {
		return Bail("data URL payload cut at the window boundary"), 0
	}
	p.openPart(w)
	w.RawString(`{"inlineData":{"mimeType":`)
	w.JSONString(mime)
	w.RawString(`,"data":"`)
	return Pass().Wrap(nil, []byte(`"}}`)), resume
}

// openPart starts the next element of contents[0].parts, pushing the levels on the first one.
func (p *vertexImageProto) openPart(w *Writer) {
	if !p.partsOpen {
		w.PushArr("contents")
		w.PushObj("")
		w.Key("role")
		w.RawString(`"user"`)
		w.PushArr("parts")
		p.partsOpen = true
	}
	w.Elem()
	p.parts++
}

func (p *vertexImageProto) OnLeave(t *Transformer) {
	if p.objBase != 0 && t.Depth() == p.objBase {
		p.objBase, p.urlDone = 0, false
	}
}

func (p *vertexImageProto) Tail(t *Transformer) {
	w := t.W()
	if !p.modelSeen {
		if _, err := p.opt.MapModel(""); err != nil {
			t.Bail(err.Error())
			return
		}
	}
	prompt := p.prompt
	switch p.opt.Kind {
	case VertexImageEdit:
		if p.parts == 0 {
			t.Bail("missing image_url in request")
			return
		}
		if prompt == "" {
			t.Bail("missing prompt in request")
			return
		}
	case VertexImageVariation:
		if p.parts == 0 {
			t.Bail("missing image_url in request")
			return
		}
		if prompt == "" {
			prompt = vertexImageVariationDefaultPrompt
		}
	}
	if prompt != "" {
		p.openPart(w)
		w.RawString(`{"text":`)
		w.JSONString(prompt)
		w.Byte('}')
	}
	if p.parts == 0 {
		t.Bail("missing prompt and image_url in request")
		return
	}
	w.Pop()
	w.Pop()
	w.Pop()
	if len(p.opt.SafetySettings) > 0 { // the buffered field has omitempty: an empty slice is left out
		b, _ := json.Marshal(p.opt.SafetySettings)
		w.Key("safetySettings")
		w.Raw(b)
	}
	aspect, size := p.opt.ParseSize(p.size)
	mime := "image/png"
	switch p.format {
	case "jpeg", "jpg":
		mime = "image/jpeg"
	case "webp":
		mime = "image/webp"
	}
	cfg := vertexImageGenerationConfig{
		Temperature:        1.0,
		MaxOutputTokens:    32768,
		ResponseModalities: []string{"TEXT", "IMAGE"},
		ImageConfig: &vertexImageConfig{
			AspectRatio:        aspect,
			ImageSize:          size,
			ImageOutputOptions: &vertexImageOutputOptions{MimeType: mime},
			PersonGeneration:   "ALLOW_ALL",
		},
	}
	b, _ := json.Marshal(cfg)
	w.Key("generationConfig")
	w.Raw(b)
}
