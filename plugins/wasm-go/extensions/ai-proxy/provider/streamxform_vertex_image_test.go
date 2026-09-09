package provider

// Differential tests of the OpenAI images → Vertex generateContent conversion against buildVertexImageRequest and
// the checks around it in the buffered edit / variation handlers: hand-written shapes for the three endpoints, a
// random corpus, the documented deviations, and the fallbacks, at chunk sizes 1, 7 and 4096.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

var vxImgMapping = map[string]string{"m": "gemini-2.5-flash-image", "empty": ""}

func newVertexImageProvider() *vertexProvider {
	return &vertexProvider{config: ProviderConfig{
		vertexProjectId:     "p",
		vertexRegion:        "us-central1",
		modelMapping:        vxImgMapping,
		geminiSafetySetting: map[string]string{"HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH"},
	}}
}

func mapVxImg(model string) (string, error) {
	if model == "" {
		return "", errors.New("missing model in request")
	}
	mapped := getMappedModel(model, vxImgMapping)
	if mapped == "" {
		return "", errors.New("model becomes empty after applying the configured mapping")
	}
	return mapped, nil
}

// officialVertexImage is onImageGenerationRequestBody / onImageEditRequestBody / onImageVariationRequestBody without
// the host: decode, map the model, the handler's own checks, buildVertexImageRequest.
func officialVertexImage(v *vertexProvider, apiName ApiName, in string) (map[string]any, string, error) {
	var (
		vr    *vertexChatRequest
		model string
		err   error
	)
	switch apiName {
	case ApiNameImageGeneration:
		req := &imageGenerationRequest{}
		if err = decodeImageGenerationRequest([]byte(in), req); err != nil {
			return nil, "", err
		}
		if req.Model, err = mapVxImg(req.Model); err != nil {
			return nil, "", err
		}
		model = req.Model
		vr, err = v.buildVertexImageGenerationRequest(req)
	case ApiNameImageEdit:
		req := &imageEditRequest{}
		if err = decodeImageEditRequest([]byte(in), req); err != nil {
			return nil, "", err
		}
		if req.Model, err = mapVxImg(req.Model); err != nil {
			return nil, "", err
		}
		model = req.Model
		if req.HasMask() {
			return nil, "", errors.New("mask is not supported for vertex image edits yet")
		}
		urls := req.GetImageURLs()
		if len(urls) == 0 {
			return nil, "", errors.New("missing image_url in request")
		}
		if req.Prompt == "" {
			return nil, "", errors.New("missing prompt in request")
		}
		vr, err = v.buildVertexImageRequest(req.Prompt, req.Size, req.OutputFormat, urls)
	case ApiNameImageVariation:
		req := &imageVariationRequest{}
		if err = decodeImageVariationRequest([]byte(in), req); err != nil {
			return nil, "", err
		}
		if req.Model, err = mapVxImg(req.Model); err != nil {
			return nil, "", err
		}
		model = req.Model
		urls := req.GetImageURLs()
		if len(urls) == 0 {
			return nil, "", errors.New("missing image_url in request")
		}
		prompt := req.Prompt
		if prompt == "" {
			prompt = vertexImageVariationDefaultPrompt
		}
		vr, err = v.buildVertexImageRequest(prompt, req.Size, req.OutputFormat, urls)
	}
	if err != nil {
		return nil, "", err
	}
	b, err := json.Marshal(vr)
	if err != nil {
		return nil, "", err
	}
	m, err := decodeMap(b)
	if err != nil {
		return nil, "", err
	}
	if ss, ok := m["safetySettings"].([]any); ok { // a map on the buffered path: compare as a set
		sort.Slice(ss, func(i, j int) bool {
			return ss[i].(map[string]any)["category"].(string) < ss[j].(map[string]any)["category"].(string)
		})
	}
	return m, v.getRequestPath(newMapCtx(), apiName, model, false), nil
}

func vxImgKind(apiName ApiName) streamxform.VertexImageKind {
	switch apiName {
	case ApiNameImageEdit:
		return streamxform.VertexImageEdit
	case ApiNameImageVariation:
		return streamxform.VertexImageVariation
	}
	return streamxform.VertexImageGeneration
}

func newVxImgTransformer(v *vertexProvider, apiName ApiName) *streamxform.Transformer {
	var ss []streamxform.GeminiSafetySetting
	for k, val := range v.config.geminiSafetySetting {
		ss = append(ss, streamxform.GeminiSafetySetting{Category: k, Threshold: val})
	}
	sort.Slice(ss, func(i, j int) bool { return ss[i].Category < ss[j].Category })
	tr := streamxform.NewVertexImage(streamxform.VertexImageOptions{
		Kind: vxImgKind(apiName), MapModel: mapVxImg, SafetySettings: ss, DetectMime: detectMimeTypeFromURL, ParseSize: v.parseImageSize,
	})
	switch apiName {
	case ApiNameImageEdit:
		tr.SetFieldTree(imageEditFieldTree)
	case ApiNameImageVariation:
		tr.SetFieldTree(imageVariationFieldTree)
	default:
		tr.SetFieldTree(imageGenerationFieldTree)
	}
	return tr
}

func checkVxImg(t *testing.T, v *vertexProvider, apiName ApiName, in string) {
	t.Helper()
	off, wantPath, err := officialVertexImage(v, apiName, in)
	for _, cs := range []int{1, 7, 4096} {
		tr := newVxImgTransformer(v, apiName)
		str, ok, why := runStream(tr, in, cs)
		if err != nil {
			require.False(t, ok, "%s chunk=%d: buffered failed (%v) but streaming passed\n  %s", apiName, cs, err, truncateIn(in))
			continue
		}
		require.True(t, ok, "%s chunk=%d unexpected fallback: %s\n  %s", apiName, cs, why, truncateIn(in))
		require.Empty(t, diffMaps(off, str), "%s chunk=%d\n  %s", apiName, cs, truncateIn(in))
		pre := tr.Protocol().(streamxform.Preluder).Prelude()
		mapped, _ := mapVxImg(pre.Model)
		require.Equal(t, wantPath, v.getRequestPath(newMapCtx(), apiName, mapped, false), "%s chunk=%d path\n  %s", apiName, cs, truncateIn(in))
	}
}

func truncateIn(in string) string {
	if len(in) > 300 {
		return in[:300] + "..."
	}
	return in
}

var vxImgAPIs = []ApiName{ApiNameImageGeneration, ApiNameImageEdit, ApiNameImageVariation}

func vxImgCases() []string {
	png := "data:image/png;base64," + strings.Repeat("iVBORw0KGgo", 20)
	big := "data:image/jpeg;base64," + strings.Repeat("/9j/4AAQSkZJRg", 20000) // ~280KB
	return []string{
		// generations
		`{"model":"m","prompt":"a cat"}`,
		`{"model":"m","prompt":"a cat","size":"1792x1024","output_format":"jpeg","n":2,"quality":"hd","response_format":"b64_json"}`,
		`{"prompt":"a cat","size":"1024x1792","model":"m","output_format":"webp"}`,
		`{"model":"m","prompt":"a cat","size":"weird","output_format":"gif"}`,
		`{"model":"m","prompt":"a cat","size":"","output_format":""}`,
		`{"model":"m","prompt":"a cat","size":null,"output_format":null}`,
		`{"model":"m","prompt":"café \"quoted\" \\ back\nline","size":"512x512"}`,
		`{"model":"m","prompt":""}`,
		`{"model":"m","prompt":null}`,
		`{"model":"m"}`,
		`{"prompt":"x"}`,
		`{"model":null,"prompt":"x"}`,
		`{"model":"empty","prompt":"x"}`,
		`{"model":"m","prompt":"x","n":"two"}`,
		`{"model":"m","prompt":5}`,
		`{"model":"m","prompt":"x","size":42}`,
		`{"model":"m","prompt":"` + strings.Repeat("p", 70000) + `","size":"2048x2048"}`,
		// generations ignore image inputs; edits and variations read them
		`{"model":"m","prompt":"blue","image":"` + png + `"}`,
		`{"image":"` + png + `","model":"m","prompt":"blue"}`,
		`{"image":"` + png + `","prompt":"blue","model":"m"}`,
		`{"model":"m","prompt":"blue","image":{"url":"` + png + `"}}`,
		`{"model":"m","prompt":"blue","image":{"image_url":{"url":"` + png + `","detail":"high"}}}`,
		`{"model":"m","prompt":"blue","image":{"image_url":{"detail":"low","url":"` + png + `"}}}`,
		`{"model":"m","prompt":"blue","image_url":"` + png + `"}`,
		`{"model":"m","prompt":"blue","image_url":{"url":"` + png + `"}}`,
		`{"model":"m","prompt":"blue","images":["` + png + `",{"url":"https://x.test/a/b.JPG?x=1#f"},{"image_url":{"url":"http://x.test/noext"}}]}`,
		`{"model":"m","prompt":"blue","images":["` + png + `"],"image":"https://x.test/c.webp","image_url":{"url":"` + png + `"}}`,
		`{"model":"m","prompt":"blue","images":[]}`,
		`{"model":"m","prompt":"blue","images":null,"image":null,"image_url":null}`,
		`{"model":"m","prompt":"blue","images":[null,"","` + png + `",{},{"url":""},{"url":null},{"image_url":null},{"image_url":{}}]}`,
		`{"model":"m","prompt":"blue","image":""}`,
		`{"model":"m","prompt":"blue","image":{}}`,
		`{"model":"m","prompt":"blue","image":{"url":"` + png + `","image_url":{"url":""}}}`,
		`{"model":"m","prompt":"blue","image":{"url":"` + png + `","image_url":null}}`,
		`{"model":"m","prompt":"blue","image":{"url":"","image_url":{"url":"` + png + `"}}}`,
		`{"model":"m","prompt":"blue","image":{"url":"` + png + `","foo":1,"image_url":{"url":"","detail":null,"bar":[1]}}}`,
		`{"model":"m","prompt":"blue","image":"data:image\/png;base64,AAAA\/BBBB"}`,
		`{"model":"m","prompt":"blue","image":"data:image/svg+xml;base64,PHN2Zz4="}`,
		`{"model":"m","prompt":"blue","image":"data:image/png;base64,"}`,
		`{"model":"m","prompt":"blue","image":"` + big + `"}`,
		`{"model":"m","image":"` + big + `","prompt":"blue"}`,
		`{"model":"m","prompt":"blue","images":["` + big + `","` + big + `"],"size":"1024x1024","output_format":"jpg"}`,
		// edits require a prompt, variations default it
		`{"model":"m","image":"` + png + `"}`,
		`{"model":"m","prompt":"","image":"` + png + `"}`,
		`{"model":"m","prompt":null,"image":"` + png + `"}`,
		// masks: edits reject a non-empty one, variations have no such field
		`{"model":"m","prompt":"blue","image":"` + png + `","mask":"` + png + `"}`,
		`{"model":"m","prompt":"blue","image":"` + png + `","mask":""}`,
		`{"model":"m","prompt":"blue","image":"` + png + `","mask":null}`,
		`{"model":"m","prompt":"blue","image":"` + png + `","mask":{"url":""}}`,
		`{"model":"m","prompt":"blue","image":"` + png + `","mask":{"image_url":{"url":"x"}}}`,
		`{"model":"m","prompt":"blue","image":"` + png + `","mask_url":"` + png + `"}`,
		`{"model":"m","prompt":"blue","image":"` + png + `","mask_url":{}}`,
		`{"model":"m","prompt":"blue","mask":"` + png + `","image":"` + png + `"}`,
		// URLs the buffered path rejects
		`{"model":"m","prompt":"blue","image":"foo"}`,
		`{"model":"m","prompt":"blue","image":"data:;base64,AAAA"}`,
		`{"model":"m","prompt":"blue","image":"data:imagepng;base64,AAAA"}`,
		`{"model":"m","prompt":"blue","image":"data:image/png,AAAA"}`,
		`{"model":"m","prompt":"blue","image":"DATA:image/png;base64,AAAA"}`,
		`{"model":"m","prompt":"blue","image":"data:image/png;charset=x;base64,AAAA"}`,
		`{"model":"m","prompt":"blue","images":["` + png + `","foo"]}`,
		// shapes the buffered decode rejects
		`{"model":"m","prompt":"blue","image":5}`,
		`{"model":"m","prompt":"blue","image":true}`,
		`{"model":"m","prompt":"blue","image":[]}`,
		`{"model":"m","prompt":"blue","images":"` + png + `"}`,
		`{"model":"m","prompt":"blue","images":{}}`,
		`{"model":"m","prompt":"blue","images":[5]}`,
		`{"model":"m","prompt":"blue","images":[[]]}`,
		`{"model":"m","prompt":"blue","image":{"url":5}}`,
		`{"model":"m","prompt":"blue","image":{"url":{}}}`,
		`{"model":"m","prompt":"blue","image":{"image_url":"` + png + `"}}`,
		`{"model":"m","prompt":"blue","image":{"image_url":{"url":5}}}`,
		`{"model":"m","prompt":"blue","image":{"image_url":{"url":"` + png + `","detail":5}}}`,
		// the model after a large image: the differential does not model the commit window, the emulator test does
		`{"prompt":"blue","image":"` + big + `","model":"m"}`,
	}
}

func TestVertexImageDifferential(t *testing.T) {
	v := newVertexImageProvider()
	for _, apiName := range vxImgAPIs {
		for _, in := range vxImgCases() {
			checkVxImg(t, v, apiName, in)
		}
	}
}

// Without safety settings the field is left out on both sides.
func TestVertexImageNoSafetySettings(t *testing.T) {
	v := newVertexImageProvider()
	v.config.geminiSafetySetting = nil
	png := "data:image/png;base64,AAAA"
	for _, apiName := range vxImgAPIs {
		checkVxImg(t, v, apiName, `{"model":"m","prompt":"blue","image":"`+png+`"}`)
	}
}

// The documented deviations: image before images comes out in arrival order, and an input naming both url and
// image_url (non-empty) is rejected rather than resolved.
func TestVertexImageDeviations(t *testing.T) {
	v := newVertexImageProvider()
	png := "data:image/png;base64,AAAA"
	jpg := "data:image/jpeg;base64,BBBB"
	in := `{"model":"m","prompt":"blue","image":"` + png + `","images":["` + jpg + `"]}`
	off, _, err := officialVertexImage(v, ApiNameImageEdit, in)
	require.NoError(t, err)
	str, ok, why := runStream(newVxImgTransformer(v, ApiNameImageEdit), in, 7)
	require.True(t, ok, why)
	require.NotEmpty(t, diffMaps(off, str), "arrival order differs from the buffered images-first order")
	parts := func(m map[string]any) []any {
		return m["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	}
	op, sp := parts(off), parts(str)
	require.Equal(t, []any{op[1], op[0], op[2]}, sp)

	for _, in := range []string{
		`{"model":"m","prompt":"blue","image":{"url":"` + png + `","image_url":{"url":"` + jpg + `"}}}`,
		`{"model":"m","prompt":"blue","image":{"image_url":{"url":"` + jpg + `"},"url":"` + png + `"}}`,
	} {
		_, ok, why := runStream(newVxImgTransformer(v, ApiNameImageEdit), in, 7)
		require.False(t, ok)
		require.Contains(t, why, "both url and image_url")
	}
}

// An http(s) URL has to fit the prefix window whole: its mime type comes from the extension at its end.
func TestVertexImageLongHTTPURL(t *testing.T) {
	v := newVertexImageProvider()
	in := `{"model":"m","prompt":"blue","image":"https://x.test/` + strings.Repeat("a", 9000) + `.png"}`
	_, _, err := officialVertexImage(v, ApiNameImageEdit, in)
	require.NoError(t, err)
	for _, cs := range []int{1, 4096} {
		_, ok, why := runStream(newVxImgTransformer(v, ApiNameImageEdit), in, cs)
		require.False(t, ok)
		require.Contains(t, why, "prefix window")
	}
}

// genVertexImageRequest: random field order and values over every field the buffered handlers read, with the
// image keys kept in the buffered order (images, image, image_url) when more than one is present, and never both
// url and image_url set in one input -- the two documented deviations.
func genVertexImageRequest(r *rand.Rand) string {
	bad := r.Intn(5) == 0 // one request in five may carry values the buffered path rejects
	pick := func(xs ...string) string { return xs[r.Intn(len(xs))] }
	pickBad := func(good []string, badOnes ...string) string {
		if bad && r.Intn(3) == 0 {
			return badOnes[r.Intn(len(badOnes))]
		}
		return good[r.Intn(len(good))]
	}
	q := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	b64 := func() string {
		n := r.Intn(200)
		if r.Intn(20) == 0 {
			n = 100000
		}
		const alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
		sb := make([]byte, n)
		for i := range sb {
			sb[i] = alpha[r.Intn(len(alpha))]
		}
		return string(sb)
	}
	urlVal := func() string { // a URL string value, JSON-encoded
		switch r.Intn(12) {
		case 0:
			if bad {
				return `""`
			}
			return q("data:image/png;base64," + b64())
		case 1:
			return q("https://x.test/img" + fmt.Sprint(r.Intn(9)) + pick(".png", ".jpg", ".webp", "", ".GIF", ".bin?x=1", ".svg#frag"))
		case 2:
			return q("http://x.test/" + pick("a.png", "b.jpeg", "noext", "c.tiff"))
		case 3:
			if bad {
				return q(pick("foo", "data:;base64,AAAA", "data:imagepng;base64,AAAA", "data:image/png,AAAA", "DATA:image/png;base64,AA", "ftp://x/a.png"))
			}
			return q("data:image/webp;base64," + b64())
		case 4:
			return `"data:image\/png;base64,` + b64() + `"`
		default:
			return q("data:" + pick("image/png", "image/jpeg", "image/webp", "image/gif", "audio/mp3") + ";base64," + b64())
		}
	}
	input := func() string { // an imageInputURL in one of its shapes
		switch r.Intn(10) {
		case 0:
			if bad {
				return pick("null", "{}")
			}
			return urlVal()
		case 1:
			return `{"url":` + urlVal() + `}`
		case 2:
			return `{"image_url":{"url":` + urlVal() + pick("", `,"detail":"high"`, `,"detail":null`) + `}}`
		case 3:
			return `{"url":` + urlVal() + `,"image_url":` + pick("null", "{}", `{"url":""}`, `{"detail":"auto"}`) + `}`
		case 4:
			return `{"url":"","image_url":{"url":` + urlVal() + `}}`
		case 5:
			if bad && r.Intn(3) == 0 {
				return pick("5", "true", "[]", `{"url":5}`, `{"image_url":"x"}`, `{"image_url":{"url":5}}`, `{"image_url":{"detail":5}}`)
			}
			return urlVal()
		default:
			return urlVal()
		}
	}
	type kv struct{ k, v string }
	var fields []kv
	add := func(k, v string) { fields = append(fields, kv{k, v}) }
	if !bad || r.Intn(10) != 0 {
		add("model", pickBad([]string{`"m"`, `"gemini-2.5-flash-image"`}, `"empty"`, `""`, "null", "5"))
	}
	if !bad || r.Intn(6) != 0 {
		add("prompt", pickBad([]string{`"a cat"`, `"a cat"`, q("café \"q\" \\ \n"), q(strings.Repeat("p", 70000))}, `""`, "null", "7"))
	}
	if r.Intn(2) == 0 {
		add("size", pickBad([]string{`"1024x1024"`, `"1792x1024"`, `"1024x1792"`, `"256x256"`, `"2560x1080"`, `"weird"`, `""`, "null"}, "42"))
	}
	if r.Intn(2) == 0 {
		add("output_format", pickBad([]string{`"jpeg"`, `"jpg"`, `"webp"`, `"png"`, `"gif"`, `""`, "null"}, "1"))
	}
	if r.Intn(3) == 0 {
		add("n", pickBad([]string{"1", "2", "null"}, `"two"`))
	}
	if r.Intn(3) == 0 {
		add(pick("quality", "style", "background", "response_format", "user", "foo"), pick(`"x"`, "1", "null", `{"a":[1,2]}`))
	}
	if r.Intn(4) == 0 {
		add(pick("mask", "mask_url"), pickBad([]string{`""`, "null", "{}", `{"url":""}`, `{"image_url":{"url":""}}`}, urlVal(), `{"url":`+urlVal()+`}`))
	}
	r.Shuffle(len(fields), func(i, j int) { fields[i], fields[j] = fields[j], fields[i] })
	// image keys in the buffered order, inserted at random positions that keep that order
	var imgs []kv
	if r.Intn(3) == 0 {
		var els []string
		for i := r.Intn(4); i > 0; i-- {
			els = append(els, input())
		}
		imgs = append(imgs, kv{"images", pickBad([]string{"[" + strings.Join(els, ",") + "]", "[" + strings.Join(els, ",") + "]", "null", "[]"}, `"x"`, "{}")})
	}
	if r.Intn(2) == 0 {
		imgs = append(imgs, kv{"image", input()})
	}
	if r.Intn(3) == 0 {
		imgs = append(imgs, kv{"image_url", input()})
	}
	if len(imgs) == 0 && r.Intn(4) != 0 {
		imgs = append(imgs, kv{"image", input()})
	}
	for _, im := range imgs {
		pos := r.Intn(len(fields) + 1)
		fields = append(fields[:pos], append([]kv{im}, fields[pos:]...)...)
	}
	// keep images < image < image_url
	order := map[string]int{"images": 0, "image": 1, "image_url": 2}
	var idx []int
	for i, f := range fields {
		if _, ok := order[f.k]; ok {
			idx = append(idx, i)
		}
	}
	sort.SliceStable(imgs, func(i, j int) bool { return order[imgs[i].k] < order[imgs[j].k] })
	for n, i := range idx {
		fields[i] = imgs[n]
	}
	var sb strings.Builder
	sb.WriteByte('{')
	for i, f := range fields {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(q(f.k))
		sb.WriteByte(':')
		sb.WriteString(f.v)
	}
	sb.WriteByte('}')
	return sb.String()
}

func TestVertexImageFuzz(t *testing.T) {
	v := newVertexImageProvider()
	r := rand.New(rand.NewSource(53))
	for i := 0; i < 300; i++ {
		in := genVertexImageRequest(r)
		for _, apiName := range vxImgAPIs {
			checkVxImg(t, v, apiName, in)
		}
	}
}
