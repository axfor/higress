package provider

// Differential tests of the small non-chat conversions -- embeddings for Gemini, Vertex and native Qwen, Gemini
// image generation -- against the buffered builders, body and path.

import (
	"encoding/json"
	"errors"
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

var embMapping = map[string]string{"m": "text-embedding-004", "empty": ""}

var errBuildPanic = errors.New("buffered builder panics")

func mapEmb(model string) (string, error) {
	if model == "" {
		return "", errors.New("missing model in request")
	}
	mapped := getMappedModel(model, embMapping)
	if mapped == "" {
		return "", errors.New("model becomes empty after applying the configured mapping")
	}
	return mapped, nil
}

func decodeEmb(in string) (*embeddingsRequest, error) {
	req := &embeddingsRequest{}
	if err := decodeEmbeddingsRequest([]byte(in), req); err != nil {
		return nil, err
	}
	mapped, err := mapEmb(req.Model)
	if err != nil {
		return nil, err
	}
	req.Model = mapped
	return req, nil
}

type embOfficial func(in string) (map[string]any, string, error)

func officialGeminiEmb(in string) (m map[string]any, path string, err error) {
	req, err := decodeEmb(in)
	if err != nil {
		return nil, "", err
	}
	g := &geminiProvider{config: ProviderConfig{modelMapping: embMapping}}
	b, err := json.Marshal(g.buildBatchEmbeddingRequest(req))
	if err != nil {
		return nil, "", err
	}
	m, err = decodeMap(b)
	return m, g.getRequestPath(ApiNameEmbeddings, req.Model, false), err
}

func officialVertexEmb(in string) (map[string]any, string, error) {
	req, err := decodeEmb(in)
	if err != nil {
		return nil, "", err
	}
	v := &vertexProvider{config: ProviderConfig{modelMapping: embMapping, vertexProjectId: "p", vertexRegion: "us-central1"}}
	b, err := json.Marshal(v.buildEmbeddingRequest(req))
	if err != nil {
		return nil, "", err
	}
	m, err := decodeMap(b)
	return m, v.getRequestPath(newMapCtx(), ApiNameEmbeddings, req.Model, false), err
}

func officialQwenEmb(in string) (m map[string]any, path string, err error) {
	defer func() {
		if r := recover(); r != nil {
			m, err = nil, errBuildPanic // buildQwenTextEmbeddingRequest calls reflect.TypeOf(nil).String() on a missing input
		}
	}()
	req, err := decodeEmb(in)
	if err != nil {
		return nil, "", err
	}
	q := &qwenProvider{config: ProviderConfig{modelMapping: embMapping}}
	out, err := q.buildQwenTextEmbeddingRequest(req)
	if err != nil {
		return nil, "", err
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil, "", err
	}
	m, err = decodeMap(b)
	return m, "", err
}

func officialGeminiImage(in string) (map[string]any, string, error) {
	req := &imageGenerationRequest{}
	if err := decodeImageGenerationRequest([]byte(in), req); err != nil {
		return nil, "", err
	}
	mapped, err := mapEmb(req.Model)
	if err != nil {
		return nil, "", err
	}
	req.Model = mapped
	g := &geminiProvider{config: ProviderConfig{modelMapping: embMapping}}
	b, err := json.Marshal(g.buildGeminiImageGenerationRequest(req))
	if err != nil {
		return nil, "", err
	}
	m, err := decodeMap(b)
	return m, g.getRequestPath(ApiNameImageGeneration, req.Model, false), err
}

func checkEmb(t *testing.T, name string, official embOfficial, mk func() *streamxform.Transformer, pathOf func(model string) string, in string) {
	t.Helper()
	off, wantPath, err := official(in)
	if errors.Is(err, errBuildPanic) {
		return
	}
	for _, cs := range []int{1, 7, 4096} {
		tr := mk()
		str, ok, why := runStream(tr, in, cs)
		if err != nil {
			require.False(t, ok, "%s chunk=%d: buffered failed (%v) but streaming passed\n  %s", name, cs, err, in)
			continue
		}
		require.True(t, ok, "%s chunk=%d unexpected fallback: %s\n  %s", name, cs, why, in)
		require.Empty(t, diffMaps(off, str), "%s chunk=%d\n  %s", name, cs, in)
		if pathOf != nil {
			pre := tr.Protocol().(streamxform.Preluder).Prelude()
			mapped, _ := mapEmb(pre.Model)
			require.Equal(t, wantPath, pathOf(mapped), "%s chunk=%d path\n  %s", name, cs, in)
		}
	}
}

func embCases() []string {
	big := strings.Repeat("e", 70000)
	var many []string
	for i := 0; i < 300; i++ {
		many = append(many, `"item `+strings.Repeat("y", i%50)+`"`)
	}
	return []string{
		`{"model":"m","input":"hello"}`,
		`{"model":"m","input":["a","b","c"]}`,
		`{"input":["x","y"],"model":"m"}`,
		`{"input":"single first","model":"m","encoding_format":"float","dimensions":3,"user":"u"}`,
		`{"model":"m","input":["a",1,null,{"o":1},"b<>&\"\n",""]}`,
		`{"model":"m","input":[]}`,
		`{"model":"m","input":null}`,
		`{"model":"m"}`,
		`{"model":"m","input":""}`,
		`{"model":"m","input":"` + big + `"}`,
		`{"model":"m","input":[` + strings.Join(many, ",") + `]}`,
		`{"input":[` + strings.Join(many, ",") + `],"model":"m"}`,
		`{"model":"m","input":{"weird":1}}`,
		`{"model":"m","input":42}`,
		`{"input":["a"]}`,
		`{"model":"empty","input":["a"]}`,
		`{"model":"m","input":["a"],"dimensions":"x"}`,
	}
}

func TestEmbeddingsDifferential(t *testing.T) {
	g := &geminiProvider{config: ProviderConfig{modelMapping: embMapping}}
	v := &vertexProvider{config: ProviderConfig{modelMapping: embMapping, vertexProjectId: "p", vertexRegion: "us-central1"}}
	for _, in := range embCases() {
		checkEmb(t, "gemini", officialGeminiEmb, func() *streamxform.Transformer {
			tr := streamxform.NewGeminiEmbeddings(streamxform.EmbeddingsOptions{MapModel: mapEmb})
			tr.SetFieldTree(embeddingsFieldTree)
			return tr
		}, func(m string) string { return g.getRequestPath(ApiNameEmbeddings, m, false) }, in)
		checkEmb(t, "vertex", officialVertexEmb, func() *streamxform.Transformer {
			tr := streamxform.NewVertexEmbeddings(streamxform.EmbeddingsOptions{MapModel: mapEmb})
			tr.SetFieldTree(embeddingsFieldTree)
			return tr
		}, func(m string) string { return v.getRequestPath(newMapCtx(), ApiNameEmbeddings, m, false) }, in)
		checkEmb(t, "qwen", officialQwenEmb, func() *streamxform.Transformer {
			tr := streamxform.NewQwenEmbeddings(streamxform.EmbeddingsOptions{MapModel: mapEmb})
			tr.SetFieldTree(embeddingsFieldTree)
			return tr
		}, nil, in)
	}
}

func TestEmbeddingsFuzz(t *testing.T) {
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	items := []string{`"a"`, `""`, `"b\n"`, `1`, `null`, `{"o":1}`, `"` + strings.Repeat("z", 3000) + `"`}
	for i := 0; i < fuzzN(200); i++ {
		var sel []string
		for _, j := range rnd.Perm(len(items))[:rnd.Intn(len(items)+1)] {
			sel = append(sel, items[j])
		}
		input := `"input":[` + strings.Join(sel, ",") + `]`
		if rnd.Intn(4) == 0 {
			input = `"input":"solo"`
		}
		fields := []string{`"model":"m"`, input, `"dimensions":8`, `"user":"u"`}
		var parts []string
		for _, j := range rnd.Perm(len(fields)) {
			if j < 2 || rnd.Intn(2) == 0 {
				parts = append(parts, fields[j])
			}
		}
		in := "{" + strings.Join(parts, ",") + "}"
		checkEmb(t, "gemini", officialGeminiEmb, func() *streamxform.Transformer {
			return streamxform.NewGeminiEmbeddings(streamxform.EmbeddingsOptions{MapModel: mapEmb})
		}, nil, in)
		checkEmb(t, "vertex", officialVertexEmb, func() *streamxform.Transformer {
			return streamxform.NewVertexEmbeddings(streamxform.EmbeddingsOptions{MapModel: mapEmb})
		}, nil, in)
		checkEmb(t, "qwen", officialQwenEmb, func() *streamxform.Transformer {
			return streamxform.NewQwenEmbeddings(streamxform.EmbeddingsOptions{MapModel: mapEmb})
		}, nil, in)
	}
}

func TestGeminiImageDifferential(t *testing.T) {
	g := &geminiProvider{config: ProviderConfig{modelMapping: embMapping}}
	for _, in := range []string{
		`{"model":"m","prompt":"a cat"}`,
		`{"model":"m","prompt":"a cat","n":2,"size":"1024x1024","quality":"hd"}`,
		`{"prompt":"","model":"m","n":0}`,
		`{"model":"m","prompt":null,"n":null}`,
		`{"model":"m"}`,
		`{"prompt":"x"}`,
		`{"model":"m","prompt":"x","n":"two"}`,
		`{"model":"m","prompt":"` + strings.Repeat("p", 70000) + `","n":3}`,
	} {
		checkEmb(t, "gemini-image", officialGeminiImage, func() *streamxform.Transformer {
			tr := streamxform.NewGeminiImage(streamxform.EmbeddingsOptions{MapModel: mapEmb})
			tr.SetFieldTree(imageGenerationFieldTree)
			return tr
		}, func(m string) string { return g.getRequestPath(ApiNameImageGeneration, m, false) }, in)
	}
}
