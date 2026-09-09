package provider

// Differential tests of the OpenAI videos → Kling rewrite against transformOpenAIVideoRequest, plus the image-to-video decision.

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

var klingMapping = map[string]string{"m": "kling-v2-master", "k*": "kling-v1"}

func checkKling(t *testing.T, in string) {
	t.Helper()
	k := &klingProvider{config: ProviderConfig{modelMapping: klingMapping}}
	offBody, err := k.transformOpenAIVideoRequest(newMapCtx(), []byte(in))
	require.NoError(t, err, in)
	off, err := decodeMap(offBody)
	require.NoError(t, err, in)
	wantImage := k.isImageToVideoRequest([]byte(in))
	for _, cs := range []int{1, 7, 4096} {
		tr := streamxform.NewKling(streamxform.KlingOptions{MapModel: func(m string) string { return getMappedModel(m, klingMapping) }})
		str, ok, why := runStream(tr, in, cs)
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", cs, why, in)
		require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", cs, in)
		require.Equal(t, wantImage, tr.Protocol().(interface{ ImageToVideo() bool }).ImageToVideo(), "chunk=%d image decision\n  %s", cs, in)
	}
}

func TestKlingDifferential(t *testing.T) {
	big := strings.Repeat("p", 70000)
	for _, in := range []string{
		`{"model":"m","prompt":"a cat"}`,
		`{"model_name":"k1","prompt":"a cat","duration":"5"}`,
		`{"model":"m","model_name":"k1","prompt":"x"}`,
		`{"prompt":"no model at all","seconds":"10"}`,
		`{"model":"unmapped","prompt":"` + big + `","image":"data:image/png;base64,AAAA"}`,
		`{"prompt":"x","input_image":{"url":"u"},"model":"m"}`,
		`{"prompt":"x","nested":{"image":"not top level"},"model":"m"}`,
		`{"prompt":"x","image_urls":null,"model":"m"}`,
		`{"model":"m","prompt":"x","last_frame_image":"", "camera_control":{"type":"simple"}}`,
	} {
		checkKling(t, in)
	}
}

func TestKlingFuzz(t *testing.T) {
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	fields := []string{`"model":"m"`, `"model_name":"k9"`, `"prompt":"p"`, `"image":"i"`, `"image_tail_url":"u"`, `"duration":"5"`, `"negative_prompt":"n"`, `"nested":{"images":["x"]}`, `"seconds":"10"`}
	for i := 0; i < fuzzN(200); i++ {
		var parts []string
		seen := map[string]bool{}
		for _, j := range rnd.Perm(len(fields)) {
			k := fields[j][:strings.Index(fields[j], ":")]
			if seen[k] || rnd.Intn(3) == 0 {
				continue
			}
			seen[k] = true
			parts = append(parts, fields[j])
		}
		checkKling(t, "{"+strings.Join(parts, ",")+"}")
	}
}
