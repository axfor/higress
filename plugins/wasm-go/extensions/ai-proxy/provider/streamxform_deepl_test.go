package provider

// Differential tests of the OpenAI → DeepL conversion against the buffered path's pure parts: developer → system
// (handleRequestBody), struct decode, the host chosen by model, deeplTextGenRequest, marshal.

import (
	"encoding/json"
	"errors"
	"math/rand"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

func officialDeepL(in string, d *deeplProvider) (map[string]any, string, error) {
	body, err := convertDeveloperRoleToSystem([]byte(in))
	if err != nil {
		return nil, "", err
	}
	req := &chatCompletionRequest{}
	if err := decodeChatCompletionRequest(body, req); err != nil {
		return nil, "", err
	}
	hdr := http.Header{}
	if err := d.overwriteRequestHost(hdr, req.Model); err != nil {
		return nil, "", err
	}
	b, err := json.Marshal(d.deeplTextGenRequest(req))
	if err != nil {
		return nil, "", err
	}
	m, err := decodeMap(b)
	return m, hdr.Get(":authority"), err
}

func deeplStream(d *deeplProvider) *streamxform.Transformer {
	return typed(streamxform.NewDeepL(streamxform.DeepLOptions{TargetLang: d.config.targetLang}))
}

func deeplHostFor(model string) (string, error) {
	switch model {
	case "Pro":
		return deeplHostPro, nil
	case "Free":
		return deeplHostFree, nil
	}
	return "", errors.New("bad model")
}

func checkDeepL(t *testing.T, d *deeplProvider, in string) {
	t.Helper()
	off, wantHost, err := officialDeepL(in, d)
	for _, cs := range []int{1, 7, 4096} {
		tr := deeplStream(d)
		str, ok, why := runStream(tr, in, cs)
		if err != nil {
			require.False(t, ok, "chunk=%d: buffered failed (%v) but streaming passed\n  %s", cs, err, in)
			continue
		}
		require.True(t, ok, "chunk=%d unexpected fallback: %s\n  %s", cs, why, in)
		require.Empty(t, diffMaps(off, str), "chunk=%d\n  %s", cs, in)
		pre := tr.Protocol().(streamxform.Preluder).Prelude()
		host, herr := deeplHostFor(pre.Model)
		require.NoError(t, herr, in)
		require.Equal(t, wantHost, host, "chunk=%d host\n  %s", cs, in)
	}
}

func TestDeepLDifferential(t *testing.T) {
	d := &deeplProvider{config: ProviderConfig{targetLang: "ZH"}}
	big := strings.Repeat("d", 70000)
	for _, in := range []string{
		`{"model":"Free","messages":[{"role":"user","content":"hello"}]}`,
		`{"model":"Pro","messages":[{"role":"system","content":"S"},{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}`,
		`{"messages":[{"role":"developer","content":"D"},{"role":"user","content":"` + big + `"}],"model":"Pro"}`,
		`{"model":"Free","messages":[{"role":"system","content":"S1"},{"role":"system","content":"S2"},{"role":"user","content":[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"u"}},{"type":"text","text":"y"}]}]}`,
		`{"model":"Free","messages":[]}`,
		`{"model":"Free"}`,
		`{"model":"Free","messages":[{"role":"user","content":null},{"role":"user"},{"content":"no role"}]}`,
		`{"model":"Free","messages":[{"role":"user","content":"c<>&\"\n"}],"temperature":0.5,"stream":true,"tools":[]}`,
		`{"model":"Other","messages":[{"role":"user","content":"x"}]}`,
		`{"messages":[{"role":"user","content":"x"}]}`,
		`{"model":"Pro","messages":[{"role":"user","content":"x","name":"n"}],"max_tokens":"bad"}`,
	} {
		checkDeepL(t, d, in)
	}
}

func TestDeepLFuzz(t *testing.T) {
	d := &deeplProvider{config: ProviderConfig{targetLang: "EN"}}
	rnd := rand.New(rand.NewSource(fuzzSeed()))
	msgs := []string{
		`{"role":"system","content":"S"}`, `{"role":"developer","content":"D"}`, `{"role":"user","content":"u1"}`,
		`{"role":"assistant","content":"a1"}`, `{"content":"first","role":"user"}`, `{"role":"user","content":null}`,
		`{"role":"user","content":[{"type":"text","text":"t1"},{"type":"text","text":"` + strings.Repeat("t", 3000) + `"}]}`,
	}
	for i := 0; i < fuzzN(200); i++ {
		var sel []string
		for _, j := range rnd.Perm(len(msgs))[:rnd.Intn(len(msgs)+1)] {
			sel = append(sel, msgs[j])
		}
		model := []string{`"Free"`, `"Pro"`, `"X"`}[rnd.Intn(3)]
		fields := []string{`"model":` + model, `"messages":[` + strings.Join(sel, ",") + `]`, `"stream":true`, `"temperature":0.1`}
		var parts []string
		for _, j := range rnd.Perm(len(fields)) {
			parts = append(parts, fields[j])
		}
		checkDeepL(t, d, "{"+strings.Join(parts, ",")+"}")
	}
}
