package provider

// Differential tests of the multipart transformer against defaultTransformMultipartRequestBody's two outcomes:
// the body untouched when the mapping changes nothing, rewriteMultipartFormModel's re-encoding when it does.
// Hand-written bodies and a random corpus, at chunk sizes 1, 7 and 4096, compared byte for byte.

import (
	"bytes"
	"fmt"
	"math/rand"
	"mime/multipart"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

var mpMapping = map[string]string{"m": "gpt-image-1", "same": "same", "empty": ""}

// officialMultipart reproduces defaultTransformMultipartRequestBody without the host.
func officialMultipart(body []byte, contentType string) ([]byte, string, error) {
	model, err := extractMultipartModel(body, contentType)
	if err != nil {
		return nil, "", err
	}
	mapped := getMappedModel(model, mpMapping)
	if mapped == model || (mapped == "" && model == "") {
		return body, model, nil
	}
	out, err := rewriteMultipartFormModel(body, contentType, mapped)
	return out, model, err
}

func runMultipart(t *testing.T, boundary string, body []byte, chunk int) ([]byte, bool, string) {
	t.Helper()
	tr := streamxform.NewMultipart(streamxform.MultipartOptions{Boundary: boundary, MapModel: func(m string) string { return getMappedModel(m, mpMapping) }})
	var out []byte
	for i := 0; i < len(body); i += chunk {
		j := i + chunk
		if j > len(body) {
			j = len(body)
		}
		tr.Write(body[i:j])
		out = append(out, tr.Out()...)
	}
	out = append(out, tr.Finish()...)
	if bad, why := tr.Unsupported(); bad {
		return nil, false, why
	}
	return out, true, ""
}

func checkMultipart(t *testing.T, name, boundary string, body []byte) {
	t.Helper()
	ct := "multipart/form-data; boundary=" + boundary
	off, model, err := officialMultipart(body, ct)
	for _, chunk := range []int{1, 7, 4096} {
		str, ok, why := runMultipart(t, boundary, body, chunk)
		if err != nil {
			require.False(t, ok, "%s chunk=%d: buffered failed (%v) but streaming passed", name, chunk, err)
			continue
		}
		if !ok {
			// a fallback: the buffered path runs, so it is acceptable only where that leaves the body as it came
			require.Equal(t, string(body), string(off), "%s chunk=%d: fallback (%s) but the buffered path changes the body", name, chunk, why)
			continue
		}
		require.Equal(t, string(off), string(str), "%s chunk=%d (model %q)", name, chunk, model)
	}
}

type mpPart struct{ name, filename, content string }

// buildParts writes a body with Go's writer in the given order (SDKs send the fields first, curl in argument order).
func buildParts(t *testing.T, boundary string, parts []mpPart) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	require.NoError(t, w.SetBoundary(boundary))
	for _, p := range parts {
		if p.filename == "" {
			require.NoError(t, w.WriteField(p.name, p.content))
			continue
		}
		f, err := w.CreateFormFile(p.name, p.filename)
		require.NoError(t, err)
		_, _ = f.Write([]byte(p.content))
	}
	require.NoError(t, w.Close())
	return buf.Bytes()
}

// formFields decodes a body with Go's reader: the semantic view a server would take.
func formFields(t *testing.T, boundary string, body []byte) map[string][]string {
	t.Helper()
	r := multipart.NewReader(bytes.NewReader(body), boundary)
	out := map[string][]string{}
	for {
		p, err := r.NextPart()
		if err != nil {
			break
		}
		var b bytes.Buffer
		_, _ = b.ReadFrom(p)
		out[p.FormName()] = append(out[p.FormName()], b.String())
	}
	return out
}

// build writes a body with Go's writer, the way SDKs do.
func buildMultipart(t *testing.T, boundary string, fields [][2]string, files [][3]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	require.NoError(t, w.SetBoundary(boundary))
	for _, f := range fields {
		require.NoError(t, w.WriteField(f[0], f[1]))
	}
	for _, f := range files {
		p, err := w.CreateFormFile(f[0], f[1])
		require.NoError(t, err)
		_, _ = p.Write([]byte(f[2]))
	}
	require.NoError(t, w.Close())
	return buf.Bytes()
}

func TestMultipartDifferential(t *testing.T) {
	const b = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
	png := "\x89PNG\r\n\x1a\n" + strings.Repeat("\x00\xffbinary\r\n--not-a-boundary\r\n", 3000)
	for _, c := range []struct {
		name string
		body []byte
	}{
		{"mapped, fields first", buildMultipart(t, b, [][2]string{{"model", "m"}, {"prompt", "make it blue"}, {"n", "1"}}, [][3]string{{"image", "a.png", png}})},
		{"unchanged, fields first", buildMultipart(t, b, [][2]string{{"model", "same"}, {"prompt", "p"}}, [][3]string{{"image", "a.png", png}})},
		{"no mapping match", buildMultipart(t, b, [][2]string{{"model", "dall-e-2"}, {"prompt", "p"}}, [][3]string{{"image", "a.png", png}})},
		{"no model, image only", buildMultipart(t, b, nil, [][3]string{{"image", "a.png", "tiny"}})},
		{"model after a small image", buildParts(t, b, []mpPart{{"image", "a.png", "tiny"}, {"prompt", "", "p"}, {"model", "", "m"}})},
		{"model after a small image, unchanged", buildParts(t, b, []mpPart{{"image", "a.png", "tiny"}, {"model", "", "same"}})},
		{"two images and a mask", buildMultipart(t, b, [][2]string{{"model", "m"}, {"prompt", "p"}, {"size", "1024x1024"}}, [][3]string{{"image[]", "a.png", png}, {"image[]", "b.png", "second"}, {"mask", "m.png", "maskdata"}})},
		{"model to empty", buildMultipart(t, b, [][2]string{{"model", "empty"}, {"prompt", "p"}}, [][3]string{{"image", "a.png", "x"}})},
		{"model with spaces", []byte("--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n  m \r\n--" + b + "\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\np\r\n--" + b + "--\r\n")},
		{"preamble and epilogue", []byte("preamble text\r\n--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n--" + b + "\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\np\r\n--" + b + "--\r\nepilogue")},
		{"preamble, unchanged", []byte("preamble text\r\n--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nsame\r\n--" + b + "--\r\n")},
		{"lower-case headers and extra header", []byte("--" + b + "\r\ncontent-disposition: form-data; name=\"model\"\r\nx-extra: 1\r\ncontent-type: text/plain\r\n\r\nm\r\n--" + b + "\r\ncontent-disposition: form-data; name=\"image\"; filename=\"a.png\"\r\ncontent-type: image/png\r\n\r\n" + png + "\r\n--" + b + "--\r\n")},
		{"padding after boundary", []byte("--" + b + " \t\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n--" + b + "--  \r\n")},
		{"part without disposition", []byte("--" + b + "\r\nX-Nothing: 1\r\n\r\nstray\r\n--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n--" + b + "--\r\n")},
		{"part without headers", []byte("--" + b + "\r\n\r\nstray\r\n--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n--" + b + "--\r\n")},
		{"attachment disposition is not a field", []byte("--" + b + "\r\nContent-Disposition: attachment; name=\"model\"\r\n\r\nm\r\n--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n--" + b + "--\r\n")},
		{"empty model", buildMultipart(t, b, [][2]string{{"model", ""}, {"prompt", "p"}}, nil)},
		{"no closing boundary", []byte("--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n")},
		{"empty body", []byte("")},
		{"only closing", []byte("--" + b + "--\r\n")},
	} {
		checkMultipart(t, c.name, b, c.body)
	}
}

// Shapes the transformer declines and the buffered path handles: a bare-LF body, a second model field, a model
// field beyond the window.
func TestMultipartUnsupported(t *testing.T) {
	const b = "xyz"
	for _, c := range []struct{ name, body, why string }{
		{"bare LF", "--" + b + "\nContent-Disposition: form-data; name=\"model\"\n\nm\n--" + b + "--\n", "LF"},
		{"second model", "--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm2\r\n--" + b + "--\r\n", "second model"},
		{"huge model", "--" + b + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n" + strings.Repeat("m", 5000) + "\r\n--" + b + "--\r\n", "4KB"},
		{"huge headers", "--" + b + "\r\nX: " + strings.Repeat("h", 9000) + "\r\n\r\nm\r\n--" + b + "--\r\n", "8KB"},
	} {
		_, ok, why := runMultipart(t, b, []byte(c.body), 7)
		require.False(t, ok, c.name)
		require.Contains(t, why, c.why, c.name)
	}
	// the model after a large image: the window fills first, the rendering is fixed in place; with the plan's
	// RequireModelBeforeCommit that is a fallback on the gateway, here the in-place rendering is checked
	big := strings.Repeat("i", 100000)
	body := buildMultipart(t, b, [][2]string{{"prompt", "p"}}, [][3]string{{"image", "a.png", big}})
	body = append(body[:len(body)-len("\r\n--"+b+"--\r\n")], "\r\n--"+b+"\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n--"+b+"--\r\n"...)
	out, ok, why := runMultipart(t, b, body, 4096)
	require.True(t, ok, why)
	require.Contains(t, string(out), "name=\"model\"\r\n\r\ngpt-image-1\r\n--"+b+"--")
	require.Contains(t, string(out), big)
	off, _, err := officialMultipart(body, "multipart/form-data; boundary="+b)
	require.NoError(t, err)
	require.Equal(t, formFields(t, b, off), formFields(t, b, out), "the same form, whatever the spelling")
}

func TestMultipartFuzz(t *testing.T) {
	r := rand.New(rand.NewSource(58))
	pick := func(xs ...string) string { return xs[r.Intn(len(xs))] }
	for i := 0; i < 300; i++ {
		b := pick("----WebKitFormBoundaryABC", "xyz", "b0uNd-ary_123")
		var fields [][2]string
		var files [][3]string
		if r.Intn(6) != 0 {
			fields = append(fields, [2]string{"model", pick("m", "same", "empty", "dall-e-2", " m ", "")})
		}
		for j := r.Intn(4); j > 0; j-- {
			fields = append(fields, [2]string{pick("prompt", "n", "size", "response_format", "user"), pick("p", "1", "1024x1024", "b64_json", "line\r\nbreak", "")})
		}
		for j := r.Intn(3); j > 0; j-- {
			n := r.Intn(3000)
			if r.Intn(10) == 0 {
				n = 70000
			}
			content := make([]byte, n)
			for k := range content {
				content[k] = byte(r.Intn(256))
			}
			files = append(files, [3]string{pick("image", "image[]", "mask"), pick("a.png", "b.jpg", ""), string(content)})
		}
		r.Shuffle(len(fields), func(a, c int) { fields[a], fields[c] = fields[c], fields[a] })
		body := buildMultipart(t, b, fields, files)
		if r.Intn(5) == 0 {
			body = append([]byte("pre\r\n"), body...)
		}
		checkMultipart(t, fmt.Sprintf("fuzz %d", i), b, body)
	}
}
