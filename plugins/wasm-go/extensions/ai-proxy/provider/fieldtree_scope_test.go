package provider

import (
	"encoding/json"
	"math/rand"
	"testing"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

// 嵌套校验值不值得做，取决于它相对根级能多抓多少 —— 而这个倍数必须用真实的
// chatCompletionRequest 语料来量，不能用别处的合成语料推。
// 这里只做离线判定，引擎里没有接类型树，行为不变。

func walkTree(t *streamxform.FieldTree, v any) bool {
	if t == nil || t.Any {
		return true
	}
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return t.Types&streamxform.TypeString != 0
	case float64:
		return t.Types&streamxform.TypeNumber != 0
	case bool:
		return t.Types&streamxform.TypeBool != 0
	case map[string]any:
		if t.Types&streamxform.TypeObject == 0 {
			return false
		}
		for k, vv := range x {
			if t.Keys != nil {
				if sub, ok := t.Keys[k]; ok && !walkTree(sub, vv) {
					return false
				}
				continue
			}
			if !walkTree(t.Elem, vv) {
				return false
			}
		}
		return true
	case []any:
		if t.Types&streamxform.TypeArray == 0 {
			return false
		}
		for _, vv := range x {
			if !walkTree(t.Elem, vv) {
				return false
			}
		}
		return true
	}
	return true
}

func TestNestedCheckScopeOnRealCorpus(t *testing.T) {
	tree := streamxform.FieldTreeOf(&chatCompletionRequest{}, 6)
	flat := chatRequestFieldTypes
	r := rand.New(rand.NewSource(fuzzSeed()))

	total, bufFail, rootCatch, treeCatch, falseReject := 0, 0, 0, 0, 0
	for i := 0; i < 20000; i++ {
		in := genRequest(r)
		var generic any
		if json.Unmarshal([]byte(in), &generic) != nil {
			continue
		}
		total++
		req := &chatCompletionRequest{}
		unmarshalOK := json.Unmarshal([]byte(in), req) == nil
		treeOK := walkTree(tree, generic)

		if unmarshalOK && !treeOK {
			falseReject++
			if falseReject <= 3 {
				t.Errorf("误拒：Unmarshal 接受而按树拒绝\n  %s", in)
			}
			continue
		}
		if unmarshalOK {
			continue
		}
		bufFail++
		if !treeOK {
			treeCatch++
		}
		// 根级能否抓到
		if m, ok := generic.(map[string]any); ok {
			for k, v := range m {
				want, known := flat[k]
				if !known {
					continue
				}
				var bit streamxform.FieldTypes
				switch v.(type) {
				case string:
					bit = streamxform.TypeString
				case float64:
					bit = streamxform.TypeNumber
				case bool:
					bit = streamxform.TypeBool
				case map[string]any:
					bit = streamxform.TypeObject
				case []any:
					bit = streamxform.TypeArray
				default:
					continue
				}
				if want&bit == 0 {
					rootCatch++
					break
				}
			}
		}
	}
	if falseReject > 0 {
		t.Fatalf("误拒 %d 例：有这个数就不能谈上线嵌套校验", falseReject)
	}
	t.Logf("%d 例中缓冲路径拒绝 %d 例：根级校验能抓 %d 例（%.1f%%），按树校验能抓 %d 例（%.1f%%），倍数 %.2f",
		total, bufFail, rootCatch, 100*float64(rootCatch)/float64(bufFail),
		treeCatch, 100*float64(treeCatch)/float64(bufFail), float64(treeCatch)/float64(max(rootCatch, 1)))
}
