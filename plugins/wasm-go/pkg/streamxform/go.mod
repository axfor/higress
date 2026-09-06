module github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform

go 1.24.1

require (
	github.com/axfor/ason v0.0.0
	github.com/higress-group/proxy-wasm-go-sdk v0.0.0-20251103120604-77e9cce339d2
)

// TODO(PR): ason 发版后改为版本号
replace github.com/axfor/ason => /Users/axx/ason
