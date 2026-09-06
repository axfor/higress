module github.com/alibaba/higress/plugins/wasm-go/examples/jwt-auth

go 1.24.1

toolchain go1.24.4

require (
	github.com/go-jose/go-jose/v3 v3.0.3
	github.com/higress-group/proxy-wasm-go-sdk v0.0.0-20251103120604-77e9cce339d2
	github.com/higress-group/wasm-go v1.0.2-0.20250821081215-b573359becf8
	github.com/tidwall/gjson v1.18.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/stretchr/testify v1.9.0 // indirect
	github.com/tetratelabs/wazero v1.7.2 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/resp v0.1.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	golang.org/x/crypto v0.26.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/higress-group/wasm-go => github.com/axfor/wasm-go v1.1.3-0.20260906150907-5858ed563432
