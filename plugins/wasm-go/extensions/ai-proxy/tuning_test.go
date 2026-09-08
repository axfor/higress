package main

import (
	"testing"

	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/tidwall/gjson"

	"github.com/alibaba/higress/plugins/wasm-go/extensions/ai-proxy/provider"
	"github.com/alibaba/higress/plugins/wasm-go/pkg/streamxform"
)

// 宿主之外没有插件日志实例，注入一个空实现，否则 log.Infof 会解引用空指针。
type quietLog struct{}

func (quietLog) Trace(string)                       {}
func (quietLog) Tracef(string, ...interface{})      {}
func (quietLog) Debug(string)                       {}
func (quietLog) Debugf(string, ...interface{})      {}
func (quietLog) Info(string)                        {}
func (quietLog) Infof(string, ...interface{})       {}
func (quietLog) UnsafeInfo(string)                  {}
func (quietLog) UnsafeInfof(string, ...interface{}) {}
func (quietLog) Warn(string)                        {}
func (quietLog) Warnf(string, ...interface{})       {}
func (quietLog) Error(string)                       {}
func (quietLog) Errorf(string, ...interface{})      {}
func (quietLog) Critical(string)                    {}
func (quietLog) Criticalf(string, ...interface{})   {}
func (quietLog) ResetID(string)                     {}

// 两个旋钮都有实测过的收益，但错误的取值会安静地毁掉正确性或内存表现，
// 所以越界值必须被拒绝并保留原值，而不是悄悄夹到边界上。
func TestApplyStreamTuning(t *testing.T) {
	log.SetPluginLog(quietLog{})
	reset := func() {
		streamCommitWindowBytes = 0
		wrapper.GCWatchdogFloor = 64 << 20
	}
	t.Cleanup(reset)

	cases := []struct {
		name       string
		json       string
		wantWindow int
		wantFloor  uint64
	}{
		{"缺省不动", `{}`, streamxform.CommitBytes, 64 << 20},
		{"都设置", `{"streamCommitWindowBytes":16384,"streamGcFloorBytes":33554432}`, 16384, 32 << 20},
		{"零表示恢复默认", `{"streamCommitWindowBytes":0,"streamGcFloorBytes":0}`, streamxform.CommitBytes, 64 << 20},
		{"窗口过小被拒", `{"streamCommitWindowBytes":100}`, streamxform.CommitBytes, 64 << 20},
		{"窗口过大被拒", `{"streamCommitWindowBytes":8388608}`, streamxform.CommitBytes, 64 << 20},
		{"下限过小被拒", `{"streamGcFloorBytes":1024}`, streamxform.CommitBytes, 64 << 20},
		{"下限过大被拒", `{"streamGcFloorBytes":1073741824}`, streamxform.CommitBytes, 64 << 20},
		{"只设窗口", `{"streamCommitWindowBytes":65536}`, 65536, 64 << 20},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			reset()
			applyStreamTuning(gjson.Parse(c.json))
			if got := effectiveCommitWindow(); got != c.wantWindow {
				t.Errorf("提交窗口 = %d, 应为 %d", got, c.wantWindow)
			}
			if wrapper.GCWatchdogFloor != c.wantFloor {
				t.Errorf("GC 下限 = %d, 应为 %d", wrapper.GCWatchdogFloor, c.wantFloor)
			}
		})
	}
}

// 越界值被拒之后，先前设置好的值必须还在 —— 拒绝不能顺手把状态清掉。
func TestApplyStreamTuningKeepsPreviousOnBadValue(t *testing.T) {
	log.SetPluginLog(quietLog{})
	t.Cleanup(func() { streamCommitWindowBytes = 0; wrapper.GCWatchdogFloor = 64 << 20 })
	applyStreamTuning(gjson.Parse(`{"streamCommitWindowBytes":16384,"streamGcFloorBytes":33554432}`))
	applyStreamTuning(gjson.Parse(`{"streamCommitWindowBytes":1,"streamGcFloorBytes":1}`))
	if effectiveCommitWindow() != 16384 {
		t.Errorf("越界值不该覆盖先前的窗口，得到 %d", effectiveCommitWindow())
	}
	if wrapper.GCWatchdogFloor != 32<<20 {
		t.Errorf("越界值不该覆盖先前的下限，得到 %d", wrapper.GCWatchdogFloor)
	}
}

// 类型校验的开关是为灰度准备的：它会让网关变严格，一个字段类型写错的请求从"转发给供应商"
// 变成"网关直接 500"。虽然那本来就是缓冲路径一直以来的行为，但对已经依赖当前宽松行为的调用方
// 是可感知的变化，所以必须能在不回滚构建的前提下关掉。
func TestStreamTypeCheckSwitch(t *testing.T) {
	log.SetPluginLog(quietLog{})
	t.Cleanup(func() { streamTypeCheck = true; provider.ChatRequestTypeCheck = true })

	applyStreamTuning(gjson.Parse(`{"streamTypeCheck":false}`))
	if streamTypeCheck || provider.ChatRequestTypeCheck {
		t.Fatal("关不掉")
	}
	applyStreamTuning(gjson.Parse(`{"streamTypeCheck":true}`))
	if !streamTypeCheck || !provider.ChatRequestTypeCheck {
		t.Fatal("开不回来")
	}
	// 没写这个字段时不该改变现状
	provider.ChatRequestTypeCheck = false
	applyStreamTuning(gjson.Parse(`{"streamCommitWindowBytes":16384}`))
	if provider.ChatRequestTypeCheck {
		t.Fatal("没写这个字段却被改动了")
	}
}

// 准入上限必须跟着提交窗口走：受约束的是"在传请求数 × 每请求持有字节"，
// 用请求数表达上限，窗口一变它就悄悄变成了另一个意思。
func TestAdmissionLimitDerivedFromBudget(t *testing.T) {
	log.SetPluginLog(quietLog{})
	t.Cleanup(func() {
		streamCommitWindowBytes, admBudgetBytes, admOverride = 0, 32<<20, 0
	})

	cases := []struct {
		name      string
		json      string
		wantLimit float64
	}{
		{"默认 32MB / 64KB", `{}`, 512},
		{"窗口翻四倍，上限降四倍", `{"streamCommitWindowBytes":262144}`, 128},
		{"预算翻倍", `{"streamInflightBudgetBytes":67108864}`, 1024},
		{"预算与窗口同时给", `{"streamInflightBudgetBytes":67108864,"streamCommitWindowBytes":262144}`, 256},
		{"小预算撞下限", `{"streamInflightBudgetBytes":1048576,"streamCommitWindowBytes":1048576}`, admMin},
		{"大预算撞上限", `{"streamInflightBudgetBytes":1073741824,"streamCommitWindowBytes":4096}`, admMax},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			streamCommitWindowBytes, admBudgetBytes, admOverride = 0, 32<<20, 0
			applyStreamTuning(gjson.Parse(c.json))
			if got := admLimit(); got != c.wantLimit {
				t.Errorf("上限 = %.0f，应为 %.0f（窗口 %d，预算 %d）",
					got, c.wantLimit, effectiveCommitWindow(), admBudgetBytes)
			}
		})
	}
}

// 无论怎么配，在传请求持有的字节都不该超过预算 —— 这是这个上限存在的全部理由。
func TestAdmissionBudgetIsNeverExceeded(t *testing.T) {
	log.SetPluginLog(quietLog{})
	t.Cleanup(func() { streamCommitWindowBytes, admBudgetBytes, admOverride = 0, 32<<20, 0 })
	for _, win := range []int{4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20} {
		for _, budget := range []int{4 << 20, 32 << 20, 128 << 20} {
			streamCommitWindowBytes, admBudgetBytes, admOverride = win, budget, 0
			held := admLimit() * float64(win)
			// 撞到 admMin 时会超预算：那是刻意的下限，不能让上限低到完全不放行
			if held > float64(budget) && admLimit() != admMin {
				t.Errorf("窗口 %d 预算 %d：持有 %.0f 字节超出预算", win, budget, held)
			}
		}
	}
}

// 准入把超出上限的请求赶去缓冲路径，而缓冲路径持有的是整份 body、流式持有的是一个窗口。
// 所以"回落"只在两者相当时才划算；body 远大于窗口时回落反而更费内存，实测在 1MB/256KB 上
// 多花了 170MB 的 Envoy 堆并损失三分之一吞吐。
func TestAdmissionDivertsOnlyWhenBufferingIsNotMoreExpensive(t *testing.T) {
	log.SetPluginLog(quietLog{})
	t.Cleanup(func() { streamCommitWindowBytes = 0 })

	cases := []struct {
		name       string
		window     int
		declared   int64
		wantDivert bool
	}{
		{"体积未声明：维持原行为", 64 << 10, 0, true},
		{"body 等于窗口", 64 << 10, 64 << 10, true},
		{"body 两倍窗口：仍在可比范围", 64 << 10, 128 << 10, true},
		{"body 略超两倍：不再回落", 64 << 10, (128 << 10) + 1, false},
		{"1MB body 对 256KB 窗口", 256 << 10, 1 << 20, false},
		{"1MB body 对 1MB 窗口", 1 << 20, 1 << 20, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			streamCommitWindowBytes = c.window
			ctx := newTuningCtx()
			if c.declared > 0 {
				ctx.SetContext(ctxKeyDeclaredBodyBytes, c.declared)
			}
			if got := admDivert(ctx); got != c.wantDivert {
				t.Errorf("回落判定 = %v，应为 %v（窗口 %d，声明体积 %d）",
					got, c.wantDivert, c.window, c.declared)
			}
		})
	}
}

// 只需要上下文读写的最小实现。
type tuningCtx struct {
	wrapper.HttpContext
	m map[string]interface{}
}

func newTuningCtx() *tuningCtx                          { return &tuningCtx{m: map[string]interface{}{}} }
func (c *tuningCtx) SetContext(k string, v interface{}) { c.m[k] = v }
func (c *tuningCtx) GetContext(k string) interface{}    { return c.m[k] }
