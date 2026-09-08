package main

import (
	"testing"

	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/tidwall/gjson"

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
