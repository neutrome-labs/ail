package ail

import (
	"bytes"
	"testing"
)

func TestAsmDisasmRoundTrip(t *testing.T) {
	// Build a program manually
	prog := NewProgram()
	prog.EmitString(SET_MODEL, "openai/gpt-4")
	prog.Emit(SET_STREAM)
	prog.Emit(MSG_START)
	prog.Emit(ROLE_USR)
	prog.EmitString(TXT_CHUNK, "Hello, world!")
	prog.Emit(MSG_END)

	// Disassemble, then reassemble
	text := prog.Disasm()
	t.Logf("Disasm:\n%s", text)

	got, err := Asm(text)
	if err != nil {
		t.Fatalf("Asm failed: %v", err)
	}

	// Compare instruction counts
	if len(got.Code) != len(prog.Code) {
		t.Fatalf("instruction count mismatch: got %d, want %d", len(got.Code), len(prog.Code))
	}

	// Compare each instruction
	for i, want := range prog.Code {
		g := got.Code[i]
		if g.Op != want.Op {
			t.Errorf("inst %d: op %s != %s", i, g.Op, want.Op)
		}
		if g.Str != want.Str {
			t.Errorf("inst %d: str %q != %q", i, g.Str, want.Str)
		}
		if g.Num != want.Num {
			t.Errorf("inst %d: num %v != %v", i, g.Num, want.Num)
		}
		if g.Int != want.Int {
			t.Errorf("inst %d: int %v != %v", i, g.Int, want.Int)
		}
	}
}

func TestAsmWithComment(t *testing.T) {
	text := `; This is a comment
SET_MODEL test-model
MSG_START
  ROLE_SYS
  TXT_CHUNK You are helpful
MSG_END
`
	prog, err := Asm(text)
	if err != nil {
		t.Fatalf("Asm failed: %v", err)
	}

	// Comments are silently skipped — first instruction should be SET_MODEL
	if prog.Code[0].Op != SET_MODEL || prog.Code[0].Str != "test-model" {
		t.Errorf("expected SET_MODEL, got %s", prog.Code[0].Op)
	}
	if len(prog.Code) != 5 {
		t.Errorf("expected 5 instructions (comment skipped), got %d", len(prog.Code))
	}
}

func TestAsmNumericArgs(t *testing.T) {
	text := `SET_TEMP 0.7000
SET_TOPP 0.9500
SET_MAX 1024
`
	prog, err := Asm(text)
	if err != nil {
		t.Fatalf("Asm failed: %v", err)
	}

	if prog.Code[0].Op != SET_TEMP || prog.Code[0].Num != 0.7 {
		t.Errorf("SET_TEMP: %+v", prog.Code[0])
	}
	if prog.Code[1].Op != SET_TOPP || prog.Code[1].Num != 0.95 {
		t.Errorf("SET_TOPP: %+v", prog.Code[1])
	}
	if prog.Code[2].Op != SET_MAX || prog.Code[2].Int != 1024 {
		t.Errorf("SET_MAX: %+v", prog.Code[2])
	}
}

func TestAsmJSON(t *testing.T) {
	text := `USAGE {"completion_tokens":14,"prompt_tokens":21,"total_tokens":35}
`
	prog, err := Asm(text)
	if err != nil {
		t.Fatalf("Asm failed: %v", err)
	}

	if prog.Code[0].Op != USAGE {
		t.Errorf("expected USAGE, got %s", prog.Code[0].Op)
	}
	if string(prog.Code[0].JSON) != `{"completion_tokens":14,"prompt_tokens":21,"total_tokens":35}` {
		t.Errorf("JSON mismatch: %s", prog.Code[0].JSON)
	}
}

func TestAsmSetMeta(t *testing.T) {
	text := `SET_META key value
`
	prog, err := Asm(text)
	if err != nil {
		t.Fatalf("Asm failed: %v", err)
	}

	if prog.Code[0].Op != SET_META || prog.Code[0].Key != "key" || prog.Code[0].Str != "value" {
		t.Errorf("SET_META: %+v", prog.Code[0])
	}
}

func TestAsmExtData(t *testing.T) {
	text := `EXT_DATA provider {"foo":"bar"}
`
	prog, err := Asm(text)
	if err != nil {
		t.Fatalf("Asm failed: %v", err)
	}

	if prog.Code[0].Op != EXT_DATA || prog.Code[0].Key != "provider" {
		t.Errorf("EXT_DATA: %+v", prog.Code[0])
	}
	if string(prog.Code[0].JSON) != `{"foo":"bar"}` {
		t.Errorf("EXT_DATA JSON: %s", prog.Code[0].JSON)
	}
}

func TestAsmRefs(t *testing.T) {
	text := `IMG_REF ref:0
AUD_REF ref:1
TXT_REF ref:2
`
	prog, err := Asm(text)
	if err != nil {
		t.Fatalf("Asm failed: %v", err)
	}

	if prog.Code[0].Op != IMG_REF || prog.Code[0].Ref != 0 {
		t.Errorf("IMG_REF: %+v", prog.Code[0])
	}
	if prog.Code[1].Op != AUD_REF || prog.Code[1].Ref != 1 {
		t.Errorf("AUD_REF: %+v", prog.Code[1])
	}
	if prog.Code[2].Op != TXT_REF || prog.Code[2].Ref != 2 {
		t.Errorf("TXT_REF: %+v", prog.Code[2])
	}
}

func TestAsmFullRoundTrip(t *testing.T) {
	// Build a complex program
	orig := NewProgram()
	orig.EmitString(SET_MODEL, "openai/gpt-4")
	orig.EmitFloat(SET_TEMP, 0.7)
	orig.EmitFloat(SET_TOPP, 0.95)
	orig.EmitInt(SET_MAX, 2048)
	orig.Emit(SET_STREAM)
	orig.EmitKeyVal(SET_META, "user_id", "usr_123")
	orig.Emit(DEF_START)
	orig.EmitString(DEF_NAME, "get_weather")
	orig.EmitString(DEF_DESC, "Get weather info")
	orig.EmitJSON(DEF_SCHEMA, []byte(`{"type":"object","properties":{"city":{"type":"string"}}}`))
	orig.Emit(DEF_END)
	orig.Emit(MSG_START)
	orig.Emit(ROLE_SYS)
	orig.EmitString(TXT_CHUNK, "You are a helpful assistant.")
	orig.Emit(MSG_END)
	orig.Emit(MSG_START)
	orig.Emit(ROLE_USR)
	orig.EmitString(TXT_CHUNK, "What's the weather in Paris?")
	orig.Emit(MSG_END)

	// Disasm → Asm → verify binary round-trip
	text := orig.Disasm()
	reassembled, err := Asm(text)
	if err != nil {
		t.Fatalf("Asm failed: %v", err)
	}

	// Verify by binary encoding both and comparing
	var origBuf, reassemBuf bytes.Buffer
	if err := orig.Encode(&origBuf); err != nil {
		t.Fatalf("encode orig: %v", err)
	}
	if err := reassembled.Encode(&reassemBuf); err != nil {
		t.Fatalf("encode reassembled: %v", err)
	}

	if !bytes.Equal(origBuf.Bytes(), reassemBuf.Bytes()) {
		t.Errorf("binary mismatch after Disasm→Asm round-trip")
		t.Logf("Original disasm:\n%s", text)
		t.Logf("Reassembled disasm:\n%s", reassembled.Disasm())
	}
}

func TestAsmSampleFile(t *testing.T) {
	// Replicate the exact format of a sample .ail.txt file
	text := `SET_MODEL semantyka/enei-1-chat+slwin
SET_STREAM
MSG_START
  ROLE_USR
  TXT_CHUNK How many r` + "`" + `s are in the word ` + "`" + `strawberry?` + "`" + `
MSG_END
`
	prog, err := Asm(text)
	if err != nil {
		t.Fatalf("Asm failed: %v", err)
	}

	if prog.GetModel() != "semantyka/enei-1-chat+slwin" {
		t.Errorf("model: %q", prog.GetModel())
	}
	if !prog.IsStreaming() {
		t.Error("expected streaming")
	}
	if len(prog.Code) != 6 {
		t.Errorf("expected 6 instructions, got %d", len(prog.Code))
	}
}

func TestAsmInvalidOpcode(t *testing.T) {
	_, err := Asm("INVALID_OP\n")
	if err == nil {
		t.Error("expected error for unknown opcode")
	}
}
