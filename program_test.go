package ail

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestProgramBuildAndDisasm(t *testing.T) {
	p := NewProgram()
	p.EmitString(SET_MODEL, "gpt-4o")
	p.EmitFloat(SET_TEMP, 0.7)
	p.Emit(MSG_START)
	p.Emit(ROLE_SYS)
	p.EmitString(TXT_CHUNK, "Be helpful.")
	p.Emit(MSG_END)
	p.Emit(MSG_START)
	p.Emit(ROLE_USR)
	p.EmitString(TXT_CHUNK, "Hello")
	p.Emit(MSG_END)
	p.Emit(SET_STREAM)

	if p.Len() != 11 {
		t.Fatalf("expected 11 instructions, got %d", p.Len())
	}
	if m := p.GetModel(); m != "gpt-4o" {
		t.Fatalf("expected model gpt-4o, got %q", m)
	}
	if !p.IsStreaming() {
		t.Fatal("expected streaming to be true")
	}

	asm := p.Disasm()
	if !strings.Contains(asm, "SET_MODEL gpt-4o") {
		t.Fatalf("disasm missing SET_MODEL:\n%s", asm)
	}
	if !strings.Contains(asm, "ROLE_SYS") {
		t.Fatalf("disasm missing ROLE_SYS:\n%s", asm)
	}
	if !strings.Contains(asm, "TXT_CHUNK Hello") {
		t.Fatalf("disasm missing TXT_CHUNK:\n%s", asm)
	}
}

func TestProgramClone(t *testing.T) {
	a := NewProgram()
	a.EmitString(SET_MODEL, "gpt-4")
	a.Emit(MSG_START)
	a.Emit(ROLE_USR)
	a.EmitString(TXT_CHUNK, "Hi")
	a.Emit(MSG_END)

	b := a.Clone()
	b.SetModel("claude-3")

	if a.GetModel() != "gpt-4" {
		t.Fatal("clone modified original")
	}
	if b.GetModel() != "claude-3" {
		t.Fatal("clone model not set")
	}
}

func TestProgramAppend(t *testing.T) {
	a := NewProgram()
	a.EmitString(SET_MODEL, "gpt-4")

	b := NewProgram()
	b.Emit(MSG_START)
	b.Emit(ROLE_USR)
	b.EmitString(TXT_CHUNK, "Hello")
	b.Emit(MSG_END)

	combined := a.Append(b)
	if combined.Len() != 5 {
		t.Fatalf("expected 5, got %d", combined.Len())
	}
	if combined.GetModel() != "gpt-4" {
		t.Fatalf("expected gpt-4, got %s", combined.GetModel())
	}
}

func TestSetModelReplace(t *testing.T) {
	p := NewProgram()
	p.EmitString(SET_MODEL, "old-model")
	p.Emit(MSG_START)
	p.Emit(MSG_END)

	p.SetModel("new-model")
	if p.GetModel() != "new-model" {
		t.Fatalf("expected new-model, got %s", p.GetModel())
	}
	if p.Len() != 3 {
		t.Fatalf("expected 3 instructions (no extra SET_MODEL), got %d", p.Len())
	}
}

func TestBufferSideChannel(t *testing.T) {
	p := NewProgram()
	imgData := []byte("fake-png-data")
	ref := p.AddBuffer(imgData)
	p.Emit(MSG_START)
	p.Emit(ROLE_USR)
	p.EmitRef(IMG_REF, ref)
	p.Emit(MSG_END)

	if p.Buffers[ref] == nil {
		t.Fatal("buffer not stored")
	}
	if string(p.Buffers[ref]) != "fake-png-data" {
		t.Fatalf("buffer mismatch: %s", p.Buffers[ref])
	}
}

func TestCloneDeepCopyJSON(t *testing.T) {
	a := NewProgram()
	a.EmitJSON(CALL_ARGS, json.RawMessage(`{"key":"original"}`))

	b := a.Clone()

	// Mutate the clone's JSON
	b.Code[0].JSON = json.RawMessage(`{"key":"mutated"}`)

	// Original must be unaffected
	if string(a.Code[0].JSON) != `{"key":"original"}` {
		t.Fatalf("clone mutated original JSON: got %s", a.Code[0].JSON)
	}
}

func TestCloneDeepCopyJSONUnderlying(t *testing.T) {
	a := NewProgram()
	a.EmitJSON(CALL_ARGS, json.RawMessage(`{"key":"original"}`))

	b := a.Clone()

	// Mutate byte by byte in the clone to verify underlying array is separate
	copy(b.Code[0].JSON, []byte(`{"key":"XXXXXXXX"}`))

	if string(a.Code[0].JSON) != `{"key":"original"}` {
		t.Fatalf("clone shares underlying byte array with original: got %s", a.Code[0].JSON)
	}
}
