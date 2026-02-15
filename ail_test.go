package ail

import (
	"bytes"
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
	if !strings.Contains(asm, `SET_MODEL "gpt-4o"`) {
		t.Fatalf("disasm missing SET_MODEL:\n%s", asm)
	}
	if !strings.Contains(asm, `ROLE_SYS`) {
		t.Fatalf("disasm missing ROLE_SYS:\n%s", asm)
	}
	if !strings.Contains(asm, `TXT_CHUNK "Hello"`) {
		t.Fatalf("disasm missing TXT_CHUNK:\n%s", asm)
	}
}

func TestBinaryRoundTrip(t *testing.T) {
	orig := NewProgram()
	orig.EmitString(SET_MODEL, "claude-3")
	orig.EmitFloat(SET_TEMP, 0.5)
	orig.EmitInt(SET_MAX, 4096)
	orig.Emit(SET_STREAM)
	orig.Emit(MSG_START)
	orig.Emit(ROLE_USR)
	orig.EmitString(TXT_CHUNK, "Hello world")

	// Add image buffer reference
	imgRef := orig.AddBuffer([]byte("fake-image-data-base64"))
	orig.EmitKeyVal(SET_META, "media_type", "image/jpeg")
	orig.EmitRef(IMG_REF, imgRef)

	orig.Emit(MSG_END)

	// Tool definition
	orig.Emit(DEF_START)
	orig.EmitString(DEF_NAME, "get_weather")
	orig.EmitString(DEF_DESC, "Get weather for a location")
	schema := json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`)
	orig.EmitJSON(DEF_SCHEMA, schema)
	orig.Emit(DEF_END)

	// Tool call
	orig.EmitString(CALL_START, "call_123")
	orig.EmitString(CALL_NAME, "get_weather")
	args := json.RawMessage(`{"location":"NYC"}`)
	orig.EmitJSON(CALL_ARGS, args)
	orig.Emit(CALL_END)

	// Tool result
	orig.EmitString(RESULT_START, "call_123")
	orig.EmitString(RESULT_DATA, "72°F, sunny")
	orig.Emit(RESULT_END)

	// Meta and ext
	orig.EmitKeyVal(SET_META, "user", "test-user")
	orig.EmitKeyJSON(EXT_DATA, "response_format", json.RawMessage(`{"type":"json_object"}`))

	// Encode
	var buf bytes.Buffer
	if err := orig.Encode(&buf); err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Decode
	decoded, err := Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if len(decoded.Code) != len(orig.Code) {
		t.Fatalf("instruction count: got %d, want %d", len(decoded.Code), len(orig.Code))
	}

	// Verify buffers survived round-trip
	if len(decoded.Buffers) != len(orig.Buffers) {
		t.Fatalf("buffer count: got %d, want %d", len(decoded.Buffers), len(orig.Buffers))
	}
	for i, got := range decoded.Buffers {
		want := orig.Buffers[i]
		if string(got) != string(want) {
			t.Errorf("buffer %d: got %q, want %q", i, got, want)
		}
	}

	for i, got := range decoded.Code {
		want := orig.Code[i]
		if got.Op != want.Op {
			t.Errorf("inst %d: op 0x%02X != 0x%02X", i, got.Op, want.Op)
		}
		if got.Str != want.Str {
			t.Errorf("inst %d (%s): str %q != %q", i, want.Op, got.Str, want.Str)
		}
		if got.Num != want.Num {
			t.Errorf("inst %d (%s): num %f != %f", i, want.Op, got.Num, want.Num)
		}
		if got.Int != want.Int {
			t.Errorf("inst %d (%s): int %d != %d", i, want.Op, got.Int, want.Int)
		}
		if got.Key != want.Key {
			t.Errorf("inst %d (%s): key %q != %q", i, want.Op, got.Key, want.Key)
		}
		if got.Ref != want.Ref {
			t.Errorf("inst %d (%s): ref %d != %d", i, want.Op, got.Ref, want.Ref)
		}
		if string(got.JSON) != string(want.JSON) {
			t.Errorf("inst %d (%s): json %s != %s", i, want.Op, got.JSON, want.JSON)
		}
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

func TestStreamAssemblerTextOnly(t *testing.T) {
	asm := NewStreamAssembler()

	// Chunk 1: stream start with metadata
	c1 := NewProgram()
	c1.Emit(STREAM_START)
	c1.EmitString(RESP_ID, "resp-123")
	c1.EmitString(RESP_MODEL, "gpt-4o")
	asm.Push(c1)

	// Chunk 2: text delta
	c2 := NewProgram()
	c2.EmitString(STREAM_DELTA, "Hello")
	asm.Push(c2)

	// Chunk 3: more text
	c3 := NewProgram()
	c3.EmitString(STREAM_DELTA, " world!")
	asm.Push(c3)

	// Chunk 4: done
	c4 := NewProgram()
	c4.EmitString(RESP_DONE, "stop")
	c4.Emit(STREAM_END)
	asm.Push(c4)

	if !asm.Done() {
		t.Fatal("expected assembler to be done")
	}

	prog := asm.Program()
	t.Logf("Assembled:\n%s", prog.Disasm())

	// Verify assembled content
	found := false
	for _, inst := range prog.Code {
		if inst.Op == TXT_CHUNK && inst.Str == "Hello world!" {
			found = true
		}
	}
	if !found {
		t.Fatal("expected assembled text 'Hello world!' not found")
	}
}

func TestStreamAssemblerToolCalls(t *testing.T) {
	asm := NewStreamAssembler()

	// Stream start
	c1 := NewProgram()
	c1.Emit(STREAM_START)
	asm.Push(c1)

	// Tool call start (name + id)
	c2 := NewProgram()
	td1, _ := json.Marshal(map[string]any{"index": 0, "id": "call_1", "name": "get_weather"})
	c2.EmitJSON(STREAM_TOOL_DELTA, td1)
	asm.Push(c2)

	// Tool call args fragment 1
	c3 := NewProgram()
	td2, _ := json.Marshal(map[string]any{"index": 0, "arguments": `{"loc`})
	c3.EmitJSON(STREAM_TOOL_DELTA, td2)
	asm.Push(c3)

	// Tool call args fragment 2
	c4 := NewProgram()
	td3, _ := json.Marshal(map[string]any{"index": 0, "arguments": `ation":"NYC"}`})
	c4.EmitJSON(STREAM_TOOL_DELTA, td3)
	asm.Push(c4)

	// Done
	c5 := NewProgram()
	c5.EmitString(RESP_DONE, "tool_calls")
	c5.Emit(STREAM_END)
	asm.Push(c5)

	prog := asm.Program()
	t.Logf("Assembled:\n%s", prog.Disasm())

	// Verify tool call reassembly
	foundArgs := false
	for _, inst := range prog.Code {
		if inst.Op == CALL_ARGS {
			if string(inst.JSON) == `{"location":"NYC"}` {
				foundArgs = true
			}
		}
	}
	if !foundArgs {
		t.Fatal("expected reassembled tool call args not found")
	}
}

func TestBinaryInvalidMagic(t *testing.T) {
	data := bytes.NewReader([]byte("NOPE\x01"))
	_, err := Decode(data)
	if err == nil {
		t.Fatal("expected error for invalid magic bytes")
	}
}

func TestBinaryInvalidVersion(t *testing.T) {
	data := bytes.NewReader([]byte{'A', 'I', 'L', 0x00, 0xFF, 0, 0, 0, 0})
	_, err := Decode(data)
	if err == nil {
		t.Fatal("expected error for unsupported version")
	}
}
