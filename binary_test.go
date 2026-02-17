package ail

import (
	"bytes"
	"encoding/json"
	"testing"
)

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
