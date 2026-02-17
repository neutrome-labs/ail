package ail

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// e2eCase maps a subfolder under e2e_tests/ to a style and operation kind.
type e2eCase struct {
	dir   string // relative to e2e_tests/
	style Style
	kind  string // "request", "response", or "stream"
}

var e2eCases = []e2eCase{
	// OpenAI Chat Completions
	{"chat/request", StyleChatCompletions, "request"},
	{"chat/response", StyleChatCompletions, "response"},
	{"chat/stream", StyleChatCompletions, "stream"},

	// OpenAI Responses API (request-only, no response/stream emitter)
	{"responses/request", StyleResponses, "request"},

	// Anthropic Messages
	{"anthropic/request", StyleAnthropic, "request"},
	{"anthropic/response", StyleAnthropic, "response"},
	{"anthropic/stream", StyleAnthropic, "stream"},

	// Google GenAI
	{"genai/request", StyleGoogleGenAI, "request"},
	{"genai/response", StyleGoogleGenAI, "response"},
	{"genai/stream", StyleGoogleGenAI, "stream"},
}

func TestE2ERoundTrip(t *testing.T) {
	const root = "e2e_tests"

	for _, tc := range e2eCases {
		dir := filepath.Join(root, tc.dir)
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			continue // skip dirs that don't exist yet
		}

		files, err := filepath.Glob(filepath.Join(dir, "*.json"))
		if err != nil {
			t.Fatalf("glob %s: %v", dir, err)
		}

		for _, file := range files {
			name := filepath.Base(file)
			t.Run(tc.dir+"/"+name, func(t *testing.T) {
				input, err := os.ReadFile(file)
				if err != nil {
					t.Fatalf("read %s: %v", file, err)
				}

				output, err := roundTrip(input, tc.style, tc.kind)
				if err != nil {
					t.Fatalf("roundtrip: %v", err)
				}

				assertJSONEqual(t, input, output)
			})
		}
	}
}

// roundTrip parses JSON with the appropriate parser, then emits it back.
func roundTrip(input []byte, style Style, kind string) ([]byte, error) {
	switch kind {
	case "request":
		parser, err := GetParser(style)
		if err != nil {
			return nil, err
		}
		prog, err := parser.ParseRequest(input)
		if err != nil {
			return nil, err
		}
		emitter, err := GetEmitter(style)
		if err != nil {
			return nil, err
		}
		return emitter.EmitRequest(prog)

	case "response":
		parser, err := GetResponseParser(style)
		if err != nil {
			return nil, err
		}
		prog, err := parser.ParseResponse(input)
		if err != nil {
			return nil, err
		}
		emitter, err := GetResponseEmitter(style)
		if err != nil {
			return nil, err
		}
		return emitter.EmitResponse(prog)

	case "stream":
		parser, err := GetStreamChunkParser(style)
		if err != nil {
			return nil, err
		}
		prog, err := parser.ParseStreamChunk(input)
		if err != nil {
			return nil, err
		}
		emitter, err := GetStreamChunkEmitter(style)
		if err != nil {
			return nil, err
		}
		return emitter.EmitStreamChunk(prog)

	default:
		return nil, nil
	}
}

// assertJSONEqual compares two JSON blobs by value (key order independent).
func assertJSONEqual(t *testing.T, expected, actual []byte) {
	t.Helper()

	var want, got any
	if err := json.Unmarshal(expected, &want); err != nil {
		t.Fatalf("unmarshal expected: %v", err)
	}
	if err := json.Unmarshal(actual, &got); err != nil {
		t.Fatalf("unmarshal actual: %v", err)
	}

	if !reflect.DeepEqual(want, got) {
		wantPretty, _ := json.MarshalIndent(want, "", "  ")
		gotPretty, _ := json.MarshalIndent(got, "", "  ")
		t.Errorf("JSON mismatch\n─── expected ───\n%s\n─── actual ───\n%s",
			wantPretty, gotPretty)
	}
}
