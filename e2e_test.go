package ail

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// e2eCase maps a subfolder under fixtures/ to a style and operation kind.
type e2eCase struct {
	dir   string // relative to fixtures/
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
	const root = "fixtures"

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

	// Normalize: strip null-valued keys since "key": null and absent key
	// are semantically equivalent in API round-trips.
	want = stripNulls(want)
	got = stripNulls(got)

	if !reflect.DeepEqual(want, got) {
		diffs := jsonDiff("$", want, got)
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("JSON mismatch (%d difference(s)):\n", len(diffs)))
		for _, d := range diffs {
			sb.WriteString(fmt.Sprintf("  %s\n", d))
		}
		/*// Also include full pretty-printed output for context.
		wantPretty, _ := json.MarshalIndent(want, "", "  ")
		gotPretty, _ := json.MarshalIndent(got, "", "  ")
		sb.WriteString(fmt.Sprintf("\n─── expected ───\n%s\n─── actual ───\n%s",
			wantPretty, gotPretty))*/
		t.Error(sb.String())
	}
}

// jsonDiff recursively compares two decoded JSON values and returns
// human-readable descriptions of each difference, annotated with JSON paths.
func jsonDiff(path string, want, got any) []string {
	if reflect.DeepEqual(want, got) {
		return nil
	}

	var diffs []string

	switch w := want.(type) {
	case map[string]any:
		g, ok := got.(map[string]any)
		if !ok {
			return []string{fmt.Sprintf("%s: type mismatch: expected object, got %T", path, got)}
		}
		// Keys in expected but missing or different in actual.
		for k, wv := range w {
			gv, exists := g[k]
			childPath := path + "." + k
			if !exists {
				pretty, _ := json.Marshal(wv)
				diffs = append(diffs, fmt.Sprintf("%s: missing from actual (expected %s)", childPath, pretty))
			} else {
				diffs = append(diffs, jsonDiff(childPath, wv, gv)...)
			}
		}
		// Keys in actual but not in expected.
		for k, gv := range g {
			if _, exists := w[k]; !exists {
				pretty, _ := json.Marshal(gv)
				diffs = append(diffs, fmt.Sprintf("%s.%s: unexpected key in actual (value %s)", path, k, pretty))
			}
		}

	case []any:
		g, ok := got.([]any)
		if !ok {
			return []string{fmt.Sprintf("%s: type mismatch: expected array, got %T", path, got)}
		}
		if len(w) != len(g) {
			diffs = append(diffs, fmt.Sprintf("%s: array length mismatch: expected %d, got %d", path, len(w), len(g)))
		}
		limit := len(w)
		if len(g) < limit {
			limit = len(g)
		}
		for i := 0; i < limit; i++ {
			diffs = append(diffs, jsonDiff(fmt.Sprintf("%s[%d]", path, i), w[i], g[i])...)
		}
		// Show extra elements in actual.
		for i := len(w); i < len(g); i++ {
			pretty, _ := json.Marshal(g[i])
			diffs = append(diffs, fmt.Sprintf("%s[%d]: unexpected extra element in actual: %s", path, i, pretty))
		}
		// Note missing elements.
		for i := len(g); i < len(w); i++ {
			pretty, _ := json.Marshal(w[i])
			diffs = append(diffs, fmt.Sprintf("%s[%d]: missing from actual (expected %s)", path, i, pretty))
		}

	default:
		// Scalar comparison (string, float64, bool, nil).
		wPretty, _ := json.Marshal(want)
		gPretty, _ := json.Marshal(got)
		diffs = append(diffs, fmt.Sprintf("%s: expected %s, got %s", path, wPretty, gPretty))
	}

	return diffs
}

// stripNulls recursively removes null-valued keys from maps so that
// {"content": null} and {} compare as equal (semantically equivalent in APIs).
func stripNulls(v any) any {
	switch val := v.(type) {
	case map[string]any:
		cleaned := make(map[string]any, len(val))
		for k, child := range val {
			if child == nil {
				continue
			}
			cleaned[k] = stripNulls(child)
		}
		return cleaned
	case []any:
		cleaned := make([]any, len(val))
		for i, child := range val {
			cleaned[i] = stripNulls(child)
		}
		return cleaned
	default:
		return v
	}
}
