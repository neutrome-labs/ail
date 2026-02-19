package ail

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// nameToOpcode is the reverse lookup of opcodeNames.
var nameToOpcode map[string]Opcode

func init() {
	nameToOpcode = make(map[string]Opcode, len(opcodeNames))
	for op, name := range opcodeNames {
		nameToOpcode[name] = op
	}
}

// opcodes that take a plain string argument (rest of line after opcode).
var stringArgOps = map[Opcode]bool{
	TXT_CHUNK: true, DEF_NAME: true, DEF_DESC: true,
	CALL_START: true, CALL_NAME: true,
	RESULT_START: true, RESULT_DATA: true,
	RESP_ID: true, RESP_MODEL: true, RESP_DONE: true,
	SET_MODEL: true, SET_STOP: true, STREAM_DELTA: true,
	THINK_CHUNK: true, STREAM_THINK_DELTA: true,
}

// opcodes that take a float64 argument.
var floatArgOps = map[Opcode]bool{
	SET_TEMP: true, SET_TOPP: true,
}

// opcodes that take an int32 argument.
var intArgOps = map[Opcode]bool{
	SET_MAX: true,
}

// opcodes that take a raw JSON argument.
var jsonArgOps = map[Opcode]bool{
	DEF_SCHEMA: true, CALL_ARGS: true, USAGE: true, STREAM_TOOL_DELTA: true,
	SET_THINK: true,
}

// opcodes that take a ref:N argument.
var refArgOps = map[Opcode]bool{
	IMG_REF: true, AUD_REF: true, TXT_REF: true, THINK_REF: true,
}

// Asm parses a human-readable assembly listing (as produced by Disasm) back
// into an AIL Program. Lines are separated by newlines; leading whitespace
// (indentation) is ignored. Comment lines starting with ";" are silently
// skipped (real-asm style).
//
// Multiline string and JSON values can be encoded using a heredoc block:
//
//	TXT_CHUNK <<<
//	line one
//	line two
//	>>>
//
// The block starts with the opcode followed by <<<, and ends with >>> on its
// own line (leading/trailing whitespace on the >>> line is ignored). The body
// lines are taken verbatim — indentation is NOT stripped — so the value
// preserves exactly the bytes between the <<< and >>> markers.
//
// This is the inverse of Program.Disasm().
func Asm(text string) (*Program, error) {
	prog := NewProgram()
	lines := strings.Split(text, "\n")

	// collectHeredoc collects lines after index i until a line whose trimmed
	// content is ">>>" and returns the joined body and the index of the ">>>"
	// line (so the caller can advance i to that position).
	collectHeredoc := func(i int) (string, int, error) {
		var parts []string
		for j := i + 1; j < len(lines); j++ {
			if strings.TrimSpace(lines[j]) == ">>>" {
				return strings.Join(parts, "\n"), j, nil
			}
			parts = append(parts, lines[j])
		}
		return "", i, fmt.Errorf("line %d: heredoc block started with <<< but never closed with >>>", i+1)
	}

	// compactJSON compacts raw JSON bytes, returning them unchanged on error
	// (the validity check that follows will surface the real error).
	compactJSON := func(raw string) string {
		var buf bytes.Buffer
		if err := json.Compact(&buf, []byte(raw)); err != nil {
			return raw
		}
		return buf.String()
	}

	for i := 0; i < len(lines); i++ {
		raw := lines[i]
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}

		// Comment lines: skip anything starting with ";"
		if strings.HasPrefix(line, ";") {
			continue
		}

		// Buffer declaration: ".ref N <base64>"
		// Produced by Disasm() for IMG_REF / AUD_REF / TXT_REF payloads.
		if strings.HasPrefix(line, ".ref ") {
			parts := strings.SplitN(line[5:], " ", 2)
			if len(parts) != 2 {
				return nil, fmt.Errorf("line %d: .ref requires index and base64 data", i+1)
			}
			idx, err := strconv.ParseUint(strings.TrimSpace(parts[0]), 10, 32)
			if err != nil {
				return nil, fmt.Errorf("line %d: .ref invalid index %q: %w", i+1, parts[0], err)
			}
			data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(parts[1]))
			if err != nil {
				return nil, fmt.Errorf("line %d: .ref invalid base64: %w", i+1, err)
			}
			// Grow Buffers slice to fit idx.
			for uint32(len(prog.Buffers)) <= uint32(idx) {
				prog.Buffers = append(prog.Buffers, nil)
			}
			prog.Buffers[idx] = data
			continue
		}

		// Split "OPCODE rest..."
		opName, rest := splitFirst(line)
		op, ok := nameToOpcode[opName]
		if !ok {
			return nil, fmt.Errorf("line %d: unknown opcode %q", i+1, opName)
		}

		switch {
		case stringArgOps[op]:
			val := rest
			if strings.TrimSpace(rest) == "<<<" {
				var err error
				val, i, err = collectHeredoc(i)
				if err != nil {
					return nil, err
				}
			}
			prog.EmitString(op, val)

		case floatArgOps[op]:
			f, err := strconv.ParseFloat(strings.TrimSpace(rest), 64)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid float %q: %w", i+1, rest, err)
			}
			prog.EmitFloat(op, f)

		case intArgOps[op]:
			n, err := strconv.ParseInt(strings.TrimSpace(rest), 10, 32)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid int %q: %w", i+1, rest, err)
			}
			prog.EmitInt(op, int32(n))

		case jsonArgOps[op]:
			j := strings.TrimSpace(rest)
			if j == "<<<" {
				var err error
				j, i, err = collectHeredoc(i)
				if err != nil {
					return nil, err
				}
				j = strings.TrimSpace(j)
			}
			j = compactJSON(j)
			if !json.Valid([]byte(j)) {
				return nil, fmt.Errorf("line %d: invalid JSON for %s: %s", i+1, opName, j)
			}
			prog.EmitJSON(op, json.RawMessage(j))

		case refArgOps[op]:
			ref, err := parseRef(rest, i)
			if err != nil {
				return nil, err
			}
			prog.EmitRef(op, ref)

		case op == SET_META:
			key, val := splitFirst(rest)
			if key == "" {
				return nil, fmt.Errorf("line %d: SET_META requires key and value", i+1)
			}
			prog.EmitKeyVal(op, key, val)

		case op == EXT_DATA:
			key, j := splitFirst(rest)
			if key == "" {
				return nil, fmt.Errorf("line %d: EXT_DATA requires key and JSON", i+1)
			}
			if strings.TrimSpace(j) == "<<<" {
				var err error
				j, i, err = collectHeredoc(i)
				if err != nil {
					return nil, err
				}
				j = strings.TrimSpace(j)
			}
			if j == "" {
				return nil, fmt.Errorf("line %d: EXT_DATA requires key and JSON", i+1)
			}
			j = compactJSON(j)
			if !json.Valid([]byte(j)) {
				return nil, fmt.Errorf("line %d: EXT_DATA invalid JSON: %s", i+1, j)
			}
			prog.EmitKeyJSON(op, key, json.RawMessage(j))

		default:
			// No-arg opcodes: MSG_START, MSG_END, ROLE_*, SET_STREAM, DEF_START, DEF_END, etc.
			prog.Emit(op)
		}
	}

	return prog, nil
}

// splitFirst splits a string on the first whitespace boundary.
// Returns (first_word, rest). rest may be empty.
func splitFirst(s string) (string, string) {
	idx := strings.IndexByte(s, ' ')
	if idx < 0 {
		return s, ""
	}
	return s[:idx], s[idx+1:]
}

// parseRef parses "ref:N" and returns N as uint32.
func parseRef(rest string, lineNo int) (uint32, error) {
	rest = strings.TrimSpace(rest)
	if !strings.HasPrefix(rest, "ref:") {
		return 0, fmt.Errorf("line %d: expected ref:N, got %q", lineNo+1, rest)
	}
	n, err := strconv.ParseUint(rest[4:], 10, 32)
	if err != nil {
		return 0, fmt.Errorf("line %d: invalid ref number %q: %w", lineNo+1, rest[4:], err)
	}
	return uint32(n), nil
}
