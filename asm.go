package ail

import (
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
}

// opcodes that take a ref:N argument.
var refArgOps = map[Opcode]bool{
	IMG_REF: true, AUD_REF: true, TXT_REF: true,
}

// Asm parses a human-readable assembly listing (as produced by Disasm) back
// into an AIL Program. Lines are separated by newlines; leading whitespace
// (indentation) is ignored. Comment lines starting with ";" are silently
// skipped (real-asm style).
//
// This is the inverse of Program.Disasm().
func Asm(text string) (*Program, error) {
	prog := NewProgram()
	lines := strings.Split(text, "\n")

	for lineNo, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}

		// Comment lines: skip anything starting with ";"
		if strings.HasPrefix(line, ";") {
			continue
		}

		// Split "OPCODE rest..."
		opName, rest := splitFirst(line)
		op, ok := nameToOpcode[opName]
		if !ok {
			return nil, fmt.Errorf("line %d: unknown opcode %q", lineNo+1, opName)
		}

		switch {
		case stringArgOps[op]:
			prog.EmitString(op, rest)

		case floatArgOps[op]:
			f, err := strconv.ParseFloat(strings.TrimSpace(rest), 64)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid float %q: %w", lineNo+1, rest, err)
			}
			prog.EmitFloat(op, f)

		case intArgOps[op]:
			i, err := strconv.ParseInt(strings.TrimSpace(rest), 10, 32)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid int %q: %w", lineNo+1, rest, err)
			}
			prog.EmitInt(op, int32(i))

		case jsonArgOps[op]:
			j := strings.TrimSpace(rest)
			if !json.Valid([]byte(j)) {
				return nil, fmt.Errorf("line %d: invalid JSON for %s: %s", lineNo+1, opName, j)
			}
			prog.EmitJSON(op, json.RawMessage(j))

		case refArgOps[op]:
			ref, err := parseRef(rest, lineNo)
			if err != nil {
				return nil, err
			}
			prog.EmitRef(op, ref)

		case op == SET_META:
			key, val := splitFirst(rest)
			if key == "" {
				return nil, fmt.Errorf("line %d: SET_META requires key and value", lineNo+1)
			}
			prog.EmitKeyVal(op, key, val)

		case op == EXT_DATA:
			key, j := splitFirst(rest)
			if key == "" || j == "" {
				return nil, fmt.Errorf("line %d: EXT_DATA requires key and JSON", lineNo+1)
			}
			if !json.Valid([]byte(j)) {
				return nil, fmt.Errorf("line %d: EXT_DATA invalid JSON: %s", lineNo+1, j)
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
