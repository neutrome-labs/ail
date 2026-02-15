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

// opcodes that take a single quoted-string argument.
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
// (indentation) is ignored. Comment lines starting with "; " are preserved
// as COMMENT instructions.
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

		// Comment lines: "; <text>"
		if strings.HasPrefix(line, "; ") {
			prog.EmitComment(line[2:])
			continue
		}
		if line == ";" {
			prog.EmitComment("")
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
			s, err := parseQuotedString(rest, lineNo)
			if err != nil {
				return nil, err
			}
			prog.EmitString(op, s)

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
			key, val, err := parseTwoQuotedStrings(rest, lineNo)
			if err != nil {
				return nil, err
			}
			prog.EmitKeyVal(op, key, val)

		case op == EXT_DATA:
			key, j, err := parseQuotedStringThenJSON(rest, lineNo)
			if err != nil {
				return nil, err
			}
			prog.EmitKeyJSON(op, key, j)

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

// parseQuotedString extracts a Go-style quoted string from the rest of a line.
func parseQuotedString(rest string, lineNo int) (string, error) {
	rest = strings.TrimSpace(rest)
	if rest == "" {
		return "", fmt.Errorf("line %d: expected quoted string argument", lineNo+1)
	}
	s, err := strconv.Unquote(rest)
	if err != nil {
		return "", fmt.Errorf("line %d: invalid quoted string %s: %w", lineNo+1, rest, err)
	}
	return s, nil
}

// parseTwoQuotedStrings parses 'key "val"' from the rest portion.
func parseTwoQuotedStrings(rest string, lineNo int) (string, string, error) {
	rest = strings.TrimSpace(rest)
	// Find the end of the first quoted string
	key, remaining, err := extractQuotedString(rest, lineNo)
	if err != nil {
		return "", "", fmt.Errorf("line %d: SET_META key: %w", lineNo+1, err)
	}
	val, err := parseQuotedString(remaining, lineNo)
	if err != nil {
		return "", "", fmt.Errorf("line %d: SET_META val: %w", lineNo+1, err)
	}
	return key, val, nil
}

// parseQuotedStringThenJSON parses '"key" {json...}' from the rest portion.
func parseQuotedStringThenJSON(rest string, lineNo int) (string, json.RawMessage, error) {
	rest = strings.TrimSpace(rest)
	key, remaining, err := extractQuotedString(rest, lineNo)
	if err != nil {
		return "", nil, fmt.Errorf("line %d: EXT_DATA key: %w", lineNo+1, err)
	}
	j := strings.TrimSpace(remaining)
	if !json.Valid([]byte(j)) {
		return "", nil, fmt.Errorf("line %d: EXT_DATA invalid JSON: %s", lineNo+1, j)
	}
	return key, json.RawMessage(j), nil
}

// extractQuotedString reads the first Go-quoted string from s and returns
// (unquoted value, remaining text after the closing quote).
func extractQuotedString(s string, lineNo int) (string, string, error) {
	if len(s) == 0 || s[0] != '"' {
		return "", "", fmt.Errorf("line %d: expected '\"' at start of %q", lineNo+1, s)
	}
	// Walk through the string respecting escapes
	i := 1
	for i < len(s) {
		if s[i] == '\\' {
			i += 2
			continue
		}
		if s[i] == '"' {
			// Found the closing quote
			val, err := strconv.Unquote(s[:i+1])
			if err != nil {
				return "", "", fmt.Errorf("line %d: invalid quoted string %s: %w", lineNo+1, s[:i+1], err)
			}
			return val, strings.TrimSpace(s[i+1:]), nil
		}
		i++
	}
	return "", "", fmt.Errorf("line %d: unterminated quoted string", lineNo+1)
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
