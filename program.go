package ail

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// ─── Context helpers ─────────────────────────────────────────────────────────

type programContextKey struct{}

// ContextWithProgram stores an AIL program in a context.
// Used by recursive handler plugins to pass a program to re-invocations without re-serializing.
func ContextWithProgram(ctx context.Context, prog *Program) context.Context {
	return context.WithValue(ctx, programContextKey{}, prog)
}

// ProgramFromContext retrieves an AIL program from context, if present.
func ProgramFromContext(ctx context.Context) (*Program, bool) {
	prog, ok := ctx.Value(programContextKey{}).(*Program)
	return prog, ok
}

// Instruction is a single AIL instruction with its opcode and typed argument.
type Instruction struct {
	Op   Opcode
	Str  string          // used by TXT_CHUNK, DEF_NAME, SET_MODEL, CALL_START, etc.
	Num  float64         // used by SET_TEMP, SET_TOPP
	Int  int32           // used by SET_MAX
	JSON json.RawMessage // used by DEF_SCHEMA, CALL_ARGS, USAGE, EXT_DATA, STREAM_TOOL_DELTA
	Key  string          // used by SET_META, EXT_DATA (the key part)
	Ref  uint32          // used by IMG_REF, AUD_REF, TXT_REF
}

// Program is an ordered list of instructions plus a side-buffer for large blobs.
type Program struct {
	Code    []Instruction
	Buffers [][]byte // side-buffer for IMG_REF, AUD_REF, TXT_REF payloads
}

// NewProgram creates an empty program.
func NewProgram() *Program {
	return &Program{
		Code:    make([]Instruction, 0, 32),
		Buffers: make([][]byte, 0),
	}
}

// ─── Emit helpers ────────────────────────────────────────────────────────────

// Emit appends a bare opcode with no arguments.
func (p *Program) Emit(op Opcode) {
	p.Code = append(p.Code, Instruction{Op: op})
}

// EmitString appends an opcode with a string argument.
func (p *Program) EmitString(op Opcode, s string) {
	p.Code = append(p.Code, Instruction{Op: op, Str: s})
}

// EmitFloat appends an opcode with a float64 argument.
func (p *Program) EmitFloat(op Opcode, f float64) {
	p.Code = append(p.Code, Instruction{Op: op, Num: f})
}

// EmitInt appends an opcode with an int32 argument.
func (p *Program) EmitInt(op Opcode, i int32) {
	p.Code = append(p.Code, Instruction{Op: op, Int: i})
}

// EmitJSON appends an opcode with a raw JSON argument.
func (p *Program) EmitJSON(op Opcode, j json.RawMessage) {
	p.Code = append(p.Code, Instruction{Op: op, JSON: j})
}

// EmitKeyVal appends an opcode with key + string-value arguments (SET_META).
func (p *Program) EmitKeyVal(op Opcode, key, val string) {
	p.Code = append(p.Code, Instruction{Op: op, Key: key, Str: val})
}

// EmitKeyJSON appends an opcode with key + JSON-value arguments (EXT_DATA).
func (p *Program) EmitKeyJSON(op Opcode, key string, j json.RawMessage) {
	p.Code = append(p.Code, Instruction{Op: op, Key: key, JSON: j})
}

// EmitRef appends an opcode with a buffer reference.
func (p *Program) EmitRef(op Opcode, ref uint32) {
	p.Code = append(p.Code, Instruction{Op: op, Ref: ref})
}

// AddBuffer appends data to the side-buffer and returns its index.
func (p *Program) AddBuffer(data []byte) uint32 {
	idx := uint32(len(p.Buffers))
	p.Buffers = append(p.Buffers, data)
	return idx
}

// ─── Append / Clone ──────────────────────────────────────────────────────────

// Append creates a new program by concatenating this program's code with other's.
// Buffers from other are re-indexed. All instructions are deep-copied.
func (p *Program) Append(other *Program) *Program {
	result := NewProgram()
	for _, inst := range p.Code {
		result.Code = append(result.Code, cloneInstruction(inst))
	}
	for _, b := range p.Buffers {
		buf := make([]byte, len(b))
		copy(buf, b)
		result.Buffers = append(result.Buffers, buf)
	}

	bufOffset := uint32(len(p.Buffers))
	for _, inst := range other.Code {
		clone := cloneInstruction(inst)
		switch inst.Op {
		case IMG_REF, AUD_REF, TXT_REF:
			clone.Ref += bufOffset
		}
		result.Code = append(result.Code, clone)
	}
	for _, b := range other.Buffers {
		buf := make([]byte, len(b))
		copy(buf, b)
		result.Buffers = append(result.Buffers, buf)
	}
	return result
}

// cloneInstruction returns a deep copy of an instruction, including its
// json.RawMessage field (which is a []byte and would otherwise share the
// underlying array with the source).
func cloneInstruction(inst Instruction) Instruction {
	if len(inst.JSON) > 0 {
		j := make(json.RawMessage, len(inst.JSON))
		copy(j, inst.JSON)
		inst.JSON = j
	}
	return inst
}

// Clone creates a deep copy of the program.
func (p *Program) Clone() *Program {
	result := NewProgram()
	result.Code = make([]Instruction, len(p.Code))
	for i, inst := range p.Code {
		result.Code[i] = cloneInstruction(inst)
	}
	result.Buffers = make([][]byte, len(p.Buffers))
	for i, b := range p.Buffers {
		buf := make([]byte, len(b))
		copy(buf, b)
		result.Buffers[i] = buf
	}
	return result
}

// Len returns the number of instructions.
func (p *Program) Len() int { return len(p.Code) }

// ─── Query helpers ───────────────────────────────────────────────────────────

// GetModel scans config instructions and returns the model name, or "".
func (p *Program) GetModel() string {
	for _, inst := range p.Code {
		if inst.Op == SET_MODEL {
			return inst.Str
		}
	}
	return ""
}

// IsStreaming returns true if SET_STREAM is present.
func (p *Program) IsStreaming() bool {
	for _, inst := range p.Code {
		if inst.Op == SET_STREAM {
			return true
		}
	}
	return false
}

// SetModel replaces the first SET_MODEL instruction or appends one.
func (p *Program) SetModel(model string) {
	for i, inst := range p.Code {
		if inst.Op == SET_MODEL {
			p.Code[i].Str = model
			return
		}
	}
	// Prepend so it's at the top of config section
	p.Code = append([]Instruction{{Op: SET_MODEL, Str: model}}, p.Code...)
}

// ─── Disassembly (human-readable) ────────────────────────────────────────────

// Disasm returns a human-readable assembly listing of the program.
func (p *Program) Disasm() string {
	var sb strings.Builder
	indent := 0
	for _, inst := range p.Code {
		// Decrease indent before END opcodes
		switch inst.Op {
		case MSG_END, DEF_END, CALL_END, RESULT_END, STREAM_END:
			indent--
			if indent < 0 {
				indent = 0
			}
		}

		for range indent {
			sb.WriteString("  ")
		}

		sb.WriteString(inst.Op.Name())

		switch inst.Op {
		case TXT_CHUNK, DEF_NAME, DEF_DESC, CALL_START, CALL_NAME,
			RESULT_START, RESULT_DATA, RESP_ID, RESP_MODEL, RESP_DONE,
			SET_MODEL, SET_STOP, STREAM_DELTA:
			sb.WriteString(" ")
			sb.WriteString(fmt.Sprintf("%q", inst.Str))

		case SET_TEMP, SET_TOPP:
			sb.WriteString(fmt.Sprintf(" %.4f", inst.Num))

		case SET_MAX:
			sb.WriteString(fmt.Sprintf(" %d", inst.Int))

		case IMG_REF, AUD_REF, TXT_REF:
			sb.WriteString(fmt.Sprintf(" ref:%d", inst.Ref))

		case DEF_SCHEMA, CALL_ARGS, USAGE, STREAM_TOOL_DELTA:
			sb.WriteString(" ")
			sb.Write(inst.JSON)

		case SET_META:
			sb.WriteString(fmt.Sprintf(" %q %q", inst.Key, inst.Str))

		case EXT_DATA:
			sb.WriteString(fmt.Sprintf(" %q ", inst.Key))
			sb.Write(inst.JSON)
		}

		sb.WriteByte('\n')

		// Increase indent after START opcodes
		switch inst.Op {
		case MSG_START, DEF_START, CALL_START, RESULT_START, STREAM_START:
			indent++
		}
	}
	return sb.String()
}
