// Package ail implements the AI Intermediate Language — a stack-based bytecode
// for representing AI provider interactions in a provider-agnostic way.
//
// The IL decouples parsing (ingesting provider-specific JSON into opcodes) from
// emitting (writing opcodes back out as provider-specific JSON), enabling
// any-to-any conversion between OpenAI, Anthropic, Google, etc.
package ail

// Opcode is a single-byte instruction identifier.
type Opcode byte

// ─── Structure (0x10-0x1F) ────────────────────────────────────────────────────
const (
	MSG_START Opcode = 0x10 // Begin message block
	MSG_END   Opcode = 0x11 // End message block
	ROLE_SYS  Opcode = 0x12 // role = system
	ROLE_USR  Opcode = 0x13 // role = user
	ROLE_AST  Opcode = 0x14 // role = assistant
	ROLE_TOOL Opcode = 0x15 // role = tool / function-result
)

// ─── Content (0x20-0x2F) ─────────────────────────────────────────────────────
const (
	TXT_CHUNK Opcode = 0x20 // arg: String — text content
	IMG_REF   Opcode = 0x21 // arg: RefID — image buffer reference
	AUD_REF   Opcode = 0x22 // arg: RefID — audio buffer reference
	TXT_REF   Opcode = 0x23 // arg: RefID — large text buffer reference
)

// ─── Reasoning / Thinking (0x28-0x2B) ────────────────────────────────────────
const (
	THINK_START Opcode = 0x28 // Begin thinking/reasoning block within a message
	THINK_CHUNK Opcode = 0x29 // arg: String — reasoning text content
	THINK_END   Opcode = 0x2A // End thinking/reasoning block
	THINK_REF   Opcode = 0x2B // arg: RefID — opaque reasoning blob (e.g., Gemini thoughtSignature)
)

// ─── Tool Definition (0x30-0x3F) ─────────────────────────────────────────────
const (
	DEF_START  Opcode = 0x30 // Begin tool definitions
	DEF_NAME   Opcode = 0x31 // arg: String — function name
	DEF_DESC   Opcode = 0x32 // arg: String — description
	DEF_SCHEMA Opcode = 0x33 // arg: JSON — parameter schema
	DEF_END    Opcode = 0x34 // End tool definitions
)

// ─── Tool Call (0x40-0x4F) ───────────────────────────────────────────────────
const (
	CALL_START Opcode = 0x40 // arg: String — call ID
	CALL_NAME  Opcode = 0x41 // arg: String — function name
	CALL_ARGS  Opcode = 0x42 // arg: JSON — arguments
	CALL_END   Opcode = 0x43 // End tool call
)

// ─── Tool Result (0x48-0x4A) ────────────────────────────────────────────────
const (
	RESULT_START Opcode = 0x48 // arg: String — call ID
	RESULT_DATA  Opcode = 0x49 // arg: String — result content
	RESULT_END   Opcode = 0x4A // End tool result
)

// ─── Response Metadata (0x50-0x5F) ───────────────────────────────────────────
const (
	RESP_ID    Opcode = 0x50 // arg: String — response ID
	RESP_MODEL Opcode = 0x51 // arg: String — model that generated response
	RESP_DONE  Opcode = 0x52 // arg: String — finish reason
	USAGE      Opcode = 0x53 // arg: JSON — usage statistics
)

// ─── Stream Events (0x60-0x6F) ───────────────────────────────────────────────
const (
	STREAM_START       Opcode = 0x60 // Begin streaming response
	STREAM_DELTA       Opcode = 0x61 // arg: String — text delta
	STREAM_TOOL_DELTA  Opcode = 0x62 // arg: JSON — tool call delta
	STREAM_END         Opcode = 0x63 // End streaming response
	STREAM_THINK_DELTA Opcode = 0x64 // arg: String — thinking/reasoning text delta
)

// ─── Configuration (0xF0-0xFF) ───────────────────────────────────────────────
const (
	SET_MODEL  Opcode = 0xF0 // arg: String
	SET_TEMP   Opcode = 0xF1 // arg: Float
	SET_TOPP   Opcode = 0xF2 // arg: Float
	SET_STOP   Opcode = 0xF3 // arg: String
	SET_MAX    Opcode = 0xF4 // arg: Int
	SET_STREAM Opcode = 0xF5 // no arg — presence means streaming
	SET_THINK  Opcode = 0xF6 // arg: JSON — thinking/reasoning configuration
	EXT_DATA   Opcode = 0xFE // arg: Key, JSON — provider-specific extension
	SET_META   Opcode = 0xFF // arg: Key, Val
)

// opcodeNames maps opcodes to their human-readable mnemonic (for Disasm).
var opcodeNames = map[Opcode]string{
	MSG_START: "MSG_START", MSG_END: "MSG_END",
	ROLE_SYS: "ROLE_SYS", ROLE_USR: "ROLE_USR", ROLE_AST: "ROLE_AST", ROLE_TOOL: "ROLE_TOOL",
	TXT_CHUNK: "TXT_CHUNK", IMG_REF: "IMG_REF", AUD_REF: "AUD_REF", TXT_REF: "TXT_REF",
	THINK_START: "THINK_START", THINK_CHUNK: "THINK_CHUNK", THINK_END: "THINK_END", THINK_REF: "THINK_REF",
	DEF_START: "DEF_START", DEF_NAME: "DEF_NAME", DEF_DESC: "DEF_DESC", DEF_SCHEMA: "DEF_SCHEMA", DEF_END: "DEF_END",
	CALL_START: "CALL_START", CALL_NAME: "CALL_NAME", CALL_ARGS: "CALL_ARGS", CALL_END: "CALL_END",
	RESULT_START: "RESULT_START", RESULT_DATA: "RESULT_DATA", RESULT_END: "RESULT_END",
	RESP_ID: "RESP_ID", RESP_MODEL: "RESP_MODEL", RESP_DONE: "RESP_DONE", USAGE: "USAGE",
	STREAM_START: "STREAM_START", STREAM_DELTA: "STREAM_DELTA", STREAM_TOOL_DELTA: "STREAM_TOOL_DELTA", STREAM_END: "STREAM_END",
	STREAM_THINK_DELTA: "STREAM_THINK_DELTA",
	SET_MODEL:          "SET_MODEL", SET_TEMP: "SET_TEMP", SET_TOPP: "SET_TOPP", SET_STOP: "SET_STOP",
	SET_MAX: "SET_MAX", SET_STREAM: "SET_STREAM", SET_THINK: "SET_THINK", EXT_DATA: "EXT_DATA", SET_META: "SET_META",
}

// Name returns the human-readable mnemonic for an opcode.
func (o Opcode) Name() string {
	if n, ok := opcodeNames[o]; ok {
		return n
	}
	return "UNKNOWN"
}

func (o Opcode) String() string { return o.Name() }
