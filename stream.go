package ail

import (
	"encoding/json"
	"strings"
	"sync"
)

// ─── Stateful Stream Assembly ────────────────────────────────────────────────

// StreamAssembler accumulates parsed streaming chunks (already in AIL form)
// into a complete response Program.  It tracks state across chunks to properly
// reassemble fragmented payloads such as tool-call argument fragments.
//
// Usage:
//
//	asm := ail.NewStreamAssembler()
//	for chunk := range chunks {
//	    prog, _ := parser.ParseStreamChunk(chunk)
//	    asm.Push(prog)
//	}
//	full := asm.Program()  // complete response as an AIL program
type StreamAssembler struct {
	mu           sync.Mutex
	respID       string
	respModel    string
	text         strings.Builder
	toolCalls    map[int]*streamToolCall // index → accumulated tool call
	toolOrder    []int                   // insertion order of tool indices
	usage        json.RawMessage
	finishReason string
	started      bool
	done         bool
}

type streamToolCall struct {
	ID   string
	Name string
	Args strings.Builder
}

// NewStreamAssembler creates a StreamAssembler ready to accept chunks.
func NewStreamAssembler() *StreamAssembler {
	return &StreamAssembler{
		toolCalls: make(map[int]*streamToolCall),
	}
}

// Push processes all instructions from a parsed stream-chunk program,
// accumulating state across calls.
func (a *StreamAssembler) Push(chunk *Program) {
	a.mu.Lock()
	defer a.mu.Unlock()

	for _, inst := range chunk.Code {
		switch inst.Op {
		case RESP_ID:
			a.respID = inst.Str
		case RESP_MODEL:
			a.respModel = inst.Str
		case STREAM_START:
			a.started = true
		case STREAM_DELTA:
			a.text.WriteString(inst.Str)
		case STREAM_TOOL_DELTA:
			a.pushToolDelta(inst.JSON)
		case USAGE:
			a.usage = inst.JSON
		case RESP_DONE:
			a.finishReason = inst.Str
		case STREAM_END:
			a.done = true
		}
	}
}

// pushToolDelta parses a STREAM_TOOL_DELTA JSON payload and accumulates
// the tool-call fragments (id, name, argument chunks) by index.
func (a *StreamAssembler) pushToolDelta(j json.RawMessage) {
	var td struct {
		Index     int    `json:"index"`
		ID        string `json:"id,omitempty"`
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	}
	if json.Unmarshal(j, &td) != nil {
		return
	}

	tc, ok := a.toolCalls[td.Index]
	if !ok {
		tc = &streamToolCall{}
		a.toolCalls[td.Index] = tc
		a.toolOrder = append(a.toolOrder, td.Index)
	}
	if td.ID != "" {
		tc.ID = td.ID
	}
	if td.Name != "" {
		tc.Name = td.Name
	}
	if td.Arguments != "" {
		tc.Args.WriteString(td.Arguments)
	}
}

// Done reports whether a STREAM_END instruction has been received.
func (a *StreamAssembler) Done() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.done
}

// Program builds a complete response AIL program from the accumulated state.
// It can be called at any time (even before Done) to get a snapshot.
func (a *StreamAssembler) Program() *Program {
	a.mu.Lock()
	defer a.mu.Unlock()

	prog := NewProgram()

	if a.respID != "" {
		prog.EmitString(RESP_ID, a.respID)
	}
	if a.respModel != "" {
		prog.EmitString(RESP_MODEL, a.respModel)
	}

	prog.Emit(MSG_START)
	prog.Emit(ROLE_AST)

	if txt := a.text.String(); txt != "" {
		prog.EmitString(TXT_CHUNK, txt)
	}

	for _, idx := range a.toolOrder {
		tc := a.toolCalls[idx]
		prog.EmitString(CALL_START, tc.ID)
		prog.EmitString(CALL_NAME, tc.Name)
		if args := tc.Args.String(); args != "" {
			prog.EmitJSON(CALL_ARGS, json.RawMessage(args))
		}
		prog.Emit(CALL_END)
	}

	if a.finishReason != "" {
		prog.EmitString(RESP_DONE, a.finishReason)
	}

	prog.Emit(MSG_END)

	if len(a.usage) > 0 {
		prog.EmitJSON(USAGE, a.usage)
	}

	return prog
}

// Reset clears all accumulated state so the assembler can be reused.
func (a *StreamAssembler) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.respID = ""
	a.respModel = ""
	a.text.Reset()
	a.toolCalls = make(map[int]*streamToolCall)
	a.toolOrder = nil
	a.usage = nil
	a.finishReason = ""
	a.started = false
	a.done = false
}
