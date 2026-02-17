package ail

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// ─── Stateful Stream Converter ───────────────────────────────────────────────

// StreamConverter converts streaming chunks from one provider format to another
// in real-time. It handles structural mismatches between providers:
//
//   - Text deltas are converted and forwarded immediately (1:1).
//   - Tool call deltas are forwarded immediately for targets supporting
//     incremental tool streaming (OpenAI, Anthropic), or buffered until
//     complete for targets that require whole function calls (Google GenAI).
//   - Response metadata (ID, model) is carried across chunks and injected
//     into every emitted chunk, since some formats (OpenAI) require it on
//     every event while others (Anthropic) send it only once.
//   - One source event may produce multiple output events (e.g., an OpenAI
//     finish chunk becomes Anthropic's message_delta + message_stop).
//
// Usage in an HTTP streaming proxy:
//
//	conv, _ := ail.NewStreamConverter(ail.StyleAnthropic, ail.StyleChatCompletions)
//	for _, chunk := range upstreamChunks {
//	    outputs, err := conv.Push(chunk)
//	    if err != nil { /* handle */ }
//	    for _, out := range outputs {
//	        fmt.Fprintf(w, "data: %s\n\n", out)
//	        flusher.Flush()
//	    }
//	}
//	// Flush any remaining buffered tool calls
//	final, _ := conv.Flush()
//	for _, out := range final {
//	    fmt.Fprintf(w, "data: %s\n\n", out)
//	    flusher.Flush()
//	}
type StreamConverter struct {
	parser      StreamChunkParser
	emitter     StreamChunkEmitter
	sourceStyle Style
	targetStyle Style

	mu        sync.Mutex
	respID    string
	respModel string

	// Tool call buffering for targets needing complete function calls.
	bufferTools  bool
	pendingTools map[int]*pendingToolCall
	toolOrder    []int
}

// pendingToolCall accumulates tool-call fragments for buffered emission.
type pendingToolCall struct {
	ID   string
	Name string
	Args strings.Builder
}

// NewStreamConverter creates a converter for real-time streaming translation
// from one provider format to another. The converter is safe for concurrent use.
func NewStreamConverter(from, to Style) (*StreamConverter, error) {
	parser, err := GetStreamChunkParser(from)
	if err != nil {
		return nil, fmt.Errorf("ail: stream converter source: %w", err)
	}
	emitter, err := GetStreamChunkEmitter(to)
	if err != nil {
		return nil, fmt.Errorf("ail: stream converter target: %w", err)
	}

	// Google GenAI needs complete function calls in one chunk,
	// so buffer tool deltas until the call is ready.
	bufferTools := (to == StyleGoogleGenAI)

	return &StreamConverter{
		parser:       parser,
		emitter:      emitter,
		sourceStyle:  from,
		targetStyle:  to,
		bufferTools:  bufferTools,
		pendingTools: make(map[int]*pendingToolCall),
	}, nil
}

// Push processes a source streaming chunk and returns zero or more converted
// output chunks. Each returned []byte is a complete JSON object suitable for
// writing as an SSE "data:" line.
//
// Returns zero chunks when the source event is purely structural or buffered.
// Returns multiple chunks when a source event expands into several target events.
func (c *StreamConverter) Push(sourceChunk []byte) ([][]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	parsed, err := c.parser.ParseStreamChunk(sourceChunk)
	if err != nil {
		return nil, fmt.Errorf("ail: stream convert parse: %w", err)
	}

	return c.pushProgramLocked(parsed)
}

// PushProgram processes an already-parsed AIL program through the converter.
// Use this when the program has been obtained separately (e.g., from a driver
// that already parsed the upstream chunk, or after plugin modification).
//
// Like Push, it returns zero or more converted output chunks.
func (c *StreamConverter) PushProgram(prog *Program) ([][]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.pushProgramLocked(prog)
}

// pushProgramLocked is the shared core of Push and PushProgram.
// Caller must hold c.mu.
func (c *StreamConverter) pushProgramLocked(parsed *Program) ([][]byte, error) {
	if parsed == nil || parsed.Len() == 0 {
		return nil, nil
	}

	// Remember metadata for injection into future chunks.
	c.trackMetadata(parsed)

	// Split into emittable sub-programs, handling buffering and
	// multi-event targets.
	units := c.processInstructions(parsed)

	var outputs [][]byte
	for _, unit := range units {
		c.injectMetadata(unit)
		out, err := c.emitter.EmitStreamChunk(unit)
		if err != nil {
			return outputs, fmt.Errorf("ail: stream convert emit: %w", err)
		}
		if out != nil {
			outputs = append(outputs, out)
		}
	}

	return outputs, nil
}

// Flush forces emission of any buffered data (e.g., pending tool call
// fragments). Call this when the upstream stream ends to ensure all data
// is delivered. It is safe to call Flush multiple times.
func (c *StreamConverter) Flush() ([][]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	toolProg := c.drainPendingTools()
	if toolProg == nil {
		return nil, nil
	}

	c.injectMetadata(toolProg)
	out, err := c.emitter.EmitStreamChunk(toolProg)
	if err != nil {
		return nil, fmt.Errorf("ail: stream convert flush: %w", err)
	}
	if out == nil {
		return nil, nil
	}
	return [][]byte{out}, nil
}

// ─── internal helpers ────────────────────────────────────────────────────────

// trackMetadata remembers RESP_ID and RESP_MODEL for later injection.
func (c *StreamConverter) trackMetadata(prog *Program) {
	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_ID:
			c.respID = inst.Str
		case RESP_MODEL:
			c.respModel = inst.Str
		}
	}
}

// injectMetadata prepends RESP_ID and RESP_MODEL to a program if they are
// missing but have been seen in a previous chunk.
func (c *StreamConverter) injectMetadata(prog *Program) {
	hasID, hasModel := false, false
	for _, inst := range prog.Code {
		if inst.Op == RESP_ID {
			hasID = true
		}
		if inst.Op == RESP_MODEL {
			hasModel = true
		}
	}

	var prepend []Instruction
	if !hasID && c.respID != "" {
		prepend = append(prepend, Instruction{Op: RESP_ID, Str: c.respID})
	}
	if !hasModel && c.respModel != "" {
		prepend = append(prepend, Instruction{Op: RESP_MODEL, Str: c.respModel})
	}
	if len(prepend) > 0 {
		prog.Code = append(prepend, prog.Code...)
	}
}

// processInstructions splits a parsed program into emittable sub-programs.
// The strategy depends on the target format:
//   - Anthropic targets: each event-producing opcode becomes its own program
//     (because Anthropic SSE uses a different JSON structure per event type).
//   - Google targets with tool buffering: STREAM_TOOL_DELTA is accumulated.
//   - Default: the whole program is emitted as one chunk.
func (c *StreamConverter) processInstructions(prog *Program) []*Program {
	if c.targetNeedsSplitting() {
		return c.splitForTarget(prog)
	}

	if c.bufferTools {
		return c.processWithBuffering(prog)
	}

	// Default: forward entire program as one unit.
	return []*Program{prog}
}

// targetNeedsSplitting reports whether the target format requires each
// event-producing opcode to be emitted as a separate SSE event.
func (c *StreamConverter) targetNeedsSplitting() bool {
	return c.targetStyle == StyleAnthropic
}

// splitForTarget splits a program so each event-producing opcode gets its
// own sub-program. Metadata is attached to the first event (or the
// STREAM_START event if present). USAGE is grouped with RESP_DONE since
// Anthropic's message_delta carries both stop_reason and usage.
func (c *StreamConverter) splitForTarget(prog *Program) []*Program {
	var meta []Instruction
	var events [][]Instruction

	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_ID, RESP_MODEL:
			meta = append(meta, inst)
		case STREAM_START, STREAM_DELTA, STREAM_TOOL_DELTA, RESP_DONE, STREAM_END:
			events = append(events, []Instruction{inst})
		case USAGE:
			// Attach usage to the preceding RESP_DONE if exists.
			attached := false
			for i := len(events) - 1; i >= 0; i-- {
				if events[i][0].Op == RESP_DONE {
					events[i] = append(events[i], inst)
					attached = true
					break
				}
			}
			if !attached {
				// No RESP_DONE yet, carry as metadata.
				meta = append(meta, inst)
			}
		}
	}

	if len(events) == 0 {
		if len(meta) > 0 {
			p := NewProgram()
			p.Code = meta
			return []*Program{p}
		}
		return nil
	}

	// Attach metadata to STREAM_START event, or first event if no STREAM_START.
	targetIdx := 0
	for i, ev := range events {
		if ev[0].Op == STREAM_START {
			targetIdx = i
			break
		}
	}
	events[targetIdx] = append(meta, events[targetIdx]...)

	var result []*Program
	for _, ev := range events {
		p := NewProgram()
		p.Code = ev
		result = append(result, p)
	}
	return result
}

// processWithBuffering handles tool call buffering for targets that require
// complete function calls (e.g., Google GenAI). Non-tool instructions are
// forwarded immediately; tool deltas are buffered and flushed when a flush
// trigger (RESP_DONE, STREAM_END) is encountered.
func (c *StreamConverter) processWithBuffering(prog *Program) []*Program {
	var results []*Program
	current := NewProgram()

	for _, inst := range prog.Code {
		switch inst.Op {
		case STREAM_TOOL_DELTA:
			c.bufferToolDelta(inst.JSON)

		case RESP_DONE, STREAM_END:
			// Emit accumulated non-tool content.
			if len(current.Code) > 0 {
				results = append(results, current)
				current = NewProgram()
			}
			// Flush buffered tool calls before the terminal event.
			if toolProg := c.drainPendingTools(); toolProg != nil {
				results = append(results, toolProg)
			}
			// Emit the terminal instruction.
			p := NewProgram()
			p.Code = append(p.Code, inst)
			results = append(results, p)

		default:
			current.Code = append(current.Code, inst)
		}
	}

	if len(current.Code) > 0 {
		results = append(results, current)
	}

	return results
}

// bufferToolDelta accumulates a STREAM_TOOL_DELTA fragment by tool index.
func (c *StreamConverter) bufferToolDelta(j json.RawMessage) {
	var td struct {
		Index     int    `json:"index"`
		ID        string `json:"id,omitempty"`
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	}
	if json.Unmarshal(j, &td) != nil {
		return
	}

	tc, ok := c.pendingTools[td.Index]
	if !ok {
		tc = &pendingToolCall{}
		c.pendingTools[td.Index] = tc
		c.toolOrder = append(c.toolOrder, td.Index)
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

// drainPendingTools converts all buffered tool call fragments into a single
// program with complete STREAM_TOOL_DELTA instructions, then clears the buffer.
// Returns nil if no tools are pending.
func (c *StreamConverter) drainPendingTools() *Program {
	if len(c.toolOrder) == 0 {
		return nil
	}

	prog := NewProgram()
	for _, idx := range c.toolOrder {
		tc := c.pendingTools[idx]
		td := map[string]any{"index": idx}
		if tc.ID != "" {
			td["id"] = tc.ID
		}
		if tc.Name != "" {
			td["name"] = tc.Name
		}
		if args := tc.Args.String(); args != "" {
			td["arguments"] = args
		}
		j, _ := json.Marshal(td)
		prog.EmitJSON(STREAM_TOOL_DELTA, j)
	}

	c.pendingTools = make(map[int]*pendingToolCall)
	c.toolOrder = nil

	return prog
}

// ConvertStreamChunk is a stateless convenience for simple cases where
// cross-chunk state isn't needed (e.g., text-only streams, same-format
// passthrough). For streams with tool calls or metadata that spans chunks,
// use StreamConverter instead.
func ConvertStreamChunk(chunk []byte, from, to Style) ([]byte, error) {
	parser, err := GetStreamChunkParser(from)
	if err != nil {
		return nil, err
	}
	emitter, err := GetStreamChunkEmitter(to)
	if err != nil {
		return nil, err
	}
	prog, err := parser.ParseStreamChunk(chunk)
	if err != nil {
		return nil, err
	}
	return emitter.EmitStreamChunk(prog)
}
