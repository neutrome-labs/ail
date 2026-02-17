package ail

import "strings"

// ─── Span types ──────────────────────────────────────────────────────────────
// Spans represent contiguous instruction ranges delimited by START/END opcodes.
// All indices are into Program.Code and are inclusive on both ends.

// MessageSpan locates a MSG_START..MSG_END block and its role.
type MessageSpan struct {
	Start int    // index of MSG_START
	End   int    // index of MSG_END
	Role  Opcode // ROLE_SYS, ROLE_USR, ROLE_AST, or ROLE_TOOL
}

// ToolDefSpan locates a DEF_START..DEF_END block for a single tool definition.
type ToolDefSpan struct {
	Start int    // index of DEF_START
	End   int    // index of DEF_END
	Name  string // DEF_NAME value within the span
}

// ToolCallSpan locates a CALL_START..CALL_END block.
type ToolCallSpan struct {
	Start  int    // index of CALL_START
	End    int    // index of CALL_END
	CallID string // CALL_START string arg (the call ID)
	Name   string // CALL_NAME value within the span
}

// ToolResultSpan locates a RESULT_START..RESULT_END block.
type ToolResultSpan struct {
	Start  int    // index of RESULT_START
	End    int    // index of RESULT_END
	CallID string // RESULT_START string arg (the call ID)
}

// ─── Traversal ───────────────────────────────────────────────────────────────

// Messages returns all message spans in instruction order.
func (p *Program) Messages() []MessageSpan {
	var spans []MessageSpan
	for i := 0; i < len(p.Code); i++ {
		if p.Code[i].Op != MSG_START {
			continue
		}
		span := MessageSpan{Start: i}
		// Scan for the role and the matching MSG_END.
		for j := i + 1; j < len(p.Code); j++ {
			switch p.Code[j].Op {
			case ROLE_SYS, ROLE_USR, ROLE_AST, ROLE_TOOL:
				if span.Role == 0 {
					span.Role = p.Code[j].Op
				}
			case MSG_END:
				span.End = j
				spans = append(spans, span)
				i = j // advance outer loop past this block
				goto next
			}
		}
	next:
	}
	return spans
}

// ToolDefs returns all tool definition spans.
func (p *Program) ToolDefs() []ToolDefSpan {
	var spans []ToolDefSpan
	for i := 0; i < len(p.Code); i++ {
		if p.Code[i].Op != DEF_START {
			continue
		}
		span := ToolDefSpan{Start: i}
		for j := i + 1; j < len(p.Code); j++ {
			switch p.Code[j].Op {
			case DEF_NAME:
				if span.Name == "" {
					span.Name = p.Code[j].Str
				}
			case DEF_END:
				span.End = j
				spans = append(spans, span)
				i = j
				goto nextDef
			}
		}
	nextDef:
	}
	return spans
}

// ToolCalls returns all tool call spans.
func (p *Program) ToolCalls() []ToolCallSpan {
	var spans []ToolCallSpan
	for i := 0; i < len(p.Code); i++ {
		if p.Code[i].Op != CALL_START {
			continue
		}
		span := ToolCallSpan{Start: i, CallID: p.Code[i].Str}
		for j := i + 1; j < len(p.Code); j++ {
			switch p.Code[j].Op {
			case CALL_NAME:
				if span.Name == "" {
					span.Name = p.Code[j].Str
				}
			case CALL_END:
				span.End = j
				spans = append(spans, span)
				i = j
				goto nextCall
			}
		}
	nextCall:
	}
	return spans
}

// ToolResults returns all tool result spans.
func (p *Program) ToolResults() []ToolResultSpan {
	var spans []ToolResultSpan
	for i := 0; i < len(p.Code); i++ {
		if p.Code[i].Op != RESULT_START {
			continue
		}
		span := ToolResultSpan{Start: i, CallID: p.Code[i].Str}
		for j := i + 1; j < len(p.Code); j++ {
			if p.Code[j].Op == RESULT_END {
				span.End = j
				spans = append(spans, span)
				i = j
				break
			}
		}
	}
	return spans
}

// ─── Content extraction ──────────────────────────────────────────────────────

// MessageText returns the concatenated TXT_CHUNK content within a message span.
func (p *Program) MessageText(span MessageSpan) string {
	var sb strings.Builder
	for i := span.Start; i <= span.End && i < len(p.Code); i++ {
		if p.Code[i].Op == TXT_CHUNK {
			sb.WriteString(p.Code[i].Str)
		}
	}
	return sb.String()
}

// SystemPrompt returns the concatenated text of all leading system messages,
// joined by "\n\n". Returns "" if no system messages exist.
func (p *Program) SystemPrompt() string {
	var parts []string
	for _, m := range p.Messages() {
		if m.Role != ROLE_SYS {
			break // only collect leading system messages
		}
		if t := p.MessageText(m); t != "" {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, "\n\n")
}

// SystemPrompts returns the spans of all consecutive system messages at the
// start of the message sequence. Stops at the first non-system message.
func (p *Program) SystemPrompts() []MessageSpan {
	var out []MessageSpan
	for _, m := range p.Messages() {
		if m.Role != ROLE_SYS {
			break
		}
		out = append(out, m)
	}
	return out
}

// LastUserMessage returns the span of the last user message.
// Returns false if no user message exists.
func (p *Program) LastUserMessage() (MessageSpan, bool) {
	msgs := p.Messages()
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == ROLE_USR {
			return msgs[i], true
		}
	}
	return MessageSpan{}, false
}

// ─── Search ──────────────────────────────────────────────────────────────────

// FindAll returns the indices of every instruction whose opcode matches op.
func (p *Program) FindAll(op Opcode) []int {
	var indices []int
	for i, inst := range p.Code {
		if inst.Op == op {
			indices = append(indices, i)
		}
	}
	return indices
}

// HasOpcode returns true if any instruction matches the given opcode.
func (p *Program) HasOpcode(op Opcode) bool {
	for _, inst := range p.Code {
		if inst.Op == op {
			return true
		}
	}
	return false
}

// WalkFunc is called for each instruction by Walk.
// Return false to stop iteration.
type WalkFunc func(index int, inst Instruction) bool

// Walk iterates over all instructions in order, calling fn for each.
// Stops early if fn returns false.
func (p *Program) Walk(fn WalkFunc) {
	for i, inst := range p.Code {
		if !fn(i, inst) {
			return
		}
	}
}

// MessagesByRole returns only messages that match the given role opcode.
func (p *Program) MessagesByRole(role Opcode) []MessageSpan {
	var out []MessageSpan
	for _, m := range p.Messages() {
		if m.Role == role {
			out = append(out, m)
		}
	}
	return out
}

// ─── Slicing & mutation ──────────────────────────────────────────────────────

// Slice returns a new program containing instructions [start, end] (inclusive)
// with a deep copy. Buffer references are preserved as-is; callers should
// ensure the original program's Buffers remain accessible or use SliceWithBuffers.
func (p *Program) Slice(start, end int) *Program {
	if start < 0 {
		start = 0
	}
	if end >= len(p.Code) {
		end = len(p.Code) - 1
	}
	result := NewProgram()
	for i := start; i <= end; i++ {
		result.Code = append(result.Code, cloneInstruction(p.Code[i]))
	}
	// Share buffers so refs remain valid.
	result.Buffers = p.Buffers
	return result
}

// ExtractMessage returns a minimal program containing just the instructions
// from the given message span (MSG_START through MSG_END inclusive).
func (p *Program) ExtractMessage(span MessageSpan) *Program {
	return p.Slice(span.Start, span.End)
}

// RemoveRange returns a new program with instructions [start, end] (inclusive) removed.
// Instructions outside the range are deep-copied; buffers are shared.
func (p *Program) RemoveRange(start, end int) *Program {
	result := NewProgram()
	for i, inst := range p.Code {
		if i >= start && i <= end {
			continue
		}
		result.Code = append(result.Code, cloneInstruction(inst))
	}
	result.Buffers = p.Buffers
	return result
}

// RemoveMessages returns a new program with the specified message spans removed.
// Spans should come from Messages() and be non-overlapping.
func (p *Program) RemoveMessages(spans ...MessageSpan) *Program {
	remove := make(map[int]bool)
	for _, s := range spans {
		for i := s.Start; i <= s.End; i++ {
			remove[i] = true
		}
	}
	result := NewProgram()
	for i, inst := range p.Code {
		if remove[i] {
			continue
		}
		result.Code = append(result.Code, cloneInstruction(inst))
	}
	result.Buffers = p.Buffers
	return result
}

// ReplaceRange returns a new program where instructions [start, end] (inclusive)
// are replaced with the given instructions. Original instructions outside the
// range are deep-copied.
func (p *Program) ReplaceRange(start, end int, instructions ...Instruction) *Program {
	result := NewProgram()
	for i, inst := range p.Code {
		if i == start {
			for _, repl := range instructions {
				result.Code = append(result.Code, cloneInstruction(repl))
			}
		}
		if i >= start && i <= end {
			continue
		}
		result.Code = append(result.Code, cloneInstruction(inst))
	}
	result.Buffers = p.Buffers
	return result
}

// InsertBefore returns a new program with additional instructions inserted
// immediately before the given index. The original is not modified.
func (p *Program) InsertBefore(index int, instructions ...Instruction) *Program {
	result := NewProgram()
	for i, inst := range p.Code {
		if i == index {
			for _, ins := range instructions {
				result.Code = append(result.Code, cloneInstruction(ins))
			}
		}
		result.Code = append(result.Code, cloneInstruction(inst))
	}
	result.Buffers = p.Buffers
	return result
}

// InsertAfter returns a new program with additional instructions inserted
// immediately after the given index. The original is not modified.
func (p *Program) InsertAfter(index int, instructions ...Instruction) *Program {
	result := NewProgram()
	for i, inst := range p.Code {
		result.Code = append(result.Code, cloneInstruction(inst))
		if i == index {
			for _, ins := range instructions {
				result.Code = append(result.Code, cloneInstruction(ins))
			}
		}
	}
	result.Buffers = p.Buffers
	return result
}

// ─── High-level convenience ──────────────────────────────────────────────────

// TruncateMessages returns a new program that keeps config/defs but only the
// last n messages. Useful for context-window management.
func (p *Program) TruncateMessages(n int) *Program {
	msgs := p.Messages()
	if n >= len(msgs) {
		return p.Clone()
	}

	// Keep messages from msgs[len(msgs)-n:]
	keep := msgs[len(msgs)-n:]
	keepSet := make(map[int]bool)
	for _, m := range keep {
		for i := m.Start; i <= m.End; i++ {
			keepSet[i] = true
		}
	}

	result := NewProgram()
	// First pass: copy non-message instructions (config, defs, etc.)
	// and only the kept message instructions.
	msgRange := make(map[int]bool)
	for _, m := range msgs {
		for i := m.Start; i <= m.End; i++ {
			msgRange[i] = true
		}
	}

	for i, inst := range p.Code {
		if msgRange[i] && !keepSet[i] {
			continue // in a message range but not a kept message
		}
		result.Code = append(result.Code, cloneInstruction(inst))
	}
	result.Buffers = p.Buffers
	return result
}

// PrependSystemPrompt inserts a new system message before all existing
// messages (including other system messages). This allows stacking multiple
// system prompts that providers can merge with "\n\n" when needed.
func (p *Program) PrependSystemPrompt(text string) *Program {
	sysInstructions := []Instruction{
		{Op: MSG_START},
		{Op: ROLE_SYS},
		{Op: TXT_CHUNK, Str: text},
		{Op: MSG_END},
	}

	msgs := p.Messages()
	if len(msgs) > 0 {
		return p.InsertBefore(msgs[0].Start, sysInstructions...)
	}

	// No messages at all — append to the end.
	result := p.Clone()
	result.Code = append(result.Code, sysInstructions...)
	return result
}

// ReplaceSystemPrompt replaces all leading system messages with a single
// system message containing the given text.
func (p *Program) ReplaceSystemPrompt(text string) *Program {
	sysMsgs := p.SystemPrompts()
	if len(sysMsgs) == 0 {
		return p.PrependSystemPrompt(text)
	}

	newSys := []Instruction{
		{Op: MSG_START},
		{Op: ROLE_SYS},
		{Op: TXT_CHUNK, Str: text},
		{Op: MSG_END},
	}

	// Remove all leading system messages and insert the replacement
	start := sysMsgs[0].Start
	end := sysMsgs[len(sysMsgs)-1].End
	return p.ReplaceRange(start, end, newSys...)
}

// AppendUserMessage appends a new user message at the end of the instruction
// stream. Returns a new program; the original is not modified.
func (p *Program) AppendUserMessage(text string) *Program {
	result := p.Clone()
	result.Emit(MSG_START)
	result.Emit(ROLE_USR)
	result.EmitString(TXT_CHUNK, text)
	result.Emit(MSG_END)
	return result
}

// CountMessages returns the total number of messages.
func (p *Program) CountMessages() int {
	return len(p.Messages())
}

// Config returns all SET_META key-value pairs as a map.
func (p *Program) Config() map[string]string {
	m := make(map[string]string)
	for _, inst := range p.Code {
		if inst.Op == SET_META {
			m[inst.Key] = inst.Str
		}
	}
	return m
}
