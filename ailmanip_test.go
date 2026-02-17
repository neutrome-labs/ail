package ail

import (
	"encoding/json"
	"testing"
)

// ─── helpers ─────────────────────────────────────────────────────────────────

// buildConversation produces a realistic multi-message program for tests.
func buildConversation() *Program {
	p := NewProgram()
	p.EmitString(SET_MODEL, "gpt-4o")
	p.EmitFloat(SET_TEMP, 0.7)
	p.EmitKeyVal(SET_META, "env", "test")

	// System
	p.Emit(MSG_START)
	p.Emit(ROLE_SYS)
	p.EmitString(TXT_CHUNK, "You are a helpful assistant.")
	p.Emit(MSG_END)

	// User 1
	p.Emit(MSG_START)
	p.Emit(ROLE_USR)
	p.EmitString(TXT_CHUNK, "What is 2+2?")
	p.Emit(MSG_END)

	// Assistant 1
	p.Emit(MSG_START)
	p.Emit(ROLE_AST)
	p.EmitString(TXT_CHUNK, "4")
	p.Emit(MSG_END)

	// User 2
	p.Emit(MSG_START)
	p.Emit(ROLE_USR)
	p.EmitString(TXT_CHUNK, "Thanks!")
	p.Emit(MSG_END)

	return p
}

// ─── Messages traversal ─────────────────────────────────────────────────────

func TestMessages(t *testing.T) {
	p := buildConversation()
	msgs := p.Messages()

	if len(msgs) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(msgs))
	}

	expectRoles := []Opcode{ROLE_SYS, ROLE_USR, ROLE_AST, ROLE_USR}
	for i, m := range msgs {
		if m.Role != expectRoles[i] {
			t.Fatalf("message %d: expected role %s, got %s", i, expectRoles[i].Name(), m.Role.Name())
		}
		if m.Start > m.End {
			t.Fatalf("message %d: start %d > end %d", i, m.Start, m.End)
		}
		if p.Code[m.Start].Op != MSG_START {
			t.Fatalf("message %d: start instruction is not MSG_START", i)
		}
		if p.Code[m.End].Op != MSG_END {
			t.Fatalf("message %d: end instruction is not MSG_END", i)
		}
	}
}

func TestMessagesEmpty(t *testing.T) {
	p := NewProgram()
	p.EmitString(SET_MODEL, "gpt-4o")
	msgs := p.Messages()
	if len(msgs) != 0 {
		t.Fatalf("expected 0 messages, got %d", len(msgs))
	}
}

// ─── MessageText ─────────────────────────────────────────────────────────────

func TestMessageText(t *testing.T) {
	p := buildConversation()
	msgs := p.Messages()

	if txt := p.MessageText(msgs[0]); txt != "You are a helpful assistant." {
		t.Fatalf("sys text = %q", txt)
	}
	if txt := p.MessageText(msgs[1]); txt != "What is 2+2?" {
		t.Fatalf("user1 text = %q", txt)
	}
	if txt := p.MessageText(msgs[2]); txt != "4" {
		t.Fatalf("assistant text = %q", txt)
	}
}

func TestMessageTextMultiChunk(t *testing.T) {
	p := NewProgram()
	p.Emit(MSG_START)
	p.Emit(ROLE_USR)
	p.EmitString(TXT_CHUNK, "Hello ")
	p.EmitString(TXT_CHUNK, "World")
	p.Emit(MSG_END)

	msgs := p.Messages()
	if txt := p.MessageText(msgs[0]); txt != "Hello World" {
		t.Fatalf("multi-chunk text = %q", txt)
	}
}

// ─── SystemPrompt & LastUserMessage ──────────────────────────────────────────

func TestSystemPrompt(t *testing.T) {
	p := buildConversation()
	if sp := p.SystemPrompt(); sp != "You are a helpful assistant." {
		t.Fatalf("system prompt = %q", sp)
	}
}

func TestSystemPromptMissing(t *testing.T) {
	p := NewProgram()
	p.Emit(MSG_START)
	p.Emit(ROLE_USR)
	p.EmitString(TXT_CHUNK, "Hi")
	p.Emit(MSG_END)

	if sp := p.SystemPrompt(); sp != "" {
		t.Fatalf("expected empty system prompt, got %q", sp)
	}
}

func TestLastUserMessage(t *testing.T) {
	p := buildConversation()
	span, ok := p.LastUserMessage()
	if !ok {
		t.Fatal("expected to find last user message")
	}
	if p.MessageText(span) != "Thanks!" {
		t.Fatalf("last user msg = %q", p.MessageText(span))
	}
}

func TestLastUserMessageMissing(t *testing.T) {
	p := NewProgram()
	p.Emit(MSG_START)
	p.Emit(ROLE_SYS)
	p.EmitString(TXT_CHUNK, "sys")
	p.Emit(MSG_END)

	_, ok := p.LastUserMessage()
	if ok {
		t.Fatal("expected no user message")
	}
}

// ─── MessagesByRole ──────────────────────────────────────────────────────────

func TestMessagesByRole(t *testing.T) {
	p := buildConversation()
	userMsgs := p.MessagesByRole(ROLE_USR)
	if len(userMsgs) != 2 {
		t.Fatalf("expected 2 user messages, got %d", len(userMsgs))
	}
	astMsgs := p.MessagesByRole(ROLE_AST)
	if len(astMsgs) != 1 {
		t.Fatalf("expected 1 assistant message, got %d", len(astMsgs))
	}
}

// ─── FindAll & HasOpcode ─────────────────────────────────────────────────────

func TestFindAll(t *testing.T) {
	p := buildConversation()
	starts := p.FindAll(MSG_START)
	if len(starts) != 4 {
		t.Fatalf("expected 4 MSG_START, got %d", len(starts))
	}
	for _, idx := range starts {
		if p.Code[idx].Op != MSG_START {
			t.Fatal("FindAll returned wrong index")
		}
	}
}

func TestHasOpcode(t *testing.T) {
	p := buildConversation()
	if !p.HasOpcode(SET_MODEL) {
		t.Fatal("should have SET_MODEL")
	}
	if p.HasOpcode(DEF_START) {
		t.Fatal("should not have DEF_START")
	}
}

// ─── Walk ────────────────────────────────────────────────────────────────────

func TestWalk(t *testing.T) {
	p := buildConversation()
	count := 0
	p.Walk(func(_ int, _ Instruction) bool {
		count++
		return true
	})
	if count != p.Len() {
		t.Fatalf("walk visited %d, expected %d", count, p.Len())
	}
}

func TestWalkEarlyStop(t *testing.T) {
	p := buildConversation()
	count := 0
	p.Walk(func(_ int, _ Instruction) bool {
		count++
		return count < 3
	})
	if count != 3 {
		t.Fatalf("walk should stop at 3, got %d", count)
	}
}

// ─── CountMessages ───────────────────────────────────────────────────────────

func TestCountMessages(t *testing.T) {
	p := buildConversation()
	if n := p.CountMessages(); n != 4 {
		t.Fatalf("expected 4, got %d", n)
	}
}

// ─── Config ──────────────────────────────────────────────────────────────────

func TestConfig(t *testing.T) {
	p := buildConversation()
	cfg := p.Config()
	if cfg["env"] != "test" {
		t.Fatalf("expected env=test, got %q", cfg["env"])
	}
}

// ─── ToolDefs / ToolCalls / ToolResults traversal ────────────────────────────

func buildToolProgram() *Program {
	p := NewProgram()
	p.EmitString(SET_MODEL, "gpt-4o")

	// Tool definitions
	p.Emit(DEF_START)
	p.EmitString(DEF_NAME, "get_weather")
	p.EmitString(DEF_DESC, "Get current weather")
	p.EmitJSON(DEF_SCHEMA, json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`))
	p.Emit(DEF_END)

	p.Emit(DEF_START)
	p.EmitString(DEF_NAME, "search")
	p.EmitString(DEF_DESC, "Search the web")
	p.EmitJSON(DEF_SCHEMA, json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`))
	p.Emit(DEF_END)

	// User message
	p.Emit(MSG_START)
	p.Emit(ROLE_USR)
	p.EmitString(TXT_CHUNK, "What's the weather in Paris?")
	p.Emit(MSG_END)

	// Assistant with tool call
	p.Emit(MSG_START)
	p.Emit(ROLE_AST)
	p.EmitString(CALL_START, "call_abc123")
	p.EmitString(CALL_NAME, "get_weather")
	p.EmitJSON(CALL_ARGS, json.RawMessage(`{"city":"Paris"}`))
	p.Emit(CALL_END)
	p.Emit(MSG_END)

	// Tool result
	p.Emit(MSG_START)
	p.Emit(ROLE_TOOL)
	p.EmitString(RESULT_START, "call_abc123")
	p.EmitString(RESULT_DATA, `{"temp":22,"unit":"C"}`)
	p.Emit(RESULT_END)
	p.Emit(MSG_END)

	return p
}

func TestToolDefs(t *testing.T) {
	p := buildToolProgram()
	defs := p.ToolDefs()
	if len(defs) != 2 {
		t.Fatalf("expected 2 tool defs, got %d", len(defs))
	}
	if defs[0].Name != "get_weather" {
		t.Fatalf("def 0 name = %q", defs[0].Name)
	}
	if defs[1].Name != "search" {
		t.Fatalf("def 1 name = %q", defs[1].Name)
	}
	for _, d := range defs {
		if p.Code[d.Start].Op != DEF_START || p.Code[d.End].Op != DEF_END {
			t.Fatal("def span boundaries wrong")
		}
	}
}

func TestToolCalls(t *testing.T) {
	p := buildToolProgram()
	calls := p.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].CallID != "call_abc123" {
		t.Fatalf("call id = %q", calls[0].CallID)
	}
	if calls[0].Name != "get_weather" {
		t.Fatalf("call name = %q", calls[0].Name)
	}
}

func TestToolResults(t *testing.T) {
	p := buildToolProgram()
	results := p.ToolResults()
	if len(results) != 1 {
		t.Fatalf("expected 1 tool result, got %d", len(results))
	}
	if results[0].CallID != "call_abc123" {
		t.Fatalf("result call id = %q", results[0].CallID)
	}
}

// ─── Slice ───────────────────────────────────────────────────────────────────

func TestSlice(t *testing.T) {
	p := buildConversation()
	msgs := p.Messages()

	// Slice just the first message
	sliced := p.Slice(msgs[0].Start, msgs[0].End)
	if sliced.Len() != msgs[0].End-msgs[0].Start+1 {
		t.Fatalf("sliced len = %d", sliced.Len())
	}
	if sliced.Code[0].Op != MSG_START {
		t.Fatal("sliced should start with MSG_START")
	}
}

func TestSliceBounds(t *testing.T) {
	p := buildConversation()
	// Out-of-bounds should clamp gracefully
	sliced := p.Slice(-5, p.Len()+10)
	if sliced.Len() != p.Len() {
		t.Fatalf("expected full program, got %d", sliced.Len())
	}
}

// ─── ExtractMessage ──────────────────────────────────────────────────────────

func TestExtractMessage(t *testing.T) {
	p := buildConversation()
	msgs := p.Messages()
	extracted := p.ExtractMessage(msgs[1]) // first user message
	if extracted.Len() != msgs[1].End-msgs[1].Start+1 {
		t.Fatalf("extracted len = %d", extracted.Len())
	}
	eMsgs := extracted.Messages()
	if len(eMsgs) != 1 {
		t.Fatalf("expected 1 message in extracted, got %d", len(eMsgs))
	}
	if extracted.MessageText(eMsgs[0]) != "What is 2+2?" {
		t.Fatalf("extracted text = %q", extracted.MessageText(eMsgs[0]))
	}
}

// ─── RemoveRange ─────────────────────────────────────────────────────────────

func TestRemoveRange(t *testing.T) {
	p := buildConversation()
	msgs := p.Messages()
	// Remove the system message
	removed := p.RemoveRange(msgs[0].Start, msgs[0].End)
	if removed.SystemPrompt() != "" {
		t.Fatal("system prompt should be gone")
	}
	if removed.CountMessages() != 3 {
		t.Fatalf("expected 3 messages, got %d", removed.CountMessages())
	}
	// Original should be unmodified
	if p.CountMessages() != 4 {
		t.Fatal("original was modified")
	}
}

// ─── RemoveMessages ──────────────────────────────────────────────────────────

func TestRemoveMessages(t *testing.T) {
	p := buildConversation()
	msgs := p.Messages()
	// Remove sys and first user
	removed := p.RemoveMessages(msgs[0], msgs[1])
	if removed.CountMessages() != 2 {
		t.Fatalf("expected 2 messages, got %d", removed.CountMessages())
	}
	remaining := removed.Messages()
	if remaining[0].Role != ROLE_AST {
		t.Fatalf("first remaining should be assistant, got %s", remaining[0].Role.Name())
	}
}

// ─── ReplaceRange ────────────────────────────────────────────────────────────

func TestReplaceRange(t *testing.T) {
	p := buildConversation()
	msgs := p.Messages()
	// Replace the system prompt with a different one
	newSys := []Instruction{
		{Op: MSG_START},
		{Op: ROLE_SYS},
		{Op: TXT_CHUNK, Str: "Be concise."},
		{Op: MSG_END},
	}
	replaced := p.ReplaceRange(msgs[0].Start, msgs[0].End, newSys...)
	if replaced.SystemPrompt() != "Be concise." {
		t.Fatalf("replaced sys = %q", replaced.SystemPrompt())
	}
	if replaced.CountMessages() != 4 {
		t.Fatalf("expected 4 messages, got %d", replaced.CountMessages())
	}
}

// ─── InsertBefore / InsertAfter ──────────────────────────────────────────────

func TestInsertBefore(t *testing.T) {
	p := buildConversation()
	msgs := p.Messages()
	// Insert a new user message before the assistant message
	ins := []Instruction{
		{Op: MSG_START},
		{Op: ROLE_USR},
		{Op: TXT_CHUNK, Str: "Inserted!"},
		{Op: MSG_END},
	}
	result := p.InsertBefore(msgs[2].Start, ins...)
	if result.CountMessages() != 5 {
		t.Fatalf("expected 5 messages, got %d", result.CountMessages())
	}
	rMsgs := result.Messages()
	if result.MessageText(rMsgs[2]) != "Inserted!" {
		t.Fatalf("inserted msg text = %q", result.MessageText(rMsgs[2]))
	}
}

func TestInsertAfter(t *testing.T) {
	p := buildConversation()
	msgs := p.Messages()
	ins := []Instruction{
		{Op: MSG_START},
		{Op: ROLE_AST},
		{Op: TXT_CHUNK, Str: "Extra reply"},
		{Op: MSG_END},
	}
	result := p.InsertAfter(msgs[2].End, ins...)
	if result.CountMessages() != 5 {
		t.Fatalf("expected 5 messages, got %d", result.CountMessages())
	}
}

// ─── TruncateMessages ────────────────────────────────────────────────────────

func TestTruncateMessages(t *testing.T) {
	p := buildConversation()
	truncated := p.TruncateMessages(2)
	if truncated.CountMessages() != 2 {
		t.Fatalf("expected 2 messages, got %d", truncated.CountMessages())
	}
	// Config should still be present
	if truncated.GetModel() != "gpt-4o" {
		t.Fatalf("model lost after truncate: %q", truncated.GetModel())
	}
	// Should keep the last 2 messages (assistant + user2)
	msgs := truncated.Messages()
	if msgs[0].Role != ROLE_AST {
		t.Fatalf("first truncated msg role = %s", msgs[0].Role.Name())
	}
	if truncated.MessageText(msgs[1]) != "Thanks!" {
		t.Fatalf("second truncated msg = %q", truncated.MessageText(msgs[1]))
	}
}

func TestTruncateMessagesNoOp(t *testing.T) {
	p := buildConversation()
	truncated := p.TruncateMessages(100)
	if truncated.CountMessages() != 4 {
		t.Fatalf("expected 4 (all) messages, got %d", truncated.CountMessages())
	}
}

// ─── PrependSystemPrompt ─────────────────────────────────────────────────────

func TestPrependSystemPromptReplace(t *testing.T) {
	p := buildConversation()
	result := p.PrependSystemPrompt("New system prompt")
	if result.SystemPrompt() != "New system prompt" {
		t.Fatalf("sys prompt = %q", result.SystemPrompt())
	}
	if result.CountMessages() != 4 {
		t.Fatalf("message count changed: %d", result.CountMessages())
	}
	// Original unchanged
	if p.SystemPrompt() != "You are a helpful assistant." {
		t.Fatal("original was modified")
	}
}

func TestPrependSystemPromptInsert(t *testing.T) {
	p := NewProgram()
	p.EmitString(SET_MODEL, "gpt-4o")
	p.Emit(MSG_START)
	p.Emit(ROLE_USR)
	p.EmitString(TXT_CHUNK, "Hi")
	p.Emit(MSG_END)

	result := p.PrependSystemPrompt("Added system")
	if result.SystemPrompt() != "Added system" {
		t.Fatalf("sys prompt = %q", result.SystemPrompt())
	}
	if result.CountMessages() != 2 {
		t.Fatalf("expected 2 messages, got %d", result.CountMessages())
	}
	// System should be first
	msgs := result.Messages()
	if msgs[0].Role != ROLE_SYS {
		t.Fatal("system should be first message")
	}
}

func TestPrependSystemPromptNoMessages(t *testing.T) {
	p := NewProgram()
	p.EmitString(SET_MODEL, "gpt-4o")

	result := p.PrependSystemPrompt("Appended system")
	if result.SystemPrompt() != "Appended system" {
		t.Fatalf("sys prompt = %q", result.SystemPrompt())
	}
}

// ─── AppendUserMessage ───────────────────────────────────────────────────────

func TestAppendUserMessage(t *testing.T) {
	p := buildConversation()
	result := p.AppendUserMessage("Follow-up question")
	if result.CountMessages() != 5 {
		t.Fatalf("expected 5 messages, got %d", result.CountMessages())
	}
	span, ok := result.LastUserMessage()
	if !ok {
		t.Fatal("no user message found")
	}
	if result.MessageText(span) != "Follow-up question" {
		t.Fatalf("appended text = %q", result.MessageText(span))
	}
	// Original unchanged
	if p.CountMessages() != 4 {
		t.Fatal("original was modified")
	}
}

// ─── Immutability checks ────────────────────────────────────────────────────

func TestManipulationsAreImmutable(t *testing.T) {
	p := buildConversation()
	originalLen := p.Len()
	originalModel := p.GetModel()
	originalCount := p.CountMessages()

	// Run a bunch of manipulations — none should touch the original
	_ = p.TruncateMessages(1)
	_ = p.PrependSystemPrompt("changed")
	_ = p.AppendUserMessage("extra")
	msgs := p.Messages()
	_ = p.RemoveMessages(msgs[0])
	_ = p.RemoveRange(0, 2)
	_ = p.ReplaceRange(0, 0, Instruction{Op: SET_MODEL, Str: "x"})
	_ = p.InsertBefore(0, Instruction{Op: SET_STREAM})
	_ = p.InsertAfter(0, Instruction{Op: SET_STREAM})

	if p.Len() != originalLen {
		t.Fatalf("original len changed: %d -> %d", originalLen, p.Len())
	}
	if p.GetModel() != originalModel {
		t.Fatalf("original model changed: %s -> %s", originalModel, p.GetModel())
	}
	if p.CountMessages() != originalCount {
		t.Fatalf("original count changed: %d -> %d", originalCount, p.CountMessages())
	}
}
