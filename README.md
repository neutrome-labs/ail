# AIL — AI Intermediate Language

[![Go](https://img.shields.io/badge/Go-1.25+-00ADD8?logo=go&logoColor=white)](https://go.dev)

```
go get github.com/neutrome-labs/ail
```

AIL is a stack-based intermediate representation for AI provider interactions.
It decouples **parsing** (ingesting provider-specific JSON into opcodes) from
**emitting** (writing opcodes back out as provider-specific JSON), enabling
any-to-any conversion between different AI provider APIs.

Supported providers:

| Provider | Style Constant | Request Parse | Request Emit | Response Parse | Response Emit | Stream Parse | Stream Emit |
|---|---|---|---|---|---|---|---|
| OpenAI Chat Completions | `StyleChatCompletions` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenAI Responses | `StyleResponses` | ✅ | ✅ | ✅ | — | ✅ | — |
| Anthropic Messages | `StyleAnthropic` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Google GenAI | `StyleGoogleGenAI` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## Quick Start

### Convert a request from one provider format to another

```go
import "github.com/neutrome-labs/ail"

// OpenAI Chat Completions → Anthropic Messages
out, err := ail.ConvertRequest(body, ail.StyleChatCompletions, ail.StyleAnthropic)

// Anthropic Messages → Google GenAI
out, err := ail.ConvertRequest(body, ail.StyleAnthropic, ail.StyleGoogleGenAI)
```

### Convert a non-streaming response

```go
out, err := ail.ConvertResponse(body, ail.StyleAnthropic, ail.StyleChatCompletions)
```

### Convert streaming chunks in real-time

```go
conv, err := ail.NewStreamConverter(ail.StyleAnthropic, ail.StyleChatCompletions)

for _, chunk := range upstreamChunks {
    outputs, err := conv.Push(chunk)
    for _, out := range outputs {
        fmt.Fprintf(w, "data: %s\n\n", out)
        flusher.Flush()
    }
}
// Flush buffered tool calls at end of stream
final, _ := conv.Flush()
for _, out := range final {
    fmt.Fprintf(w, "data: %s\n\n", out)
}
```

### Work with the AIL program directly

```go
parser, _ := ail.GetParser(ail.StyleChatCompletions)
prog, _ := parser.ParseRequest(body)

// Inspect
fmt.Println(prog.GetModel())    // "gpt-4o"
fmt.Println(prog.IsStreaming())  // true

// Modify the program
prog.SetModel("claude-sonnet-4-20250514")

// Debug: print human-readable disassembly
fmt.Println(prog.Disasm())

// Emit to a different provider
emitter, _ := ail.GetEmitter(ail.StyleAnthropic)
out, _ := emitter.EmitRequest(prog)
```

### Pass programs through context

```go
// Store in context (avoids re-serialization in recursive handler chains)
ctx = ail.ContextWithProgram(ctx, prog)

// Retrieve later
prog, ok := ail.ProgramFromContext(ctx)
```

## Example: End-to-End Conversion

```jsonc
// Input: OpenAI Chat Completions Request
{
  "model": "gpt-5-mini",
  "messages": [
    {
      "role": "user",
      "content": "How many r's are in the word 'strawberry'?"
    }
  ]
}
```

```asm
; AIL Representation (prog.Disasm() output)
SET_MODEL gpt-5-mini
MSG_START
  ROLE_USR
  TXT_CHUNK How many r's are in the word 'strawberry'?
MSG_END
```

```jsonc
// Output: OpenAI Responses API Request (via EmitRequest)
{
  "model": "gpt-5-mini",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "How many r's are in the word 'strawberry'?"
        }
      ]
    }
  ]
}
```

## Design Principles

1. **Zero-Copy Where Possible** — Large payloads (images, audio) are stored in a side buffer and referenced by index. The instruction stream itself contains only opcodes and lightweight arguments.
2. **Stack-Based** — The emitter processes opcodes linearly. No recursive descent needed.
3. **Provider-Agnostic Core** — The opcode set covers the common denominator. Provider-specific parameters are passed through via `SET_META` and `EXT_DATA`.
4. **Binary Wire Format** — Each opcode is a single byte, enabling fast internal transfer and comparison.

## Architecture

### Interfaces

Every provider is implemented as a pair of structs — a **Parser** and an **Emitter**. They satisfy up to three interface pairs each:

```go
// Request conversion
type Parser  interface { ParseRequest(body []byte) (*Program, error) }
type Emitter interface { EmitRequest(prog *Program) ([]byte, error) }

// Non-streaming response conversion
type ResponseParser  interface { ParseResponse(body []byte) (*Program, error) }
type ResponseEmitter interface { EmitResponse(prog *Program) ([]byte, error) }

// Streaming chunk conversion
type StreamChunkParser  interface { ParseStreamChunk(body []byte) (*Program, error) }
type StreamChunkEmitter interface { EmitStreamChunk(prog *Program) ([]byte, error) }
```

Use the factory functions to get the right parser/emitter for a style:

```go
ail.GetParser(style)             // → Parser
ail.GetEmitter(style)            // → Emitter
ail.GetResponseParser(style)     // → ResponseParser
ail.GetResponseEmitter(style)    // → ResponseEmitter
ail.GetStreamChunkParser(style)  // → StreamChunkParser
ail.GetStreamChunkEmitter(style) // → StreamChunkEmitter
```

### Program

`Program` holds an ordered list of `Instruction`s plus a side-buffer for large binary blobs:

```go
type Program struct {
    Code    []Instruction
    Buffers [][]byte
}

type Instruction struct {
    Op   Opcode
    Str  string           // TXT_CHUNK, DEF_NAME, SET_MODEL, CALL_START, etc.
    Num  float64          // SET_TEMP, SET_TOPP
    Int  int32            // SET_MAX
    JSON json.RawMessage  // DEF_SCHEMA, CALL_ARGS, USAGE, EXT_DATA, STREAM_TOOL_DELTA
    Key  string           // SET_META, EXT_DATA (key part)
    Ref  uint32           // IMG_REF, AUD_REF, TXT_REF
}
```

Programs support building, querying, cloning, appending, and disassembly:

```go
p := ail.NewProgram()
p.EmitString(ail.SET_MODEL, "gpt-4o")
p.EmitFloat(ail.SET_TEMP, 0.7)
p.Emit(ail.MSG_START)
p.Emit(ail.ROLE_USR)
p.EmitString(ail.TXT_CHUNK, "Hello")
p.Emit(ail.MSG_END)
fmt.Println(p.GetModel())     // "gpt-4o"
fmt.Println(p.IsStreaming())   // false
fmt.Println(p.Len())          // 7

clone := p.Clone()             // deep copy
merged := p.Append(other)     // concatenate (re-indexes buffer refs)
```

### StreamConverter

`StreamConverter` handles stateful, real-time streaming translation between providers. It manages:

- **Metadata carry-forward** — `RESP_ID` and `RESP_MODEL` from the first chunk are injected into all subsequent emitted chunks (some formats require this on every event).
- **Event splitting** — One source event may produce multiple output events (e.g., Anthropic requires separate SSE events per content type).
- **Tool call buffering** — Targets that require complete function calls in a single chunk (e.g., Google GenAI) buffer `STREAM_TOOL_DELTA` fragments until flushed.

```go
conv, _ := ail.NewStreamConverter(from, to)

// Push raw bytes (parses internally)
outputs, _ := conv.Push(chunk)

// Or push an already-parsed program (useful after plugin modification)
outputs, _ := conv.PushProgram(prog)

// Flush remaining buffered data at end of stream
final, _ := conv.Flush()
```

## Binary Encoding

Programs can be serialized to a compact binary format for storage or wire transfer.

### Wire Layout

```
┌────────────────┬─────────┬──────────────────────────────────────┬──────────────┐
│  Magic (4B)    │ Ver (1B)│  Side-Buffers                        │ Instructions │
│  "AIL\x00"    │  0x01   │  [count][len₀][data₀][len₁][data₁]… │ [op][args]…  │
└────────────────┴─────────┴──────────────────────────────────────┴──────────────┘
```

### Argument Types

| Type     | Encoding                                       |
|----------|------------------------------------------------|
| -        | No argument (opcode only)                      |
| String   | 4-byte little-endian length prefix + UTF-8 bytes |
| Float    | 8-byte IEEE 754 double (little-endian)         |
| Int      | 4-byte little-endian signed integer            |
| JSON     | 4-byte LE length prefix + raw JSON bytes       |
| RefID    | 4-byte LE buffer index                         |
| Key,Val  | Two length-prefixed strings back-to-back       |
| Key,JSON | Length-prefixed key string + length-prefixed JSON |

### Encode / Decode

```go
// Encode to binary
var buf bytes.Buffer
prog.Encode(&buf)

// Decode from binary
prog, err := ail.Decode(&buf)
```

## Opcode Table

### Structure (0x10–0x1F)

| Mnemonic    | Byte   | Args | Description                              |
|-------------|--------|------|------------------------------------------|
| `MSG_START` | `0x10` | -    | Begin a message block                    |
| `MSG_END`   | `0x11` | -    | End a message block                      |
| `ROLE_SYS`  | `0x12` | -    | Set role to system                       |
| `ROLE_USR`  | `0x13` | -    | Set role to user                         |
| `ROLE_AST`  | `0x14` | -    | Set role to assistant                    |
| `ROLE_TOOL` | `0x15` | -    | Set role to tool/function result         |

### Content (0x20–0x2F)

| Mnemonic    | Byte   | Args   | Description                            |
|-------------|--------|--------|----------------------------------------|
| `TXT_CHUNK` | `0x20` | String | Text content segment                   |
| `IMG_REF`   | `0x21` | RefID  | Reference to image in side-buffer      |
| `AUD_REF`   | `0x22` | RefID  | Reference to audio in side-buffer      |
| `TXT_REF`   | `0x23` | RefID  | Reference to large text in side-buffer |

### Tool Definition (0x30–0x3F)

| Mnemonic    | Byte   | Args   | Description                            |
|-------------|--------|--------|----------------------------------------|
| `DEF_START` | `0x30` | -      | Begin tool definitions block           |
| `DEF_NAME`  | `0x31` | String | Tool function name                     |
| `DEF_DESC`  | `0x32` | String | Tool function description              |
| `DEF_SCHEMA`| `0x33` | JSON   | Tool parameter schema (JSON)           |
| `DEF_END`   | `0x34` | -      | End tool definitions block             |

### Tool Call (0x40–0x4F)

| Mnemonic     | Byte   | Args   | Description                           |
|--------------|--------|--------|---------------------------------------|
| `CALL_START` | `0x40` | String | Begin tool call (call ID)             |
| `CALL_NAME`  | `0x41` | String | Function name being called            |
| `CALL_ARGS`  | `0x42` | JSON   | Function arguments (JSON)             |
| `CALL_END`   | `0x43` | -      | End tool call                         |

### Tool Result (0x48–0x4A)

| Mnemonic      | Byte   | Args   | Description                          |
|---------------|--------|--------|--------------------------------------|
| `RESULT_START`| `0x48` | String | Begin tool result (call ID)          |
| `RESULT_DATA` | `0x49` | String | Tool result content                  |
| `RESULT_END`  | `0x4A` | -      | End tool result                      |

### Response Metadata (0x50–0x5F)

| Mnemonic      | Byte   | Args   | Description                          |
|---------------|--------|--------|--------------------------------------|
| `RESP_ID`     | `0x50` | String | Response ID                          |
| `RESP_MODEL`  | `0x51` | String | Model that generated the response    |
| `RESP_DONE`   | `0x52` | String | Finish reason (stop/tool_calls/length)|
| `USAGE`       | `0x53` | JSON   | Token usage statistics               |

### Stream Events (0x60–0x6F)

| Mnemonic            | Byte   | Args   | Description                    |
|---------------------|--------|--------|--------------------------------|
| `STREAM_START`      | `0x60` | -      | Begin streaming response       |
| `STREAM_DELTA`      | `0x61` | String | Text delta chunk               |
| `STREAM_TOOL_DELTA` | `0x62` | JSON   | Tool call argument delta       |
| `STREAM_END`        | `0x63` | -      | End streaming response         |

### Configuration (0xF0–0xFF)

| Mnemonic    | Byte   | Args     | Description                          |
|-------------|--------|----------|--------------------------------------|
| `SET_MODEL` | `0xF0` | String   | Set target model name                |
| `SET_TEMP`  | `0xF1` | Float    | Set temperature                      |
| `SET_TOPP`  | `0xF2` | Float    | Set top_p                            |
| `SET_STOP`  | `0xF3` | String   | Add stop sequence                    |
| `SET_MAX`   | `0xF4` | Int      | Set max tokens                       |
| `SET_STREAM`| `0xF5` | -        | Enable streaming mode                |
| `EXT_DATA`  | `0xFE` | Key,JSON | Provider-specific extension data     |
| `SET_META`  | `0xFF` | Key,Val  | Set arbitrary metadata key=value     |

## Provider Mapping Details

### OpenAI Chat Completions

| AIL Opcode   | Chat Completions Equivalent                    |
|--------------|------------------------------------------------|
| `ROLE_SYS`   | `"role": "system"` (also parses `"developer"`) |
| `ROLE_USR`   | `"role": "user"`                               |
| `ROLE_AST`   | `"role": "assistant"`                          |
| `ROLE_TOOL`  | `"role": "tool"` + `tool_call_id`              |
| `TXT_CHUNK`  | `"content": "..."` (string or content parts)   |
| `IMG_REF`    | `image_url` content part                       |
| `AUD_REF`    | `input_audio` content part                     |
| `DEF_*`      | `"tools": [{ "type": "function", "function": {...} }]` |
| `CALL_*`     | `tool_calls` in assistant message              |
| `SET_MODEL`  | `"model": "..."`                               |
| `SET_TEMP`   | `"temperature": ...`                           |
| `SET_MAX`    | `"max_tokens"` or `"max_completion_tokens"`    |
| `SET_STREAM` | `"stream": true` + `stream_options: {include_usage: true}` |
| `SET_META`   | `"metadata": {...}` (except `media_type` key)  |
| `EXT_DATA`   | Any remaining top-level fields passed through  |

### OpenAI Responses

| AIL Opcode   | Responses API Equivalent                       |
|--------------|------------------------------------------------|
| `ROLE_SYS`   | Top-level `"instructions"` field               |
| `ROLE_USR`   | `"role": "user"` in `input[]`                  |
| `ROLE_AST`   | `"role": "assistant"` in `input[]`             |
| `TXT_CHUNK`  | `{"type": "input_text", "text": "..."}`        |
| `DEF_*`      | `tools[]` with flat structure (`name` at top level, no `function` wrapper) |
| `CALL_*`     | `function_call` output item                    |
| `SET_MODEL`  | `"model": "..."`                               |
| `SET_MAX`    | `"max_output_tokens": ...`                     |

### Anthropic Messages

| AIL Opcode   | Anthropic Equivalent                           |
|--------------|------------------------------------------------|
| `ROLE_SYS`   | Top-level `"system"` parameter                 |
| `ROLE_USR`   | `"role": "user"`                               |
| `ROLE_AST`   | `"role": "assistant"`                          |
| `ROLE_TOOL`  | `"role": "user"` + `tool_result` content block |
| `TXT_CHUNK`  | `{"type": "text", "text": "..."}`              |
| `IMG_REF`    | `{"type": "image", "source": {"type": "base64", ...}}` |
| `DEF_SCHEMA` | `"input_schema"` (not `"parameters"`)          |
| `SET_MAX`    | `"max_tokens": ...` (required by Anthropic)    |
| `SET_STOP`   | `"stop_sequences": [...]`                      |
| `SET_META`   | `"metadata": {...}` (except `media_type` key)  |
| `RESP_DONE`  | Stop reason mapped: `stop`↔`end_turn`, `tool_calls`↔`tool_use`, `length`↔`max_tokens` |

### Google GenAI

| AIL Opcode   | Google GenAI Equivalent                        |
|--------------|------------------------------------------------|
| `ROLE_SYS`   | `"system_instruction": {"parts": [...]}`       |
| `ROLE_USR`   | `"role": "user"`                               |
| `ROLE_AST`   | `"role": "model"`                              |
| `ROLE_TOOL`  | `"role": "function"` + `functionResponse` part |
| `TXT_CHUNK`  | `{"text": "..."}` in parts                    |
| `IMG_REF`    | `{"inlineData": {"mimeType": "...", "data": "..."}}` |
| `DEF_*`      | `tools[].function_declarations[]`              |
| `CALL_*`     | `functionCall` part                            |
| `SET_TEMP`   | `generation_config.temperature`                 |
| `SET_TOPP`   | `generation_config.topP`                        |
| `SET_MAX`    | `generation_config.maxOutputTokens`             |
| `SET_STOP`   | `generation_config.stopSequences`               |
| `RESP_DONE`  | Finish reason mapped: `stop`↔`STOP`, `length`↔`MAX_TOKENS` |

## Theory of Operation

### Incompatibility Handling

The power of AIL lies in how each **Emitter** interprets the same instruction stream differently to match the target provider's conventions:

#### System Prompt Placement

```
Input: MSG_START → ROLE_SYS → TXT_CHUNK "Be polite" → MSG_END

OpenAI Emitter:    {"role": "system", "content": "Be polite"} in messages[]
Anthropic Emitter: "system": "Be polite" as top-level field
Google Emitter:    "system_instruction": {"parts": [{"text": "Be polite"}]}
```

#### Tool Results

```
Input: MSG_START → ROLE_TOOL → RESULT_START "call_123" → RESULT_DATA "OK" → RESULT_END → MSG_END

OpenAI:    {"role": "tool", "tool_call_id": "call_123", "content": "OK"}
Anthropic: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_123", "content": "OK"}]}
Google:    {"role": "function", "parts": [{"functionResponse": {"name": "...", "response": {...}}}]}
```

#### Extension Data Passthrough

```
Input: EXT_DATA "response_format" {"type":"json_object"}

OpenAI Emitter:    Adds "response_format": {"type":"json_object"} to request body
Anthropic Emitter: Passes through (provider may ignore unsupported fields)
```

### Stream Conversion Edge Cases

The `StreamConverter` handles several structural mismatches:

- **Anthropic targets** require each event type (text delta, tool delta, start, stop) to be a separate SSE event with a different JSON structure — so one source chunk may produce multiple output events.
- **Google GenAI targets** require complete function calls in a single chunk — so tool-call argument deltas are buffered until `Flush()`.
- **Metadata injection** — Some formats (OpenAI) require `id` and `model` on every chunk, while others (Anthropic) send them only once. The converter remembers and injects as needed.

### Program Manipulation (Plugins)

Plugins operate on the `Program` directly rather than provider-specific JSON:

```go
// Inject a system prompt at the beginning
prefix := ail.NewProgram()
prefix.Emit(ail.MSG_START)
prefix.Emit(ail.ROLE_SYS)
prefix.EmitString(ail.TXT_CHUNK, "Always be helpful and safe.")
prefix.Emit(ail.MSG_END)
result := prefix.Append(prog) // buffer refs are re-indexed automatically
```

## Assembly Notation

`Program.Disasm()` produces a human-readable assembly listing with automatic indentation inside block opcodes:

```asm
SET_MODEL gpt-4o
SET_TEMP 0.1000
MSG_START
  ROLE_SYS
  TXT_CHUNK Be brief.
MSG_END
MSG_START
  ROLE_USR
  TXT_CHUNK Hello
MSG_END
SET_STREAM
DEF_START
  DEF_NAME get_weather
  DEF_DESC Get current weather for a location
  DEF_SCHEMA {"type":"object","properties":{"location":{"type":"string"}}}
DEF_END
```

Comments prefixed with `;` can appear in assembly text and are silently ignored by the parser.

### Binary Layout Example

`{"role": "user", "content": "Hello"}` in AIL binary:

```
10                              ; MSG_START
13                              ; ROLE_USR
20 05 00 00 00 48 65 6C 6C 6F  ; TXT_CHUNK len=5 "Hello"
11                              ; MSG_END
```
