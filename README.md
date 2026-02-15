# AIL — AI Intermediate Language Opcode Specification

**Version**: 1.0  
**Status**: Active  
**Updated**: 2026-02-15

## Overview

AIL (AI Intermediate Language) is a stack-based intermediate representation for AI provider interactions. It decouples the **parsing** of incoming requests from the **emitting** of outgoing requests, enabling any-to-any conversion between different AI provider APIs (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Google GenAI, etc.).

### Example (Chat Completions API to Responses Upstream)

```jsonc
// OpenAI Chat Completions Request
{
  "model": "gpt-5-mini",
  "messages": [
    {
      "role": "user",
      "content": "How many r`s are in the word `strawberry?`"
    }
  ]
}
```

```asm
; AIL Representation
SET_MODEL "gpt-5-mini"
MSG_START
  ROLE_USR
  TXT_CHUNK "How many r`s are in the word `strawberry?`"
MSG_END
```

```jsonc
// OpenAI Responses API Request
{
  "model": "gpt-5-mini",
  "prompt": "How many r`s are in the word `strawberry?`"
}
```

```jsonc
// OpenAI Responses API Response
{
  "id": "resp_XXXXXXXX",
  "model": "gpt-5-mini-2025-08-07",
  "usage": {
    "completion_tokens": 275,
    "prompt_tokens": 20,
    "total_tokens": 295
  },
  "choices": [
    {
      "text": "There are 3 r's in \"strawberry\" — they are the 3rd, 8th, and 9th letters."
    }
  ]
}
```

```asm
; AIL Representation of Response
RESP_ID "resp_XXXXXXXX"
RESP_MODEL "gpt-5-mini-2025-08-07"
USAGE {"completion_tokens":275,"prompt_tokens":20,"total_tokens":295}
MSG_START
  ROLE_AST
  TXT_CHUNK "There are 3 r's in \"strawberry\" — they are the 3rd, 8th, and 9th letters."
  RESP_DONE "stop"
MSG_END
```

```jsonc
// OpenAI Chat Completions Response
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"There are 3 r's in \"strawberry\" — they are the 3rd, 8th, and 9th letters.","role":"assistant"}}],"id":"resp_XXXXXXXX","model":"gpt-5-mini-2025-08-07","object":"chat.completion","usage":{"completion_tokens":275,"prompt_tokens":20,"total_tokens":295}}
```


### Design Principles

1. **Zero-Copy Where Possible**: Large payloads (text, images, audio) are stored in a side buffer and referenced by pointer. The IL stream itself contains only opcodes and lightweight arguments.
2. **Stack-Based**: The emitter processes opcodes linearly. No recursive descent needed.
3. **Provider-Agnostic Core**: The opcode set covers the common denominator. Provider-specific parameters are passed through via `SET_META` and `EXT_DATA`.
4. **Binary Wire Format**: Each opcode is a single byte, enabling fast internal transfer and comparison.

## Binary Encoding

```
┌─────────┬─────────────────────────────────┐
│ 1 byte  │ Variable-length arguments        │
│ Opcode  │ (type depends on opcode)         │
└─────────┴─────────────────────────────────┘
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

## Opcode Table

### Structure (0x10-0x1F)

| Mnemonic    | Byte   | Args | Description                              |
|-------------|--------|------|------------------------------------------|
| `MSG_START` | `0x10` | -    | Begin a message block                    |
| `MSG_END`   | `0x11` | -    | End a message block                      |
| `ROLE_SYS`  | `0x12` | -    | Set role to system                       |
| `ROLE_USR`  | `0x13` | -    | Set role to user                         |
| `ROLE_AST`  | `0x14` | -    | Set role to assistant                    |
| `ROLE_TOOL` | `0x15` | -    | Set role to tool/function result         |

### Content (0x20-0x2F)

| Mnemonic    | Byte   | Args   | Description                            |
|-------------|--------|--------|----------------------------------------|
| `TXT_CHUNK` | `0x20` | String | Text content segment                   |
| `IMG_REF`   | `0x21` | RefID  | Reference to image in buffer           |
| `AUD_REF`   | `0x22` | RefID  | Reference to audio in buffer           |
| `TXT_REF`   | `0x23` | RefID  | Reference to large text in buffer      |

### Tool Definition (0x30-0x3F)

| Mnemonic    | Byte   | Args   | Description                            |
|-------------|--------|--------|----------------------------------------|
| `DEF_START` | `0x30` | -      | Begin tool definitions block           |
| `DEF_NAME`  | `0x31` | String | Tool function name                     |
| `DEF_DESC`  | `0x32` | String | Tool function description              |
| `DEF_SCHEMA`| `0x33` | JSON   | Tool parameter schema (JSON)           |
| `DEF_END`   | `0x34` | -      | End tool definitions block             |

### Tool Call (0x40-0x4F)

| Mnemonic     | Byte   | Args   | Description                           |
|--------------|--------|--------|---------------------------------------|
| `CALL_START` | `0x40` | String | Begin tool call (call ID)             |
| `CALL_NAME`  | `0x41` | String | Function name being called            |
| `CALL_ARGS`  | `0x42` | JSON   | Function arguments (JSON)             |
| `CALL_END`   | `0x43` | -      | End tool call                         |

### Tool Result (0x48-0x4F)

| Mnemonic      | Byte   | Args   | Description                          |
|---------------|--------|--------|--------------------------------------|
| `RESULT_START`| `0x48` | String | Begin tool result (call ID)          |
| `RESULT_DATA` | `0x49` | String | Tool result content                  |
| `RESULT_END`  | `0x4A` | -      | End tool result                      |

### Response Metadata (0x50-0x5F)

| Mnemonic      | Byte   | Args   | Description                          |
|---------------|--------|--------|--------------------------------------|
| `RESP_ID`     | `0x50` | String | Response ID                          |
| `RESP_MODEL`  | `0x51` | String | Model that generated the response    |
| `RESP_DONE`   | `0x52` | String | Finish reason (stop/tool_calls/length)|
| `USAGE`       | `0x53` | JSON   | Token usage statistics               |

### Configuration (0xF0-0xFF)

| Mnemonic   | Byte   | Args    | Description                            |
|------------|--------|---------|----------------------------------------|
| `SET_MODEL`| `0xF0` | String  | Set target model name                  |
| `SET_TEMP` | `0xF1` | Float   | Set temperature                        |
| `SET_TOPP` | `0xF2` | Float   | Set top_p                              |
| `SET_STOP` | `0xF3` | String  | Add stop sequence                      |
| `SET_MAX`  | `0xF4` | Int     | Set max tokens                         |
| `SET_STREAM`| `0xF5`| -       | Enable streaming mode                  |
| `SET_META` | `0xFF` | Key,Val | Set arbitrary metadata key=value       |
| `EXT_DATA` | `0xFE` | Key,JSON| Provider-specific extension data        |

### Stream Events (0x60-0x6F)

| Mnemonic       | Byte   | Args   | Description                         |
|----------------|--------|--------|-------------------------------------|
| `STREAM_START` | `0x60` | -      | Begin streaming response            |
| `STREAM_DELTA` | `0x61` | String | Text delta chunk                    |
| `STREAM_TOOL_DELTA`| `0x62` | JSON | Tool call argument delta          |
| `STREAM_END`   | `0x63` | -      | End streaming response              |

## Provider Mapping

### OpenAI Chat Completions

| AIL Opcode   | Chat Completions Equivalent                    |
|--------------|------------------------------------------------|
| `MSG_START`  | `{` in `messages[]`                            |
| `MSG_END`    | `}` in `messages[]`                            |
| `ROLE_SYS`   | `"role": "system"`                             |
| `ROLE_USR`   | `"role": "user"`                               |
| `ROLE_AST`   | `"role": "assistant"`                          |
| `ROLE_TOOL`  | `"role": "tool"` + `tool_call_id`              |
| `TXT_CHUNK`  | `"content": "..."`                             |
| `IMG_REF`    | `image_url` content part                       |
| `AUD_REF`    | `input_audio` content part                     |
| `DEF_START`  | `"tools": [`                                   |
| `DEF_NAME`   | `function.name`                                |
| `DEF_DESC`   | `function.description`                         |
| `DEF_SCHEMA` | `function.parameters`                          |
| `DEF_END`    | `]` end of tools                               |
| `CALL_START` | `tool_calls` object in assistant message       |
| `CALL_NAME`  | `function.name`                                |
| `CALL_ARGS`  | `function.arguments`                           |
| `SET_MODEL`  | `"model": "..."`                               |
| `SET_TEMP`   | `"temperature": ...`                           |
| `SET_STREAM` | `"stream": true`                               |
| `EXT_DATA`   | Any remaining top-level fields passed through  |

### OpenAI Responses API

| AIL Opcode   | Responses API Equivalent                       |
|--------------|------------------------------------------------|
| `MSG_START`  | item in `input[]`                              |
| `ROLE_SYS`   | `"instructions"` top-level field               |
| `ROLE_USR`   | `"role": "user"`                               |
| `ROLE_AST`   | `"role": "assistant"`                          |
| `TXT_CHUNK`  | content text part                              |
| `DEF_*`      | `tools[]` with flat structure (name at top)    |
| `CALL_*`     | `function_call` output item                    |
| `SET_MODEL`  | `"model": "..."`                               |
| `SET_MAX`    | `"max_output_tokens": ...`                     |

### Anthropic Messages

| AIL Opcode   | Anthropic Equivalent                           |
|--------------|------------------------------------------------|
| `ROLE_SYS`   | Top-level `"system"` parameter (not in messages)|
| `ROLE_USR`   | `"role": "user"`                               |
| `ROLE_AST`   | `"role": "assistant"`                          |
| `ROLE_TOOL`  | `"role": "user"` + `tool_result` content block |
| `TXT_CHUNK`  | `{"type":"text","text":"..."}`                 |
| `IMG_REF`    | `{"type":"image","source":{...}}`              |
| `DEF_SCHEMA` | `"input_schema"` instead of `"parameters"`     |
| `SET_MAX`    | `"max_tokens": ...` (required)                 |
| `SET_STOP`   | `"stop_sequences": [...]`                      |

### Google GenAI

| AIL Opcode   | Google GenAI Equivalent                        |
|--------------|------------------------------------------------|
| `ROLE_SYS`   | `"system_instruction"` parameter               |
| `ROLE_USR`   | `"role": "user"`                               |
| `ROLE_AST`   | `"role": "model"`                              |
| `ROLE_TOOL`  | `"role": "function"`                           |
| `TXT_CHUNK`  | `parts: [{"text": "..."}]`                     |
| `DEF_START`  | `tools: { function_declarations: [`            |
| `SET_TEMP`   | `generationConfig.temperature`                 |
| `SET_STOP`   | `generationConfig.stopSequences`               |
| `SET_MODEL`  | URL parameter                                  |

## Theory of Operation

### Incompatibility Handling

The power of AIL lies in how each **Emitter** interprets the same instruction stream differently.

#### The ROLE_SYS Problem

```
Input: ROLE_SYS → TXT_CHUNK "Be polite"

OpenAI Emitter: {"role": "system", "content": "Be polite"} in messages[]
Anthropic Emitter: Assigns "Be polite" to top-level "system" field
Google Emitter: Assigns "Be polite" to "system_instruction" field
```

#### The ROLE_TOOL vs tool_result Problem

```
Input: ROLE_TOOL → RESULT_START "call_123" → RESULT_DATA "Success" → RESULT_END

OpenAI Emitter: {"role": "tool", "tool_call_id": "call_123", "content": "Success"}
Anthropic Emitter: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_123", "content": "Success"}]}
```

#### Extension Data Passthrough

```
Input: EXT_DATA "response_format" {"type":"json_object"}

OpenAI Emitter: Adds "response_format": {"type":"json_object"} to request
Anthropic Emitter: Ignores (not supported) or maps to equivalent
```

### Binary Layout Example

`{"role": "user", "content": "Hello"}` in AIL binary:

```
10                              ; MSG_START
13                              ; ROLE_USR
20 05 00 00 00 48 65 6C 6C 6F  ; TXT_CHUNK len=5 "Hello"
11                              ; MSG_END
```

Full request with model and temperature:

```
F0 06 00 00 00 67 70 74 2D 34 6F  ; SET_MODEL len=6 "gpt-4o"
F1 9A 99 99 99 99 99 B9 3F        ; SET_TEMP 0.1 (IEEE754)
10                                 ; MSG_START
12                                 ; ROLE_SYS  
20 09 00 00 00 42 65 20 62 72 69 65 66 2E  ; TXT_CHUNK "Be brief."
11                                 ; MSG_END
10                                 ; MSG_START
13                                 ; ROLE_USR
20 05 00 00 00 48 65 6C 6C 6F     ; TXT_CHUNK "Hello"
11                                 ; MSG_END
F5                                 ; SET_STREAM
```

### Plugin Interaction

Plugins operate on the `Program` (the list of instructions) rather than provider-specific JSON. This means a plugin can:

1. **Inspect**: Scan for specific opcodes (e.g., count messages, find tool calls)
2. **Modify**: Insert, remove, or rewrite instructions (e.g., strip tool calls, inject system prompts)
3. **Transform**: Map between opcode sequences (e.g., merge consecutive TXT_CHUNKs)

```go
// Example: A plugin that injects a system prompt
func (p *InjectSystemPrompt) Transform(prog *ail.Program) *ail.Program {
    // Prepend: MSG_START → ROLE_SYS → TXT_CHUNK "..." → MSG_END
    prefix := ail.NewProgram()
    prefix.Emit(ail.MSG_START)
    prefix.Emit(ail.ROLE_SYS)
    prefix.EmitString(ail.TXT_CHUNK, "Always be helpful.")
    prefix.Emit(ail.MSG_END)
    return prefix.Append(prog)
}
```

## Assembly Notation

For debugging and logging, AIL instructions can be printed in a human-readable assembly format:

```asm
SET_MODEL "gpt-4o"
SET_TEMP 0.1
MSG_START
  ROLE_SYS
  TXT_CHUNK "Be brief."
MSG_END
MSG_START
  ROLE_USR
  TXT_CHUNK "Hello"
MSG_END
SET_STREAM
DEF_START
  DEF_NAME "get_weather"
  DEF_DESC "Get current weather for a location"
  DEF_SCHEMA {"type":"object","properties":{"location":{"type":"string"}}}
DEF_END
```
