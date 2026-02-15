package ail

import (
	"encoding/json"
)

// ─── OpenAI Chat Completions Emitter ─────────────────────────────────────────

// ChatCompletionsEmitter converts an AIL Program into OpenAI Chat Completions JSON.
type ChatCompletionsEmitter struct{}

func (e *ChatCompletionsEmitter) EmitRequest(prog *Program) ([]byte, error) {
	result := make(map[string]any)
	var messages []map[string]any
	var tools []map[string]any

	var currentMsg map[string]any
	var currentRole string
	var contentParts []any // for multimodal messages
	var textContent string
	var isMultimodal bool
	var toolCalls []map[string]any

	// Tool definition state
	var currentTool map[string]any
	inToolDefs := false

	// Tool result state
	var currentToolCallID string

	// Stop sequences
	var stopSeqs []string

	for _, inst := range prog.Code {
		switch inst.Op {

		// ── Config ──
		case SET_MODEL:
			result["model"] = inst.Str
		case SET_TEMP:
			result["temperature"] = inst.Num
		case SET_TOPP:
			result["top_p"] = inst.Num
		case SET_MAX:
			result["max_tokens"] = inst.Int
		case SET_STOP:
			stopSeqs = append(stopSeqs, inst.Str)
		case SET_STREAM:
			result["stream"] = true
			result["stream_options"] = map[string]any{"include_usage": true}

		// ── Messages ──
		case MSG_START:
			currentMsg = make(map[string]any)
			currentRole = ""
			textContent = ""
			contentParts = nil
			isMultimodal = false
			toolCalls = nil
			currentToolCallID = ""

		case ROLE_SYS:
			currentRole = "system"
		case ROLE_USR:
			currentRole = "user"
		case ROLE_AST:
			currentRole = "assistant"
		case ROLE_TOOL:
			currentRole = "tool"

		case TXT_CHUNK:
			if isMultimodal {
				contentParts = append(contentParts, map[string]any{
					"type": "text",
					"text": inst.Str,
				})
			} else {
				textContent += inst.Str
			}

		case IMG_REF:
			isMultimodal = true
			url := ""
			if int(inst.Ref) < len(prog.Buffers) {
				url = string(prog.Buffers[inst.Ref])
			}
			// Promote existing text to multimodal
			if textContent != "" {
				contentParts = append(contentParts, map[string]any{
					"type": "text",
					"text": textContent,
				})
				textContent = ""
			}
			contentParts = append(contentParts, map[string]any{
				"type": "image_url",
				"image_url": map[string]any{
					"url": url,
				},
			})

		case AUD_REF:
			isMultimodal = true
			data := ""
			if int(inst.Ref) < len(prog.Buffers) {
				data = string(prog.Buffers[inst.Ref])
			}
			if textContent != "" {
				contentParts = append(contentParts, map[string]any{
					"type": "text",
					"text": textContent,
				})
				textContent = ""
			}
			contentParts = append(contentParts, map[string]any{
				"type":        "input_audio",
				"input_audio": map[string]any{"data": data},
			})

		case CALL_START:
			tc := map[string]any{
				"id":   inst.Str,
				"type": "function",
			}
			toolCalls = append(toolCalls, tc)

		case CALL_NAME:
			if len(toolCalls) > 0 {
				last := toolCalls[len(toolCalls)-1]
				fn, _ := last["function"].(map[string]any)
				if fn == nil {
					fn = make(map[string]any)
				}
				fn["name"] = inst.Str
				last["function"] = fn
			}

		case CALL_ARGS:
			if len(toolCalls) > 0 {
				last := toolCalls[len(toolCalls)-1]
				fn, _ := last["function"].(map[string]any)
				if fn == nil {
					fn = make(map[string]any)
				}
				fn["arguments"] = string(inst.JSON)
				last["function"] = fn
			}

		case CALL_END:
			// tool call already added to toolCalls

		case RESULT_START:
			currentToolCallID = inst.Str

		case RESULT_DATA:
			textContent = inst.Str

		case RESULT_END:
			// will be finalized in MSG_END

		case MSG_END:
			if currentMsg != nil {
				currentMsg["role"] = currentRole

				if currentRole == "tool" && currentToolCallID != "" {
					currentMsg["tool_call_id"] = currentToolCallID
					currentMsg["content"] = textContent
				} else if isMultimodal {
					currentMsg["content"] = contentParts
				} else if textContent != "" {
					currentMsg["content"] = textContent
				}

				if len(toolCalls) > 0 {
					currentMsg["tool_calls"] = toolCalls
				}

				messages = append(messages, currentMsg)
				currentMsg = nil
			}

		// ── Tool Definitions ──
		case DEF_START:
			inToolDefs = true
			currentTool = nil

		case DEF_NAME:
			if inToolDefs {
				if currentTool != nil {
					tools = append(tools, currentTool)
				}
				currentTool = map[string]any{
					"type":     "function",
					"function": map[string]any{"name": inst.Str},
				}
			}

		case DEF_DESC:
			if currentTool != nil {
				fn := currentTool["function"].(map[string]any)
				fn["description"] = inst.Str
			}

		case DEF_SCHEMA:
			if currentTool != nil {
				fn := currentTool["function"].(map[string]any)
				fn["parameters"] = json.RawMessage(inst.JSON)
			}

		case DEF_END:
			if inToolDefs && currentTool != nil {
				tools = append(tools, currentTool)
				currentTool = nil
			}
			inToolDefs = false

		// ── Extensions ──
		case SET_META:
			if inst.Key != "media_type" {
				meta, _ := result["metadata"].(map[string]any)
				if meta == nil {
					meta = make(map[string]any)
				}
				meta[inst.Key] = inst.Str
				result["metadata"] = meta
			}

		case EXT_DATA:
			result[inst.Key] = json.RawMessage(inst.JSON)
		}
	}

	if messages != nil {
		result["messages"] = messages
	}
	if tools != nil {
		result["tools"] = tools
	}
	if len(stopSeqs) == 1 {
		result["stop"] = stopSeqs[0]
	} else if len(stopSeqs) > 1 {
		result["stop"] = stopSeqs
	}

	return json.Marshal(result)
}

// EmitResponse converts an AIL response program into OpenAI Chat Completions response JSON.
func (e *ChatCompletionsEmitter) EmitResponse(prog *Program) ([]byte, error) {
	result := map[string]any{
		"object": "chat.completion",
	}

	var choices []map[string]any
	var currentChoice map[string]any
	var currentMessage map[string]any
	var textContent string
	var toolCalls []map[string]any
	inMessage := false

	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_ID:
			result["id"] = inst.Str
		case RESP_MODEL:
			result["model"] = inst.Str
		case USAGE:
			result["usage"] = json.RawMessage(inst.JSON)

		case MSG_START:
			inMessage = true
			currentChoice = map[string]any{"index": len(choices)}
			currentMessage = make(map[string]any)
			textContent = ""
			toolCalls = nil

		case ROLE_AST:
			if inMessage {
				currentMessage["role"] = "assistant"
			}

		case TXT_CHUNK:
			if inMessage {
				textContent += inst.Str
			}

		case CALL_START:
			tc := map[string]any{
				"id":   inst.Str,
				"type": "function",
			}
			toolCalls = append(toolCalls, tc)

		case CALL_NAME:
			if len(toolCalls) > 0 {
				last := toolCalls[len(toolCalls)-1]
				fn, _ := last["function"].(map[string]any)
				if fn == nil {
					fn = make(map[string]any)
				}
				fn["name"] = inst.Str
				last["function"] = fn
			}

		case CALL_ARGS:
			if len(toolCalls) > 0 {
				last := toolCalls[len(toolCalls)-1]
				fn, _ := last["function"].(map[string]any)
				if fn == nil {
					fn = make(map[string]any)
				}
				fn["arguments"] = string(inst.JSON)
				last["function"] = fn
			}

		case CALL_END:
			// already tracked

		case RESP_DONE:
			if currentChoice != nil {
				currentChoice["finish_reason"] = inst.Str
			}

		case MSG_END:
			if inMessage && currentChoice != nil {
				if textContent != "" {
					currentMessage["content"] = textContent
				}
				if len(toolCalls) > 0 {
					currentMessage["tool_calls"] = toolCalls
				}
				currentChoice["message"] = currentMessage
				choices = append(choices, currentChoice)
				inMessage = false
			}
		}
	}

	if choices != nil {
		result["choices"] = choices
	}

	return json.Marshal(result)
}

// EmitStreamChunk converts an AIL stream chunk program into OpenAI Chat Completions streaming chunk JSON.
func (e *ChatCompletionsEmitter) EmitStreamChunk(prog *Program) ([]byte, error) {
	result := map[string]any{
		"object": "chat.completion.chunk",
	}

	var choices []map[string]any
	var delta map[string]any

	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_ID:
			result["id"] = inst.Str
		case RESP_MODEL:
			result["model"] = inst.Str
		case USAGE:
			result["usage"] = json.RawMessage(inst.JSON)

		case STREAM_START:
			delta = make(map[string]any)
			delta["role"] = "assistant"
			choices = append(choices, map[string]any{
				"index": 0,
				"delta": delta,
			})

		case STREAM_DELTA:
			if delta == nil {
				delta = make(map[string]any)
				choices = append(choices, map[string]any{
					"index": 0,
					"delta": delta,
				})
			}
			delta["content"] = inst.Str

		case STREAM_TOOL_DELTA:
			if delta == nil {
				delta = make(map[string]any)
				choices = append(choices, map[string]any{
					"index": 0,
					"delta": delta,
				})
			}
			var toolDelta map[string]any
			if err := json.Unmarshal(inst.JSON, &toolDelta); err == nil {
				// Reconstruct tool_calls array in delta
				tc := map[string]any{
					"index": toolDelta["index"],
					"type":  "function",
				}
				if id, ok := toolDelta["id"]; ok {
					tc["id"] = id
				}
				fn := make(map[string]any)
				if name, ok := toolDelta["name"]; ok {
					fn["name"] = name
				}
				if args, ok := toolDelta["arguments"]; ok {
					fn["arguments"] = args
				}
				if len(fn) > 0 {
					tc["function"] = fn
				}
				delta["tool_calls"] = []any{tc}
			}

		case RESP_DONE:
			choice := map[string]any{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": inst.Str,
			}
			choices = append(choices, choice)

		case STREAM_END:
			// end marker - no additional data
		}
	}

	if choices != nil {
		result["choices"] = choices
	} else {
		// Empty chunk with just metadata
		result["choices"] = []any{}
	}

	return json.Marshal(result)
}
