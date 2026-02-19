package ail

import (
	"encoding/json"
)

// ─── OpenAI Chat Completions Emitter ─────────────────────────────────────────

// ChatCompletionsEmitter converts an AIL Program into OpenAI Chat Completions JSON.
type ChatCompletionsEmitter struct{}

func (e *ChatCompletionsEmitter) EmitRequest(prog *Program) ([]byte, error) {
	result := make(map[string]any)
	ec := NewExtrasCollector()
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
			ec.Push()
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
			ec.Push()
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
			if len(toolCalls) > 0 {
				ec.MergeInto(toolCalls[len(toolCalls)-1])
			}
			ec.Pop()

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

				ec.MergeInto(currentMsg)
				messages = append(messages, currentMsg)
				currentMsg = nil
			}
			ec.Pop()

		// ── Tool Definitions ──
		case DEF_START:
			ec.Push()
			inToolDefs = true
			currentTool = nil

		case DEF_NAME:
			if inToolDefs {
				if currentTool != nil {
					fn := currentTool["function"].(map[string]any)
					ec.MergeInto(fn)
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
				fn := currentTool["function"].(map[string]any)
				ec.MergeInto(fn)
				tools = append(tools, currentTool)
				currentTool = nil
			}
			ec.Pop()
			inToolDefs = false

		// ── Extensions ──
		case SET_META:
			if inst.Key == "media_type" {
				// consumed by IMG_REF / AUD_REF, not collected
			} else if ec.Depth() > 0 {
				ec.AddString(inst.Key, inst.Str)
			} else {
				meta, _ := result["metadata"].(map[string]any)
				if meta == nil {
					meta = make(map[string]any)
				}
				meta[inst.Key] = inst.Str
				result["metadata"] = meta
			}

		case EXT_DATA:
			ec.AddJSON(inst.Key, inst.JSON)
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

	ec.MergeInto(result)
	return json.Marshal(result)
}
