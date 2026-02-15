package ail

import (
	"encoding/json"
)

// ─── Anthropic Messages Emitter ──────────────────────────────────────────────

// AnthropicEmitter converts an AIL Program into Anthropic Messages API JSON.
type AnthropicEmitter struct{}

func (e *AnthropicEmitter) EmitRequest(prog *Program) ([]byte, error) {
	result := make(map[string]any)
	var messages []map[string]any
	var tools []map[string]any
	var systemText string

	var currentRole string
	var contentBlocks []any
	var simpleText string
	inMessage := false
	needsToolResultWrap := false
	var currentToolCallID string
	var lastMediaType string

	// Tool definition state
	var currentTool map[string]any
	inToolDefs := false

	// Stop sequences
	var stopSeqs []string

	for _, inst := range prog.Code {
		switch inst.Op {
		// Config
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

		// Messages
		case MSG_START:
			inMessage = true
			currentRole = ""
			contentBlocks = nil
			simpleText = ""
			needsToolResultWrap = false
			currentToolCallID = ""

		case ROLE_SYS:
			currentRole = "system"
		case ROLE_USR:
			currentRole = "user"
		case ROLE_AST:
			currentRole = "assistant"
		case ROLE_TOOL:
			// Anthropic: tool results go in a "user" message with tool_result content blocks
			currentRole = "user"
			needsToolResultWrap = true

		case TXT_CHUNK:
			if inMessage {
				simpleText += inst.Str
			}

		case IMG_REF:
			if inMessage {
				data := ""
				if int(inst.Ref) < len(prog.Buffers) {
					data = string(prog.Buffers[inst.Ref])
				}
				// Flush text first
				if simpleText != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": simpleText,
					})
					simpleText = ""
				}
				mediaType := lastMediaType
				if mediaType == "" {
					mediaType = "image/png"
				}
				lastMediaType = ""
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "image",
					"source": map[string]any{
						"type":       "base64",
						"media_type": mediaType,
						"data":       data,
					},
				})
			}

		case CALL_START:
			if inMessage {
				// Flush text
				if simpleText != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": simpleText,
					})
					simpleText = ""
				}
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "tool_use",
					"id":   inst.Str,
				})
			}

		case CALL_NAME:
			if len(contentBlocks) > 0 {
				last := contentBlocks[len(contentBlocks)-1].(map[string]any)
				if last["type"] == "tool_use" {
					last["name"] = inst.Str
				}
			}

		case CALL_ARGS:
			if len(contentBlocks) > 0 {
				last := contentBlocks[len(contentBlocks)-1].(map[string]any)
				if last["type"] == "tool_use" {
					last["input"] = json.RawMessage(inst.JSON)
				}
			}

		case RESULT_START:
			currentToolCallID = inst.Str

		case RESULT_DATA:
			if needsToolResultWrap {
				// Flush text
				if simpleText != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": simpleText,
					})
					simpleText = ""
				}
				contentBlocks = append(contentBlocks, map[string]any{
					"type":        "tool_result",
					"tool_use_id": currentToolCallID,
					"content":     inst.Str,
				})
			} else {
				simpleText += inst.Str
			}

		case RESULT_END:
			// tracked via needsToolResultWrap

		case MSG_END:
			if inMessage {
				if currentRole == "system" {
					// Anthropic: system is top-level, not in messages
					systemText += simpleText
				} else {
					msg := map[string]any{"role": currentRole}
					if len(contentBlocks) > 0 {
						// Flush remaining text
						if simpleText != "" {
							contentBlocks = append(contentBlocks, map[string]any{
								"type": "text",
								"text": simpleText,
							})
						}
						msg["content"] = contentBlocks
					} else if simpleText != "" {
						msg["content"] = simpleText
					}
					messages = append(messages, msg)
				}
				inMessage = false
			}

		// Tool definitions
		case DEF_START:
			inToolDefs = true
			currentTool = nil

		case DEF_NAME:
			if inToolDefs {
				if currentTool != nil {
					tools = append(tools, currentTool)
				}
				currentTool = map[string]any{"name": inst.Str}
			}

		case DEF_DESC:
			if currentTool != nil {
				currentTool["description"] = inst.Str
			}

		case DEF_SCHEMA:
			if currentTool != nil {
				currentTool["input_schema"] = json.RawMessage(inst.JSON)
			}

		case DEF_END:
			if inToolDefs && currentTool != nil {
				tools = append(tools, currentTool)
				currentTool = nil
			}
			inToolDefs = false

		// Extensions
		case SET_META:
			if inst.Key == "media_type" {
				lastMediaType = inst.Str
			} else {
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

	if systemText != "" {
		result["system"] = systemText
	}
	if messages != nil {
		result["messages"] = messages
	}
	if tools != nil {
		result["tools"] = tools
	}
	if len(stopSeqs) > 0 {
		result["stop_sequences"] = stopSeqs
	}

	return json.Marshal(result)
}

// EmitResponse converts an AIL response program into Anthropic Messages API response JSON.
func (e *AnthropicEmitter) EmitResponse(prog *Program) ([]byte, error) {
	result := map[string]any{
		"type": "message",
		"role": "assistant",
	}

	var contentBlocks []any
	var textContent string
	inMessage := false

	for _, inst := range prog.Code {
		switch inst.Op {
		case RESP_ID:
			result["id"] = inst.Str
		case RESP_MODEL:
			result["model"] = inst.Str
		case USAGE:
			// Convert standard usage to Anthropic format
			var usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}
			if json.Unmarshal(inst.JSON, &usage) == nil {
				result["usage"] = map[string]int{
					"input_tokens":  usage.PromptTokens,
					"output_tokens": usage.CompletionTokens,
				}
			}

		case MSG_START:
			inMessage = true
			contentBlocks = nil
			textContent = ""

		case TXT_CHUNK:
			if inMessage {
				textContent += inst.Str
			}

		case CALL_START:
			if inMessage {
				if textContent != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": textContent,
					})
					textContent = ""
				}
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "tool_use",
					"id":   inst.Str,
				})
			}

		case CALL_NAME:
			if len(contentBlocks) > 0 {
				last := contentBlocks[len(contentBlocks)-1].(map[string]any)
				if last["type"] == "tool_use" {
					last["name"] = inst.Str
				}
			}

		case CALL_ARGS:
			if len(contentBlocks) > 0 {
				last := contentBlocks[len(contentBlocks)-1].(map[string]any)
				if last["type"] == "tool_use" {
					last["input"] = json.RawMessage(inst.JSON)
				}
			}

		case RESP_DONE:
			switch inst.Str {
			case "stop":
				result["stop_reason"] = "end_turn"
			case "tool_calls":
				result["stop_reason"] = "tool_use"
			case "length":
				result["stop_reason"] = "max_tokens"
			default:
				result["stop_reason"] = inst.Str
			}

		case MSG_END:
			if inMessage {
				if textContent != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": textContent,
					})
				}
				inMessage = false
			}
		}
	}

	if len(contentBlocks) > 0 {
		result["content"] = contentBlocks
	} else {
		result["content"] = []any{}
	}

	return json.Marshal(result)
}

// EmitStreamChunk converts an AIL stream chunk into Anthropic SSE event JSON.
func (e *AnthropicEmitter) EmitStreamChunk(prog *Program) ([]byte, error) {
	// Anthropic streaming uses typed events; emit the appropriate type
	for _, inst := range prog.Code {
		switch inst.Op {
		case STREAM_START:
			event := map[string]any{"type": "message_start"}
			msgObj := map[string]any{"role": "assistant"}
			// Look ahead for RESP_ID and RESP_MODEL in same chunk
			for _, ahead := range prog.Code {
				if ahead.Op == RESP_ID {
					msgObj["id"] = ahead.Str
				}
				if ahead.Op == RESP_MODEL {
					msgObj["model"] = ahead.Str
				}
			}
			event["message"] = msgObj
			return json.Marshal(event)

		case STREAM_DELTA:
			event := map[string]any{
				"type": "content_block_delta",
				"delta": map[string]any{
					"type": "text_delta",
					"text": inst.Str,
				},
			}
			return json.Marshal(event)

		case STREAM_TOOL_DELTA:
			var td map[string]any
			if json.Unmarshal(inst.JSON, &td) == nil {
				if _, hasName := td["name"]; hasName {
					// Tool start
					event := map[string]any{
						"type":  "content_block_start",
						"index": td["index"],
						"content_block": map[string]any{
							"type": "tool_use",
							"id":   td["id"],
							"name": td["name"],
						},
					}
					return json.Marshal(event)
				}
				if args, ok := td["arguments"]; ok {
					event := map[string]any{
						"type":  "content_block_delta",
						"index": td["index"],
						"delta": map[string]any{
							"type":         "input_json_delta",
							"partial_json": args,
						},
					}
					return json.Marshal(event)
				}
			}

		case RESP_DONE:
			stopReason := "end_turn"
			switch inst.Str {
			case "stop":
				stopReason = "end_turn"
			case "tool_calls":
				stopReason = "tool_use"
			case "length":
				stopReason = "max_tokens"
			default:
				stopReason = inst.Str
			}
			event := map[string]any{
				"type":  "message_delta",
				"delta": map[string]any{"stop_reason": stopReason},
			}
			return json.Marshal(event)

		case STREAM_END:
			return json.Marshal(map[string]any{"type": "message_stop"})
		}
	}

	// Empty chunk
	return json.Marshal(map[string]any{"type": "ping"})
}
