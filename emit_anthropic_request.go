package ail

import (
	"encoding/json"
)

// ─── Anthropic Messages Emitter ──────────────────────────────────────────────

// AnthropicEmitter converts an AIL Program into Anthropic Messages API JSON.
type AnthropicEmitter struct{}

func (e *AnthropicEmitter) EmitRequest(prog *Program) ([]byte, error) {
	result := make(map[string]any)
	ec := NewExtrasCollector()
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
			ec.Push()
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
				ec.Push()
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

		case CALL_END:
			if len(contentBlocks) > 0 {
				last := contentBlocks[len(contentBlocks)-1].(map[string]any)
				if last["type"] == "tool_use" {
					ec.MergeInto(last)
				}
			}
			ec.Pop()

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
					if systemText != "" && simpleText != "" {
						systemText += "\n\n"
					}
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
					ec.MergeInto(msg)
					messages = append(messages, msg)
				}
				inMessage = false
			}
			ec.Pop()

		// Tool definitions
		case DEF_START:
			ec.Push()
			inToolDefs = true
			currentTool = nil

		case DEF_NAME:
			if inToolDefs {
				if currentTool != nil {
					ec.MergeInto(currentTool)
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
				ec.MergeInto(currentTool)
				tools = append(tools, currentTool)
				currentTool = nil
			}
			ec.Pop()
			inToolDefs = false

		// Extensions
		case SET_META:
			if inst.Key == "media_type" {
				lastMediaType = inst.Str
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

	ec.MergeInto(result)
	return json.Marshal(result)
}
