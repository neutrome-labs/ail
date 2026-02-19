package ail

import (
	"encoding/json"
)

// ─── OpenAI Responses API Emitter ────────────────────────────────────────────

// ResponsesEmitter converts an AIL Program into OpenAI Responses API JSON.
type ResponsesEmitter struct{}

func (e *ResponsesEmitter) EmitRequest(prog *Program) ([]byte, error) {
	result := make(map[string]any)
	ec := NewExtrasCollector()
	var input []map[string]any
	var tools []map[string]any
	var systemText string

	var currentMsg map[string]any
	var currentRole string
	var textContent string

	// Tool definition state
	var currentTool map[string]any
	inToolDefs := false

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
			result["max_output_tokens"] = inst.Int
		case SET_STREAM:
			result["stream"] = true

		// Messages
		case MSG_START:
			ec.Push()
			currentMsg = make(map[string]any)
			currentRole = ""
			textContent = ""

		case ROLE_SYS:
			currentRole = "system"
		case ROLE_USR:
			currentRole = "user"
		case ROLE_AST:
			currentRole = "assistant"
		case ROLE_TOOL:
			currentRole = "tool"

		case TXT_CHUNK:
			textContent += inst.Str

		case MSG_END:
			if currentMsg != nil {
				if currentRole == "system" {
					// Responses API: system goes to "instructions"
					if systemText != "" && textContent != "" {
						systemText += "\n\n"
					}
					systemText += textContent
				} else {
					currentMsg["role"] = currentRole
					if textContent != "" {
						currentMsg["content"] = textContent
					}
					ec.MergeInto(currentMsg)
					input = append(input, currentMsg)
				}
				currentMsg = nil
			}
			ec.Pop()

		// Tool definitions (Responses API: flat structure)
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
				currentTool = map[string]any{
					"type": "function",
					"name": inst.Str,
				}
			}

		case DEF_DESC:
			if currentTool != nil {
				currentTool["description"] = inst.Str
			}

		case DEF_SCHEMA:
			if currentTool != nil {
				currentTool["parameters"] = json.RawMessage(inst.JSON)
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
				// consumed by IMG_REF / AUD_REF
			} else if ec.Depth() > 0 {
				ec.AddString(inst.Key, inst.Str)
			} else {
				result[inst.Key] = inst.Str
			}

		case EXT_DATA:
			ec.AddJSON(inst.Key, inst.JSON)
		}
	}

	if systemText != "" {
		result["instructions"] = systemText
	}
	if input != nil {
		result["input"] = input
	}
	if tools != nil {
		result["tools"] = tools
	}

	ec.MergeInto(result)
	return json.Marshal(result)
}
