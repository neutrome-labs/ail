package ail

import (
	"encoding/json"
	"fmt"
)

func (p *ChatCompletionsParser) ParseResponse(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse chat completions response: %w", err)
	}

	prog := NewProgram()

	// Response ID
	if idRaw, ok := raw["id"]; ok {
		var id string
		if json.Unmarshal(idRaw, &id) == nil {
			prog.EmitString(RESP_ID, id)
		}
		delete(raw, "id")
	}

	// Model
	if modelRaw, ok := raw["model"]; ok {
		var model string
		if json.Unmarshal(modelRaw, &model) == nil {
			prog.EmitString(RESP_MODEL, model)
		}
		delete(raw, "model")
	}

	// Usage
	if usageRaw, ok := raw["usage"]; ok {
		prog.EmitJSON(USAGE, usageRaw)
		delete(raw, "usage")
	}

	// Choices
	if choicesRaw, ok := raw["choices"]; ok {
		var rawChoices []json.RawMessage
		if json.Unmarshal(choicesRaw, &rawChoices) == nil {
			for _, rc := range rawChoices {
				var choiceMap map[string]json.RawMessage
				if json.Unmarshal(rc, &choiceMap) != nil {
					continue
				}

				prog.Emit(MSG_START)

				if msgRaw, ok := choiceMap["message"]; ok {
					var msg struct {
						Role             string          `json:"role"`
						Content          json.RawMessage `json:"content"`
						ReasoningContent string          `json:"reasoning_content,omitempty"`
						ToolCalls        []struct {
							ID       string `json:"id"`
							Type     string `json:"type"`
							Function *struct {
								Name      string `json:"name"`
								Arguments string `json:"arguments"`
							} `json:"function"`
						} `json:"tool_calls,omitempty"`
					}
					if json.Unmarshal(msgRaw, &msg) == nil {
						switch msg.Role {
						case "assistant":
							prog.Emit(ROLE_AST)
						}

						// Reasoning content (before main content)
						if msg.ReasoningContent != "" {
							prog.Emit(THINK_START)
							prog.EmitString(THINK_CHUNK, msg.ReasoningContent)
							prog.Emit(THINK_END)
						}

						if msg.Content != nil {
							var contentStr string
							if json.Unmarshal(msg.Content, &contentStr) == nil && contentStr != "" {
								prog.EmitString(TXT_CHUNK, contentStr)
							}
						}

						for _, tc := range msg.ToolCalls {
							prog.EmitString(CALL_START, tc.ID)
							if tc.Function != nil {
								prog.EmitString(CALL_NAME, tc.Function.Name)
								if tc.Function.Arguments != "" {
									prog.EmitJSON(CALL_ARGS, json.RawMessage(tc.Function.Arguments))
								}
							}
							prog.Emit(CALL_END)
						}
					}
					delete(choiceMap, "message")
				}

				if frRaw, ok := choiceMap["finish_reason"]; ok {
					var fr string
					if json.Unmarshal(frRaw, &fr) == nil && fr != "" {
						prog.EmitString(RESP_DONE, fr)
					}
					delete(choiceMap, "finish_reason")
				}

				// Remove index (reconstructed by emitter from ordering)
				delete(choiceMap, "index")

				// Passthrough remaining choice-level fields as EXT_DATA
				// inside the MSG block (e.g. logprobs, content_filter_results).
				for key, val := range choiceMap {
					prog.EmitKeyJSON(EXT_DATA, key, val)
				}

				prog.Emit(MSG_END)
			}
		}
	}
	delete(raw, "choices")

	// Passthrough remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}
	return prog, nil
}
