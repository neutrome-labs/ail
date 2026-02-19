package ail

import (
	"encoding/json"
	"fmt"
)

// ─── OpenAI Chat Completions Parser ──────────────────────────────────────────

// ChatCompletionsParser parses OpenAI Chat Completions JSON into AIL.
type ChatCompletionsParser struct{}

func (p *ChatCompletionsParser) ParseRequest(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse chat completions request: %w", err)
	}

	prog := NewProgram()

	// Config: model
	if modelRaw, ok := raw["model"]; ok {
		var model string
		if err := json.Unmarshal(modelRaw, &model); err == nil {
			prog.EmitString(SET_MODEL, model)
		}
		delete(raw, "model")
	}

	// Config: temperature
	if tempRaw, ok := raw["temperature"]; ok {
		var temp float64
		if err := json.Unmarshal(tempRaw, &temp); err == nil {
			prog.EmitFloat(SET_TEMP, temp)
		}
		delete(raw, "temperature")
	}

	// Config: top_p
	if tpRaw, ok := raw["top_p"]; ok {
		var tp float64
		if err := json.Unmarshal(tpRaw, &tp); err == nil {
			prog.EmitFloat(SET_TOPP, tp)
		}
		delete(raw, "top_p")
	}

	// Config: max_tokens / max_completion_tokens
	if mtRaw, ok := raw["max_tokens"]; ok {
		var mt int32
		if err := json.Unmarshal(mtRaw, &mt); err == nil {
			prog.EmitInt(SET_MAX, mt)
		}
		delete(raw, "max_tokens")
	} else if mctRaw, ok := raw["max_completion_tokens"]; ok {
		var mct int32
		if err := json.Unmarshal(mctRaw, &mct); err == nil {
			prog.EmitInt(SET_MAX, mct)
		}
		delete(raw, "max_completion_tokens")
	}

	// Config: stop
	if stopRaw, ok := raw["stop"]; ok {
		// stop can be string or []string
		var stopStr string
		if err := json.Unmarshal(stopRaw, &stopStr); err == nil {
			prog.EmitString(SET_STOP, stopStr)
		} else {
			var stopArr []string
			if err := json.Unmarshal(stopRaw, &stopArr); err == nil {
				for _, s := range stopArr {
					prog.EmitString(SET_STOP, s)
				}
			}
		}
		delete(raw, "stop")
	}

	// Config: stream
	if streamRaw, ok := raw["stream"]; ok {
		var stream bool
		if err := json.Unmarshal(streamRaw, &stream); err == nil && stream {
			prog.Emit(SET_STREAM)
		}
		delete(raw, "stream")
	}

	// Reasoning effort → thinking config
	if effortRaw, ok := raw["reasoning_effort"]; ok {
		var effort string
		if json.Unmarshal(effortRaw, &effort) == nil && effort != "" {
			cfg, _ := json.Marshal(map[string]string{"effort": effort})
			prog.EmitJSON(SET_THINK, cfg)
		}
		delete(raw, "reasoning_effort")
	}

	// Tool definitions
	if toolsRaw, ok := raw["tools"]; ok {
		var rawTools []json.RawMessage
		if json.Unmarshal(toolsRaw, &rawTools) == nil && len(rawTools) > 0 {
			prog.Emit(DEF_START)
			for _, rt := range rawTools {
				var toolMap map[string]json.RawMessage
				if json.Unmarshal(rt, &toolMap) != nil {
					continue
				}
				funcRaw, ok := toolMap["function"]
				if !ok {
					continue
				}
				delete(toolMap, "function")
				delete(toolMap, "type") // always "function", reconstructed by emitter

				var funcMap map[string]json.RawMessage
				if json.Unmarshal(funcRaw, &funcMap) != nil {
					continue
				}

				if nameRaw, ok := funcMap["name"]; ok {
					var name string
					if json.Unmarshal(nameRaw, &name) == nil {
						prog.EmitString(DEF_NAME, name)
					}
					delete(funcMap, "name")
				}
				if descRaw, ok := funcMap["description"]; ok {
					var desc string
					if json.Unmarshal(descRaw, &desc) == nil && desc != "" {
						prog.EmitString(DEF_DESC, desc)
					}
					delete(funcMap, "description")
				}
				if paramsRaw, ok := funcMap["parameters"]; ok {
					prog.EmitJSON(DEF_SCHEMA, paramsRaw)
					delete(funcMap, "parameters")
				}

				// Remaining function-level fields as EXT_DATA (e.g., strict)
				for key, val := range funcMap {
					prog.EmitKeyJSON(EXT_DATA, key, val)
				}
				// Remaining outer tool-level fields as EXT_DATA
				for key, val := range toolMap {
					prog.EmitKeyJSON(EXT_DATA, key, val)
				}
			}
			prog.Emit(DEF_END)
		}
		delete(raw, "tools")
	}

	// Messages
	if msgsRaw, ok := raw["messages"]; ok {
		var rawMsgs []json.RawMessage
		if err := json.Unmarshal(msgsRaw, &rawMsgs); err != nil {
			return nil, fmt.Errorf("ail: parse messages: %w", err)
		}

		for _, rm := range rawMsgs {
			var msgMap map[string]json.RawMessage
			if json.Unmarshal(rm, &msgMap) != nil {
				continue
			}

			prog.Emit(MSG_START)

			// Role
			var role string
			if roleRaw, ok := msgMap["role"]; ok {
				json.Unmarshal(roleRaw, &role)
				delete(msgMap, "role")
			}
			switch role {
			case "system", "developer":
				prog.Emit(ROLE_SYS)
			case "user":
				prog.Emit(ROLE_USR)
			case "assistant":
				prog.Emit(ROLE_AST)
			case "tool":
				prog.Emit(ROLE_TOOL)
				// Tool result
				if tcidRaw, ok := msgMap["tool_call_id"]; ok {
					var tcid string
					if json.Unmarshal(tcidRaw, &tcid) == nil && tcid != "" {
						prog.EmitString(RESULT_START, tcid)
					}
					delete(msgMap, "tool_call_id")
				}
			}

			// Content: can be string or array of content parts
			if contentRaw, ok := msgMap["content"]; ok {
				var contentStr string
				if json.Unmarshal(contentRaw, &contentStr) == nil {
					// Simple string content
					if role == "tool" {
						prog.EmitString(RESULT_DATA, contentStr)
					} else {
						prog.EmitString(TXT_CHUNK, contentStr)
					}
				} else {
					// Array of content parts
					var rawParts []json.RawMessage
					if json.Unmarshal(contentRaw, &rawParts) == nil {
						for _, rp := range rawParts {
							var partMap map[string]json.RawMessage
							if json.Unmarshal(rp, &partMap) != nil {
								continue
							}
							var partType string
							if ptRaw, ok := partMap["type"]; ok {
								json.Unmarshal(ptRaw, &partType)
							}
							switch partType {
							case "text":
								var text string
								if textRaw, ok := partMap["text"]; ok {
									json.Unmarshal(textRaw, &text)
								}
								prog.EmitString(TXT_CHUNK, text)
							case "image_url":
								if iuRaw, ok := partMap["image_url"]; ok {
									var iu struct {
										URL    string `json:"url"`
										Detail string `json:"detail,omitempty"`
									}
									if json.Unmarshal(iuRaw, &iu) == nil {
										ref := prog.AddBuffer([]byte(iu.URL))
										prog.EmitRef(IMG_REF, ref)
									}
								}
							case "input_audio":
								if iaRaw, ok := partMap["input_audio"]; ok {
									var ia struct {
										Data   string `json:"data"`
										Format string `json:"format"`
									}
									if json.Unmarshal(iaRaw, &ia) == nil {
										ref := prog.AddBuffer([]byte(ia.Data))
										if ia.Format != "" {
											prog.EmitKeyVal(SET_META, "media_type", "audio/"+ia.Format)
										}
										prog.EmitRef(AUD_REF, ref)
									}
								}
							}
						}
					}
				}
				delete(msgMap, "content")
			}

			// Reasoning content (open models / DeepSeek / QwQ)
			if rcRaw, ok := msgMap["reasoning_content"]; ok {
				var rc string
				if json.Unmarshal(rcRaw, &rc) == nil && rc != "" {
					prog.Emit(THINK_START)
					prog.EmitString(THINK_CHUNK, rc)
					prog.Emit(THINK_END)
				}
				delete(msgMap, "reasoning_content")
			}

			// Tool calls in assistant messages
			if tcRaw, ok := msgMap["tool_calls"]; ok {
				var toolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function *struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				}
				if json.Unmarshal(tcRaw, &toolCalls) == nil {
					for _, tc := range toolCalls {
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
				delete(msgMap, "tool_calls")
			}

			// Close tool result
			if role == "tool" {
				prog.Emit(RESULT_END)
			}

			// Remaining per-message fields as EXT_DATA (e.g., name, refusal)
			for key, val := range msgMap {
				prog.EmitKeyJSON(EXT_DATA, key, val)
			}

			prog.Emit(MSG_END)
		}
		delete(raw, "messages")
	}

	// Passthrough remaining fields as EXT_DATA
	delete(raw, "stream_options") // handled implicitly by SET_STREAM
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}
