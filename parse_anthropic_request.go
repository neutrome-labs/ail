package ail

import (
	"encoding/json"
	"fmt"
)

// ─── Anthropic Messages Parser ───────────────────────────────────────────────

// AnthropicParser parses Anthropic Messages API JSON into AIL.
type AnthropicParser struct{}

func (p *AnthropicParser) ParseRequest(body []byte) (*Program, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("ail: parse anthropic request: %w", err)
	}

	prog := NewProgram()

	// Model
	if modelRaw, ok := raw["model"]; ok {
		var model string
		if json.Unmarshal(modelRaw, &model) == nil {
			prog.EmitString(SET_MODEL, model)
		}
		delete(raw, "model")
	}

	// Temperature
	if tempRaw, ok := raw["temperature"]; ok {
		var temp float64
		if json.Unmarshal(tempRaw, &temp) == nil {
			prog.EmitFloat(SET_TEMP, temp)
		}
		delete(raw, "temperature")
	}

	// top_p
	if tpRaw, ok := raw["top_p"]; ok {
		var tp float64
		if json.Unmarshal(tpRaw, &tp) == nil {
			prog.EmitFloat(SET_TOPP, tp)
		}
		delete(raw, "top_p")
	}

	// max_tokens (required in Anthropic)
	if maxRaw, ok := raw["max_tokens"]; ok {
		var max int32
		if json.Unmarshal(maxRaw, &max) == nil {
			prog.EmitInt(SET_MAX, max)
		}
		delete(raw, "max_tokens")
	}

	// stop_sequences
	if stopRaw, ok := raw["stop_sequences"]; ok {
		var stops []string
		if json.Unmarshal(stopRaw, &stops) == nil {
			for _, s := range stops {
				prog.EmitString(SET_STOP, s)
			}
		}
		delete(raw, "stop_sequences")
	}

	// Stream
	if streamRaw, ok := raw["stream"]; ok {
		var stream bool
		if json.Unmarshal(streamRaw, &stream) == nil && stream {
			prog.Emit(SET_STREAM)
		}
		delete(raw, "stream")
	}

	// Thinking configuration
	if thinkRaw, ok := raw["thinking"]; ok {
		prog.EmitJSON(SET_THINK, thinkRaw)
		delete(raw, "thinking")
	}

	// System (top-level in Anthropic, not in messages)
	if sysRaw, ok := raw["system"]; ok {
		var sysStr string
		if json.Unmarshal(sysRaw, &sysStr) == nil && sysStr != "" {
			prog.Emit(MSG_START)
			prog.Emit(ROLE_SYS)
			prog.EmitString(TXT_CHUNK, sysStr)
			prog.Emit(MSG_END)
		}
		delete(raw, "system")
	}

	// Tools
	if toolsRaw, ok := raw["tools"]; ok {
		var rawTools []json.RawMessage
		if json.Unmarshal(toolsRaw, &rawTools) == nil && len(rawTools) > 0 {
			prog.Emit(DEF_START)
			for _, rt := range rawTools {
				var toolMap map[string]json.RawMessage
				if json.Unmarshal(rt, &toolMap) != nil {
					continue
				}

				if nameRaw, ok := toolMap["name"]; ok {
					var name string
					if json.Unmarshal(nameRaw, &name) == nil {
						prog.EmitString(DEF_NAME, name)
					}
					delete(toolMap, "name")
				}
				if descRaw, ok := toolMap["description"]; ok {
					var desc string
					if json.Unmarshal(descRaw, &desc) == nil && desc != "" {
						prog.EmitString(DEF_DESC, desc)
					}
					delete(toolMap, "description")
				}
				if schemaRaw, ok := toolMap["input_schema"]; ok {
					prog.EmitJSON(DEF_SCHEMA, schemaRaw)
					delete(toolMap, "input_schema")
				}

				// Remaining fields as EXT_DATA (e.g., cache_control)
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
		if json.Unmarshal(msgsRaw, &rawMsgs) == nil {
			for _, rm := range rawMsgs {
				var msgMap map[string]json.RawMessage
				if json.Unmarshal(rm, &msgMap) != nil {
					continue
				}

				prog.Emit(MSG_START)

				if roleRaw, ok := msgMap["role"]; ok {
					var role string
					if json.Unmarshal(roleRaw, &role) == nil {
						switch role {
						case "user":
							prog.Emit(ROLE_USR)
						case "assistant":
							prog.Emit(ROLE_AST)
						}
					}
					delete(msgMap, "role")
				}

				// Content can be string or array of content blocks
				if contentRaw, ok := msgMap["content"]; ok {
					var contentStr string
					if json.Unmarshal(contentRaw, &contentStr) == nil {
						prog.EmitString(TXT_CHUNK, contentStr)
					} else {
						var rawBlocks []json.RawMessage
						if json.Unmarshal(contentRaw, &rawBlocks) == nil {
							for _, rb := range rawBlocks {
								var blockMap map[string]json.RawMessage
								if json.Unmarshal(rb, &blockMap) != nil {
									continue
								}

								var blockType string
								if typeRaw, ok := blockMap["type"]; ok {
									json.Unmarshal(typeRaw, &blockType)
								}

								switch blockType {
								case "text":
									var text string
									if textRaw, ok := blockMap["text"]; ok {
										json.Unmarshal(textRaw, &text)
									}
									prog.EmitString(TXT_CHUNK, text)
								case "thinking":
									prog.Emit(THINK_START)
									var thinking string
									if thinkRaw, ok := blockMap["thinking"]; ok {
										json.Unmarshal(thinkRaw, &thinking)
									}
									if thinking != "" {
										prog.EmitString(THINK_CHUNK, thinking)
									}
									// Preserve signature if present
									if sigRaw, ok := blockMap["signature"]; ok {
										var sig string
										if json.Unmarshal(sigRaw, &sig) == nil && sig != "" {
											ref := prog.AddBuffer([]byte(sig))
											prog.EmitRef(THINK_REF, ref)
										}
									}
									prog.Emit(THINK_END)
									continue // skip common tail
								case "image":
									if sourceRaw, ok := blockMap["source"]; ok {
										var source struct {
											Type      string `json:"type"`
											MediaType string `json:"media_type"`
											Data      string `json:"data"`
										}
										if json.Unmarshal(sourceRaw, &source) == nil {
											ref := prog.AddBuffer([]byte(source.Data))
											if source.MediaType != "" {
												prog.EmitKeyVal(SET_META, "media_type", source.MediaType)
											}
											prog.EmitRef(IMG_REF, ref)
										}
									}
									delete(blockMap, "source")
								case "tool_use":
									var id, name string
									if idRaw, ok := blockMap["id"]; ok {
										json.Unmarshal(idRaw, &id)
										delete(blockMap, "id")
									}
									if nameRaw, ok := blockMap["name"]; ok {
										json.Unmarshal(nameRaw, &name)
										delete(blockMap, "name")
									}
									prog.EmitString(CALL_START, id)
									prog.EmitString(CALL_NAME, name)
									if inputRaw, ok := blockMap["input"]; ok {
										if len(inputRaw) > 0 {
											prog.EmitJSON(CALL_ARGS, inputRaw)
										}
										delete(blockMap, "input")
									}
									// Remaining block-level fields as EXT_DATA
									delete(blockMap, "type")
									for key, val := range blockMap {
										prog.EmitKeyJSON(EXT_DATA, key, val)
									}
									prog.Emit(CALL_END)
									continue // skip common tail
								case "tool_result":
									var toolUseID string
									if tuiRaw, ok := blockMap["tool_use_id"]; ok {
										json.Unmarshal(tuiRaw, &toolUseID)
										delete(blockMap, "tool_use_id")
									}
									prog.EmitString(RESULT_START, toolUseID)
									if contentInner, ok := blockMap["content"]; ok {
										var resultStr string
										if json.Unmarshal(contentInner, &resultStr) == nil {
											prog.EmitString(RESULT_DATA, resultStr)
										}
										delete(blockMap, "content")
									}
									// Remaining block-level fields as EXT_DATA
									delete(blockMap, "type")
									for key, val := range blockMap {
										prog.EmitKeyJSON(EXT_DATA, key, val)
									}
									prog.Emit(RESULT_END)
									continue // skip common tail
								}
								// Common tail: passthrough remaining block-level fields
								delete(blockMap, "type")
								delete(blockMap, "text")
								for key, val := range blockMap {
									prog.EmitKeyJSON(EXT_DATA, key, val)
								}
							}
						}
					}
					delete(msgMap, "content")
				}

				// Remaining per-message fields as EXT_DATA
				for key, val := range msgMap {
					prog.EmitKeyJSON(EXT_DATA, key, val)
				}

				prog.Emit(MSG_END)
			}
		}
		delete(raw, "messages")
	}

	// Remaining fields as EXT_DATA
	for key, val := range raw {
		prog.EmitKeyJSON(EXT_DATA, key, val)
	}

	return prog, nil
}
