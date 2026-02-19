package ail

import (
	"encoding/json"
)

// ─── Google GenAI Emitter ────────────────────────────────────────────────────

// GoogleGenAIEmitter converts an AIL Program into Google GenAI JSON.
type GoogleGenAIEmitter struct{}

func (e *GoogleGenAIEmitter) EmitRequest(prog *Program) ([]byte, error) {
	result := make(map[string]any)
	ec := NewExtrasCollector()
	var contents []map[string]any
	var tools []map[string]any
	var systemParts []map[string]any

	genConfig := make(map[string]any)

	var currentRole string
	var parts []any
	inMessage := false
	var lastMediaType string

	// Tool definition state
	var funcDecls []map[string]any
	inToolDefs := false

	// Stop sequences
	var stopSeqs []string

	for _, inst := range prog.Code {
		switch inst.Op {
		// Config
		case SET_MODEL:
			result["model"] = inst.Str
		case SET_TEMP:
			genConfig["temperature"] = inst.Num
		case SET_TOPP:
			genConfig["topP"] = inst.Num
		case SET_MAX:
			genConfig["maxOutputTokens"] = inst.Int
		case SET_STOP:
			stopSeqs = append(stopSeqs, inst.Str)

		// Messages
		case MSG_START:
			ec.Push()
			inMessage = true
			currentRole = ""
			parts = nil

		case ROLE_SYS:
			currentRole = "system"
		case ROLE_USR:
			currentRole = "user"
		case ROLE_AST:
			currentRole = "model"
		case ROLE_TOOL:
			currentRole = "function"

		case TXT_CHUNK:
			if inMessage {
				parts = append(parts, map[string]any{"text": inst.Str})
			}

		case IMG_REF:
			if inMessage {
				data := ""
				if int(inst.Ref) < len(prog.Buffers) {
					data = string(prog.Buffers[inst.Ref])
				}
				mimeType := lastMediaType
				if mimeType == "" {
					mimeType = "image/png"
				}
				lastMediaType = ""
				parts = append(parts, map[string]any{
					"inlineData": map[string]any{
						"mimeType": mimeType,
						"data":     data,
					},
				})
			}

		case AUD_REF:
			if inMessage {
				data := ""
				if int(inst.Ref) < len(prog.Buffers) {
					data = string(prog.Buffers[inst.Ref])
				}
				mimeType := lastMediaType
				if mimeType == "" {
					mimeType = "audio/wav"
				}
				lastMediaType = ""
				parts = append(parts, map[string]any{
					"inlineData": map[string]any{
						"mimeType": mimeType,
						"data":     data,
					},
				})
			}

		case CALL_START:
			ec.Push()
			// Function call part (to be built up)
			parts = append(parts, map[string]any{
				"functionCall": map[string]any{},
			})

		case CALL_NAME:
			if len(parts) > 0 {
				last := parts[len(parts)-1].(map[string]any)
				if fc, ok := last["functionCall"].(map[string]any); ok {
					fc["name"] = inst.Str
				}
			}

		case CALL_ARGS:
			if len(parts) > 0 {
				last := parts[len(parts)-1].(map[string]any)
				if fc, ok := last["functionCall"].(map[string]any); ok {
					fc["args"] = json.RawMessage(inst.JSON)
				}
			}

		case CALL_END:
			if len(parts) > 0 {
				last := parts[len(parts)-1].(map[string]any)
				if fc, ok := last["functionCall"].(map[string]any); ok {
					ec.MergeInto(fc)
				}
			}
			ec.Pop()

		case RESULT_START:
			parts = append(parts, map[string]any{
				"functionResponse": map[string]any{
					"name": inst.Str,
				},
			})

		case RESULT_DATA:
			if len(parts) > 0 {
				last := parts[len(parts)-1].(map[string]any)
				if fr, ok := last["functionResponse"].(map[string]any); ok {
					fr["response"] = json.RawMessage(inst.Str)
				}
			}

		case MSG_END:
			if inMessage {
				if currentRole == "system" {
					// system_instruction in Google
					for _, p := range parts {
						if m, ok := p.(map[string]any); ok {
							systemParts = append(systemParts, m)
						}
					}
				} else if len(parts) > 0 {
					content := map[string]any{
						"role":  currentRole,
						"parts": parts,
					}
					ec.MergeInto(content)
					contents = append(contents, content)
				}
				inMessage = false
			}
			ec.Pop()

		// Tool definitions
		case DEF_START:
			ec.Push()
			inToolDefs = true
			funcDecls = nil

		case DEF_NAME:
			if inToolDefs {
				if len(funcDecls) > 0 {
					ec.MergeInto(funcDecls[len(funcDecls)-1])
				}
				funcDecls = append(funcDecls, map[string]any{
					"name": inst.Str,
				})
			}

		case DEF_DESC:
			if inToolDefs && len(funcDecls) > 0 {
				funcDecls[len(funcDecls)-1]["description"] = inst.Str
			}

		case DEF_SCHEMA:
			if inToolDefs && len(funcDecls) > 0 {
				funcDecls[len(funcDecls)-1]["parameters"] = json.RawMessage(inst.JSON)
			}

		case DEF_END:
			if inToolDefs && len(funcDecls) > 0 {
				ec.MergeInto(funcDecls[len(funcDecls)-1])
				tools = append(tools, map[string]any{
					"functionDeclarations": funcDecls,
				})
			}
			ec.Pop()
			inToolDefs = false

		case SET_META:
			if inst.Key == "media_type" {
				lastMediaType = inst.Str
			} else if ec.Depth() > 0 {
				ec.AddString(inst.Key, inst.Str)
			} else {
				result[inst.Key] = inst.Str
			}

		// Extensions
		case EXT_DATA:
			ec.AddJSON(inst.Key, inst.JSON)
		}
	}

	if len(systemParts) > 0 {
		result["system_instruction"] = map[string]any{"parts": systemParts}
	}
	if contents != nil {
		result["contents"] = contents
	}
	if tools != nil {
		result["tools"] = tools
	}
	if len(stopSeqs) > 0 {
		genConfig["stopSequences"] = stopSeqs
	}
	if len(genConfig) > 0 {
		result["generation_config"] = genConfig
	}

	ec.MergeInto(result)
	return json.Marshal(result)
}
