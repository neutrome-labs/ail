package ail

// ─── Parser interface ────────────────────────────────────────────────────────

// Parser converts a provider-specific JSON request into an AIL Program.
type Parser interface {
	// ParseRequest converts a raw JSON request body into an AIL program.
	ParseRequest(body []byte) (*Program, error)
}

// Emitter converts an AIL Program into a provider-specific JSON request.
type Emitter interface {
	// EmitRequest converts an AIL program into a raw JSON request body.
	EmitRequest(prog *Program) ([]byte, error)
}

// ResponseParser converts a provider-specific JSON response into an AIL Program.
type ResponseParser interface {
	// ParseResponse converts a raw JSON response body into an AIL program.
	ParseResponse(body []byte) (*Program, error)
}

// ResponseEmitter converts an AIL Program into a provider-specific JSON response.
type ResponseEmitter interface {
	// EmitResponse converts an AIL program into a raw JSON response body.
	EmitResponse(prog *Program) ([]byte, error)
}

// StreamChunkParser converts a provider-specific streaming chunk into AIL instructions.
type StreamChunkParser interface {
	// ParseStreamChunk converts a streaming chunk into an AIL program (partial).
	ParseStreamChunk(body []byte) (*Program, error)
}

// StreamChunkEmitter converts AIL stream instructions into provider-specific chunk JSON.
type StreamChunkEmitter interface {
	// EmitStreamChunk converts an AIL program (partial, from a stream chunk) into JSON.
	EmitStreamChunk(prog *Program) ([]byte, error)
}
