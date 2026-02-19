package ail

import "encoding/json"

// ExtrasCollector is a stack-based tracker for EXT_DATA and SET_META
// instructions at various nesting levels. Emitters push on block-open
// opcodes (MSG_START, DEF_START, CALL_START, …) and pop on the matching
// close. Before popping, call MergeInto to attach the collected extras
// to the block's output object.
//
// Level 0 represents the top level (outside any block). After the main
// instruction loop, call MergeInto(result) to capture those extras.
//
// For DEF blocks—where individual tools are separated by DEF_NAME rather
// than nested START/END pairs—call MergeInto on the current tool at each
// DEF_NAME boundary (and at DEF_END). MergeInto resets the current level,
// so the next tool starts with a clean slate.
type ExtrasCollector struct {
	levels []map[string]any
}

// NewExtrasCollector creates a collector with one level (top-level).
func NewExtrasCollector() *ExtrasCollector {
	return &ExtrasCollector{
		levels: []map[string]any{{}},
	}
}

// Push starts collecting extras for a new nested block.
func (ec *ExtrasCollector) Push() {
	ec.levels = append(ec.levels, map[string]any{})
}

// Pop discards the current level. Call MergeInto first to capture extras.
func (ec *ExtrasCollector) Pop() {
	if len(ec.levels) > 1 {
		ec.levels = ec.levels[:len(ec.levels)-1]
	}
}

// AddJSON stores a JSON value at the current level (for EXT_DATA).
func (ec *ExtrasCollector) AddJSON(key string, val json.RawMessage) {
	cp := make(json.RawMessage, len(val))
	copy(cp, val)
	ec.levels[len(ec.levels)-1][key] = cp
}

// AddString stores a string value at the current level (for SET_META).
func (ec *ExtrasCollector) AddString(key, val string) {
	ec.levels[len(ec.levels)-1][key] = val
}

// MergeInto copies all extras at the current level into obj and then
// resets the current level. The reset allows collecting for the next
// item at the same nesting depth (e.g., the next tool after DEF_NAME).
func (ec *ExtrasCollector) MergeInto(obj map[string]any) {
	cur := ec.levels[len(ec.levels)-1]
	for k, v := range cur {
		obj[k] = v
	}
	ec.levels[len(ec.levels)-1] = map[string]any{}
}

// Depth returns the current nesting depth (0 = top level).
func (ec *ExtrasCollector) Depth() int {
	return len(ec.levels) - 1
}
