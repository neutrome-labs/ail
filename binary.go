package ail

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
)

// Binary format constants.
var binaryMagic = [4]byte{'A', 'I', 'L', 0x00}

const binaryVersion uint8 = 1

// ─── Binary Encoder ──────────────────────────────────────────────────────────

// Encode writes the program to w in AIL binary format.
//
// Wire layout:
//
//	[magic 4B][version 1B][bufCount uint32][buf0Len uint32][buf0 data]…[instructions…]
func (p *Program) Encode(w io.Writer) error {
	// Header
	if _, err := w.Write(binaryMagic[:]); err != nil {
		return fmt.Errorf("ail.Encode: write magic: %w", err)
	}
	if _, err := w.Write([]byte{binaryVersion}); err != nil {
		return fmt.Errorf("ail.Encode: write version: %w", err)
	}

	// Buffers
	if err := writeUint32(w, uint32(len(p.Buffers))); err != nil {
		return fmt.Errorf("ail.Encode: write buffer count: %w", err)
	}
	for i, buf := range p.Buffers {
		if err := writeBytes(w, buf); err != nil {
			return fmt.Errorf("ail.Encode: write buffer %d: %w", i, err)
		}
	}

	// Instructions
	for _, inst := range p.Code {
		if _, err := w.Write([]byte{byte(inst.Op)}); err != nil {
			return err
		}
		switch inst.Op {
		// No-arg opcodes
		case MSG_START, MSG_END, ROLE_SYS, ROLE_USR, ROLE_AST, ROLE_TOOL,
			DEF_START, DEF_END, CALL_END, RESULT_END,
			SET_STREAM, STREAM_START, STREAM_END:
			// nothing extra

		// String arg
		case TXT_CHUNK, DEF_NAME, DEF_DESC, CALL_START, CALL_NAME,
			RESULT_START, RESULT_DATA, RESP_ID, RESP_MODEL, RESP_DONE,
			SET_MODEL, SET_STOP, STREAM_DELTA, COMMENT:
			if err := writeString(w, inst.Str); err != nil {
				return err
			}

		// Float arg
		case SET_TEMP, SET_TOPP:
			if err := writeFloat64(w, inst.Num); err != nil {
				return err
			}

		// Int arg
		case SET_MAX:
			if err := writeInt32(w, inst.Int); err != nil {
				return err
			}

		// JSON arg
		case DEF_SCHEMA, CALL_ARGS, USAGE, STREAM_TOOL_DELTA:
			if err := writeBytes(w, inst.JSON); err != nil {
				return err
			}

		// RefID arg
		case IMG_REF, AUD_REF, TXT_REF:
			if err := writeUint32(w, inst.Ref); err != nil {
				return err
			}

		// Key + Val (two strings)
		case SET_META:
			if err := writeString(w, inst.Key); err != nil {
				return err
			}
			if err := writeString(w, inst.Str); err != nil {
				return err
			}

		// Key + JSON
		case EXT_DATA:
			if err := writeString(w, inst.Key); err != nil {
				return err
			}
			if err := writeBytes(w, inst.JSON); err != nil {
				return err
			}

		default:
			return fmt.Errorf("ail.Encode: unknown opcode 0x%02X", inst.Op)
		}
	}
	return nil
}

// ─── Binary Decoder ──────────────────────────────────────────────────────────

// Decode reads an AIL binary program from r.
func Decode(r io.Reader) (*Program, error) {
	// Header
	var header [5]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, fmt.Errorf("ail.Decode: read header: %w", err)
	}
	if header[0] != binaryMagic[0] || header[1] != binaryMagic[1] ||
		header[2] != binaryMagic[2] || header[3] != binaryMagic[3] {
		return nil, fmt.Errorf("ail.Decode: invalid magic bytes %q", header[:4])
	}
	if header[4] != binaryVersion {
		return nil, fmt.Errorf("ail.Decode: unsupported version %d (want %d)", header[4], binaryVersion)
	}

	// Buffers
	bufCount, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("ail.Decode: read buffer count: %w", err)
	}
	p := NewProgram()
	for i := uint32(0); i < bufCount; i++ {
		buf, err := readBytes(r)
		if err != nil {
			return nil, fmt.Errorf("ail.Decode: read buffer %d: %w", i, err)
		}
		p.Buffers = append(p.Buffers, buf)
	}

	// Instructions
	opBuf := make([]byte, 1)

	for {
		_, err := io.ReadFull(r, opBuf)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("ail.Decode: read opcode: %w", err)
		}

		op := Opcode(opBuf[0])
		inst := Instruction{Op: op}

		switch op {
		// No-arg opcodes
		case MSG_START, MSG_END, ROLE_SYS, ROLE_USR, ROLE_AST, ROLE_TOOL,
			DEF_START, DEF_END, CALL_END, RESULT_END,
			SET_STREAM, STREAM_START, STREAM_END:
			// nothing

		// String arg
		case TXT_CHUNK, DEF_NAME, DEF_DESC, CALL_START, CALL_NAME,
			RESULT_START, RESULT_DATA, RESP_ID, RESP_MODEL, RESP_DONE,
			SET_MODEL, SET_STOP, STREAM_DELTA, COMMENT:
			s, err := readString(r)
			if err != nil {
				return nil, fmt.Errorf("ail.Decode %s: %w", op.Name(), err)
			}
			inst.Str = s

		// Float arg
		case SET_TEMP, SET_TOPP:
			f, err := readFloat64(r)
			if err != nil {
				return nil, fmt.Errorf("ail.Decode %s: %w", op.Name(), err)
			}
			inst.Num = f

		// Int arg
		case SET_MAX:
			i, err := readInt32(r)
			if err != nil {
				return nil, fmt.Errorf("ail.Decode %s: %w", op.Name(), err)
			}
			inst.Int = i

		// JSON arg
		case DEF_SCHEMA, CALL_ARGS, USAGE, STREAM_TOOL_DELTA:
			b, err := readBytes(r)
			if err != nil {
				return nil, fmt.Errorf("ail.Decode %s: %w", op.Name(), err)
			}
			inst.JSON = json.RawMessage(b)

		// RefID
		case IMG_REF, AUD_REF, TXT_REF:
			ref, err := readUint32(r)
			if err != nil {
				return nil, fmt.Errorf("ail.Decode %s: %w", op.Name(), err)
			}
			inst.Ref = ref

		// Key + Val
		case SET_META:
			k, err := readString(r)
			if err != nil {
				return nil, fmt.Errorf("ail.Decode SET_META key: %w", err)
			}
			v, err := readString(r)
			if err != nil {
				return nil, fmt.Errorf("ail.Decode SET_META val: %w", err)
			}
			inst.Key = k
			inst.Str = v

		// Key + JSON
		case EXT_DATA:
			k, err := readString(r)
			if err != nil {
				return nil, fmt.Errorf("ail.Decode EXT_DATA key: %w", err)
			}
			b, err := readBytes(r)
			if err != nil {
				return nil, fmt.Errorf("ail.Decode EXT_DATA json: %w", err)
			}
			inst.Key = k
			inst.JSON = json.RawMessage(b)

		default:
			return nil, fmt.Errorf("ail.Decode: unknown opcode 0x%02X", op)
		}

		p.Code = append(p.Code, inst)
	}

	return p, nil
}

// ─── Wire helpers ────────────────────────────────────────────────────────────

func writeString(w io.Writer, s string) error {
	return writeBytes(w, []byte(s))
}

func writeBytes(w io.Writer, b []byte) error {
	var lenBuf [4]byte
	binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(b)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return err
	}
	if len(b) > 0 {
		_, err := w.Write(b)
		return err
	}
	return nil
}

func writeFloat64(w io.Writer, f float64) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], math.Float64bits(f))
	_, err := w.Write(buf[:])
	return err
}

func writeInt32(w io.Writer, i int32) error {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], uint32(i))
	_, err := w.Write(buf[:])
	return err
}

func writeUint32(w io.Writer, u uint32) error {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], u)
	_, err := w.Write(buf[:])
	return err
}

func readString(r io.Reader) (string, error) {
	b, err := readBytes(r)
	return string(b), err
}

func readBytes(r io.Reader) ([]byte, error) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, err
	}
	n := binary.LittleEndian.Uint32(lenBuf[:])
	if n == 0 {
		return nil, nil
	}
	buf := make([]byte, n)
	_, err := io.ReadFull(r, buf)
	return buf, err
}

func readFloat64(r io.Reader) (float64, error) {
	var buf [8]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	return math.Float64frombits(binary.LittleEndian.Uint64(buf[:])), nil
}

func readInt32(r io.Reader) (int32, error) {
	var buf [4]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	return int32(binary.LittleEndian.Uint32(buf[:])), nil
}

func readUint32(r io.Reader) (uint32, error) {
	var buf [4]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(buf[:]), nil
}
