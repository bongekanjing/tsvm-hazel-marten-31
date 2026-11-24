// A tiny stack VM for make.fun's tiny stack ISA (ISA.md): the bytes each program writes and the exit code it
// ends with. MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR name the tree, /input and /output by default.
// Each step follows ISA.md's order: step limit, pc, opcode, immediate, stack checks, the instruction's traps, it.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const STACK_MAX = 256;
const CALLS_MAX = 64;
const MEM_WORDS = 256;
const STEP_LIMIT = 1000000;
const OUTPUT_MAX = 65536;

// opcode -> [immediate bytes, pops, pushes]
const OPS = new Map<number, [number, number, number]>([
  [0x00, [0, 0, 0]], // HALT
  [0x01, [1, 0, 1]], // PUSH8
  [0x02, [4, 0, 1]], // PUSH32
  [0x03, [0, 1, 0]], // POP
  [0x04, [0, 1, 2]], // DUP
  [0x05, [0, 2, 2]], // SWAP
  [0x06, [0, 2, 3]], // OVER
  [0x07, [0, 3, 3]], // ROT
  [0x10, [0, 2, 1]], // ADD
  [0x11, [0, 2, 1]], // SUB
  [0x12, [0, 2, 1]], // MUL
  [0x13, [0, 2, 1]], // DIV
  [0x14, [0, 2, 1]], // MOD
  [0x15, [0, 2, 1]], // DIVU
  [0x16, [0, 2, 1]], // MODU
  [0x17, [0, 1, 1]], // NEG
  [0x18, [0, 2, 1]], // AND
  [0x19, [0, 2, 1]], // OR
  [0x1a, [0, 2, 1]], // XOR
  [0x1b, [0, 1, 1]], // NOT
  [0x1c, [0, 2, 1]], // SHL
  [0x1d, [0, 2, 1]], // SHR
  [0x1e, [0, 2, 1]], // SHRU
  [0x20, [0, 2, 1]], // EQ
  [0x21, [0, 2, 1]], // LT
  [0x22, [0, 2, 1]], // LTU
  [0x30, [2, 0, 0]], // JMP
  [0x31, [2, 1, 0]], // JZ
  [0x32, [2, 1, 0]], // JNZ
  [0x33, [2, 0, 0]], // CALL
  [0x34, [0, 0, 0]], // RET
  [0x40, [0, 1, 1]], // LOAD
  [0x41, [0, 2, 0]], // STORE
  [0x50, [0, 1, 0]], // PRINT
  [0x51, [0, 1, 0]], // PUTC
  [0x52, [0, 0, 1]], // IN
  [0x53, [0, 1, 0]], // EXIT
]);

interface Answer {
  stdout: Uint8Array;
  exit: number;
}

/** Run one program on one input: the bytes written and the exit code. */
function run(program: Uint8Array, input: Uint8Array): Answer {
  const n = program.length;
  const D = new Int32Array(STACK_MAX);
  const C = new Int32Array(CALLS_MAX);
  const M = new Int32Array(MEM_WORDS);
  const out = new Uint8Array(OUTPUT_MAX);
  let sp = 0;
  let csp = 0;
  let olen = 0;
  let inPos = 0;
  let pc = 0;
  let steps = 0;
  const done = (exit: number): Answer => ({ stdout: out.slice(0, olen), exit });

  for (;;) {
    if (steps === STEP_LIMIT) return done(10);
    if (pc < 0 || pc >= n) return done(9);
    const spec = OPS.get(program[pc]!);
    if (spec === undefined) return done(5);
    const [imm, pops, pushes] = spec;
    if (pc + imm >= n) return done(9);
    steps += 1;
    const next = pc + 1 + imm;
    if (sp < pops) return done(2);
    if (sp - pops + pushes > STACK_MAX) return done(3);
    pc = next;
    const op = program[next - 1 - imm]!;
    const immU = imm === 1 ? program[next - 1]! : imm === 2 ? program[next - 2]! | (program[next - 1]! << 8) : 0;
    const immS = imm === 1 ? (program[next - 1]! << 24) >> 24 : imm === 2 ? (immU << 16) >> 16 : 0;
    const rel = next + immS;

    if (op >= 0x10 && op <= 0x16) {
      const b = D[sp - 1]!;
      const a = D[sp - 2]!;
      let r: number;
      if (op === 0x10) r = (a + b) | 0;
      else if (op === 0x11) r = (a - b) | 0;
      else if (op === 0x12) r = Math.imul(a, b);
      else if (op === 0x13) {
        if (b === 0) return done(4);
        r = a === -2147483648 && b === -1 ? a : (a / b) | 0;
      } else if (op === 0x14) {
        if (b === 0) return done(4);
        r = b === -1 ? 0 : (a % b) | 0;
      } else if (op === 0x15) {
        if (b === 0) return done(4);
        r = Math.floor((a >>> 0) / (b >>> 0)) | 0;
      } else {
        if (b === 0) return done(4);
        r = ((a >>> 0) % (b >>> 0)) | 0;
      }
      sp -= 2;
      D[sp] = r;
      sp += 1;
      continue;
    }
    if (op >= 0x18 && op <= 0x22 && op !== 0x1b) {
      const b = D[sp - 1]!;
      const a = D[sp - 2]!;
      let r: number;
      if (op === 0x18) r = a & b;
      else if (op === 0x19) r = a | b;
      else if (op === 0x1a) r = a ^ b;
      else if (op === 0x1c) r = a << (b & 31);
      else if (op === 0x1d) r = a >> (b & 31);
      else if (op === 0x1e) r = (a >>> (b & 31)) | 0;
      else if (op === 0x20) r = a === b ? 1 : 0;
      else if (op === 0x21) r = a < b ? 1 : 0;
      else r = (a >>> 0) < (b >>> 0) ? 1 : 0;
      sp -= 2;
      D[sp] = r;
      sp += 1;
      continue;
    }

    switch (op) {
      case 0x00:
        return done(0);
      case 0x01:
        D[sp++] = immS;
        break;
      case 0x02:
        D[sp++] =
          program[next - 4]! | (program[next - 3]! << 8) | (program[next - 2]! << 16) | (program[next - 1]! << 24);
        break;
      case 0x03:
        sp -= 1;
        break;
      case 0x04:
        D[sp] = D[sp - 1]!;
        sp += 1;
        break;
      case 0x05: {
        const t = D[sp - 1]!;
        D[sp - 1] = D[sp - 2]!;
        D[sp - 2] = t;
        break;
      }
      case 0x06:
        D[sp] = D[sp - 2]!;
        sp += 1;
        break;
      case 0x07: {
        const t = D[sp - 3]!;
        D[sp - 3] = D[sp - 2]!;
        D[sp - 2] = D[sp - 1]!;
        D[sp - 1] = t;
        break;
      }
      case 0x17:
        D[sp - 1] = (-D[sp - 1]! | 0);
        break;
      case 0x1b:
        D[sp - 1] = ~D[sp - 1]!;
        break;
      case 0x30:
        pc = rel;
        break;
      case 0x31:
      case 0x32: {
        const v = D[--sp]!;
        if (op === 0x31 ? v === 0 : v !== 0) pc = rel;
        break;
      }
      case 0x33:
        if (csp === CALLS_MAX) return done(7);
        C[csp++] = next;
        pc = immU;
        break;
      case 0x34:
        if (csp === 0) return done(8);
        pc = C[--csp]!;
        break;
      case 0x40: {
        const addr = D[sp - 1]!;
        if (addr < 0 || addr >= MEM_WORDS) return done(6);
        D[sp - 1] = M[addr]!;
        break;
      }
      case 0x41: {
        const addr = D[sp - 1]!;
        const v = D[sp - 2]!;
        if (addr < 0 || addr >= MEM_WORDS) return done(6);
        sp -= 2;
        M[addr] = v;
        break;
      }
      case 0x50: {
        const text = Buffer.from(String(D[sp - 1]!), 'utf8');
        if (olen + text.length > OUTPUT_MAX) return done(11);
        sp -= 1;
        out.set(text, olen);
        olen += text.length;
        break;
      }
      case 0x51:
        if (olen + 1 > OUTPUT_MAX) return done(11);
        out[olen++] = D[--sp]! & 255;
        break;
      case 0x52:
        D[sp++] = inPos < input.length ? input[inPos++]! : -1;
        break;
      case 0x53:
        return done(D[--sp]! & 255);
    }
  }
}

for (const id of readdirSync(I).sort()) {
  const dir = join(I, id);
  const answer = run(readFileSync(join(dir, 'program.bin')), readFileSync(join(dir, 'input.bin')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), answer.stdout);
  writeFileSync(join(O, id, 'exit.txt'), `${answer.exit}\n`);
}
