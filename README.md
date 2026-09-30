# Control Panel — AI-Native Task Engine for Snapdragon X Plus

> **Snapdragon AI Lab Build & Present Challenge 2026**  
> An AI-native, Windows 11 ARM64-native control plane for intelligent file management and local GenAI inference on Snapdragon X Plus.

---

## What It Does

Control Panel is a **persistent AI orchestrator** that takes natural-language task requests and executes them safely using ephemeral workers, local inference (CPU/NPU/GPU), and a learning memory system that improves over time.

```
You: "Organize my Downloads folder by file type"

Control Panel:
  ├─ Analyzes your intent  (AI inference or rule-based)
  ├─ Retrieves best matching template from experience memory
  ├─ Plans 3 ephemeral workers: scanner → classifier → organizer
  ├─ Spawns workers with exact tool permissions (read-only / read-write)
  ├─ Executes safely (dry-run first, safety-gated filesystem ops)
  ├─ Stores the execution experience (learns what worked)
  └─ Returns: ✓ COMPLETED — 247 files sorted into 8 categories
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLI  (Typer + Rich)                       │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                      ORCHESTRATOR                                 │
│   IDLE → ANALYZING → RETRIEVING → PLANNING → SPAWNING            │
│        → EXECUTING → COLLECTING → COMPARING → OPTIMIZING         │
│        → STORING → CLEANUP → IDLE                                │
└──┬──────────┬───────────┬──────────┬──────────┬──────────────────┘
   │          │           │          │          │
   ▼          ▼           ▼          ▼          ▼
INFERENCE  TEMPLATE   EXPERIENCE  WORKER    RESOURCE
RUNTIME    STORE      STORE       RUNTIME   GOVERNOR
   │
   ├─ Snapdragon (QNN → NPU / DirectML → GPU)
   ├─ Local CPU  (ONNX Runtime / GenAI)
   ├─ API        (OpenAI-compatible)
   └─ Rule-Based (always available)
                         │
                    SAFETY MANAGER
                         │
                    TOOL REGISTRY
                    ├─ fs.list_directory
                    ├─ fs.get_metadata
                    ├─ fs.create_directory
                    ├─ fs.move_file
                    └─ fs.copy_file
```

---

## Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **AI decides, tools execute** | Orchestrator uses inference for analysis/planning. Workers use filesystem tools with safety gates. |
| **Workers are ephemeral** | Created fresh per task, destroyed on completion. Never persistent state holders. |
| **Safety first** | Every tool call validated by SafetyManager. Denylist protects system dirs. Dry-run mode simulates. |
| **Learn from experience** | Every execution stored in SQLite. Templates updated based on outcomes. Comparator tracks improvements. |
| **Hardware-aware** | ResourceGovernor probes CPU/NPU/GPU. BackendSelectionEngine routes inference to best available accelerator. |
| **Local-first** | No internet required. Rule-based fallback always available. API optional. |

---

## Hardware Target

| Component | Spec |
|-----------|------|
| Device | Dell Inspiron 14 5441 |
| SoC | Snapdragon X Plus X1P42100 |
| CPU | 8-core Oryon @ 3.4 GHz (ARM64) |
| GPU | Adreno X1-45 |
| NPU | Hexagon (45 TOPS) |
| RAM | 16 GB LPDDR5x |
| OS | Windows 11 ARM64 Build 26200 |
| Python | 3.13.15 ARM64 (native, via uv) |

---

## Inference Backends

| Backend | Accelerator | Status | Notes |
|---------|-------------|--------|-------|
| `snapdragon` (QNN) | Hexagon NPU | 🔬 In progress | Requires QNN context binary |
| `snapdragon` (DirectML) | Adreno GPU | ⏳ Pending | No win_arm64 wheel yet |
| `local_cpu` | Oryon CPU | ✅ Working | ONNX Runtime GenAI |
| `api` | Remote | ✅ Optional | OpenAI-compatible endpoint |
| `rule_based` | — | ✅ Always | Pattern matching, zero latency |

---

## Quick Start

```powershell
# 1. Clone and set up (ARM64 Python 3.13 native)
git clone <repo>
cd control-panel
uv sync --extra dev

# 2. Run a task (dry-run = simulate, no actual file moves)
$env:PYTHONUTF8="1"
uv run control-panel run "organize my downloads" --path "C:\Users\me\Downloads" --dry-run

# 3. Check system and inference backends
uv run control-panel status
uv run control-panel backends

# 4. View task history (learns over time)
uv run control-panel history

# 5. Run for real (moves files)
uv run control-panel run "organize downloads" --path "C:\Users\me\Downloads"
```

---

## Project Structure

```
control-panel/
├── src/control_panel/
│   ├── common/          # Models, config, logging, event bus
│   ├── persistence/     # SQLite (aiosqlite, WAL, 6-table schema)
│   ├── resources/       # AcceleratorProbe, ResourceGovernor
│   ├── safety/          # SafetyManager — path validation, tool gates
│   ├── tools/           # Tool registry + 5 filesystem tools
│   ├── inference/       # 4 backends + selection engine + runtime
│   ├── experience/      # Experience store + collector (learns from history)
│   ├── templates/       # Template store + file_organizer builtin
│   ├── worker/          # WorkerFactory + WorkerRuntime (ephemeral workers)
│   ├── orchestrator/    # TaskAnalyzer, Planner, Comparator, 12-state machine
│   └── cli/             # Typer commands + Rich display
├── config/
│   ├── control_panel.toml  # Runtime config
│   └── models.toml         # Model registry
├── models/                 # Local ONNX model files (downloaded separately)
├── tests/
│   ├── unit/           # 25 tests
│   ├── integration/    # 8 tests
│   └── e2e/            # 2 tests
└── pyproject.toml
```

---

## Running Tests

```powershell
$env:PYTHONUTF8="1"
uv run pytest tests/ -v
# Expected: 35 passed
```

---

## Using a Local LLM (Phi-3 Mini)

```powershell
# Download Phi-3 Mini 4K INT4 ONNX (2.3 GB)
uv run python scripts/download_model.py

# Update config/models.toml — set file_path to models/phi3-mini-int4/...
# Then run — inference routes to local_cpu with real LLM
uv run control-panel run "sort my project files" --path "C:\path" --verbose
```

---

## NPU Acceleration (Track B — Qualcomm QNN)

The Hexagon NPU requires a **QNN context binary** (`.bin`) — a Qualcomm-compiled, quantized model artifact. Standard FP32/INT8 ONNX models do not run on the HTP.

```powershell
# Install Qualcomm AI Hub SDK
pip install qai-hub qai-hub-models

# Compile Llama or Phi model for QNN HTP (requires free AI Hub account)
uv run python scripts/compile_qnn_model.py --model phi3-mini --device snapdragon-x-plus

# Once compiled, the QNN .bin is placed in models/qnn/
# SnapdragonAdapter automatically detects and uses it
```

See [`docs/qnn_setup.md`](docs/qnn_setup.md) for the full guide.

---

## Safety

The AI **never has direct filesystem access**. Every operation goes through:

```
AI Decision → Safety Validation → Tool Execution → Side Effect Log
```

- **Denylist**: `C:\Windows`, `C:\Program Files` are permanently blocked
- **Permission modes**: Workers get `read_only` or `read_write` — never both unless needed
- **Dry-run**: All tools support `--dry-run` — simulates every operation, no writes
- **V0.1 forbidden**: `fs.delete_file`, `fs.delete_directory`, `shell.execute` are permanently blocked

---

## Built With

| Technology | Purpose |
|-----------|---------|
| Python 3.13 ARM64 | Runtime (native, no emulation) |
| ONNX Runtime 1.23 | Local CPU inference |
| ONNX Runtime GenAI 0.16 | Phi-3 / Phi-2 generation |
| ONNX Runtime QNN 2.6 | Hexagon NPU acceleration |
| Pydantic v2 | Data models throughout |
| aiosqlite | Async SQLite experience store |
| Typer + Rich | Terminal UI |
| structlog | Structured logging |
| uv | ARM64-native package management |

---

## License

MIT
