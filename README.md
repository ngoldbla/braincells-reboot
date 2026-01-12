<div align="center">

# Braincells

**Intelligent Spreadsheet Automation with Local AI**

*Transform your research workstation into an intelligent data processing hub. Think of each cell in your spreadsheet as a "brain cell" capable of understanding and transforming data.*

Developed by [Kennesaw State University Office of Research](https://research.kennesaw.edu/)

</div>

## What is Braincells?

Braincells is a desktop application that brings AI-powered automation to the familiar spreadsheet interface. Run large language models locally on your machine or connect to cloud providers - no data leaves your computer unless you choose.

**Key Features:**
- **Local LLM Inference** - Run models directly on your machine using llama.cpp with GPU acceleration
- **Cloud Fallback** - Connect to HuggingFace, OpenRouter, or OpenAI when needed
- **Privacy First** - Your data stays on your machine with local inference
- **Batch Processing** - Process hundreds of cells in parallel with intelligent rate limiting
- **No Code Required** - Define AI transformations using natural language prompts

## Installation

### Download

Download the latest release for your platform:

| Platform | Download |
|----------|----------|
| macOS (Apple Silicon) | [Braincells_x.x.x_aarch64.dmg](https://github.com/kennesaw-state-university/braincells/releases/latest) |
| macOS (Intel) | [Braincells_x.x.x_x64.dmg](https://github.com/kennesaw-state-university/braincells/releases/latest) |
| Windows | [Braincells_x.x.x_x64-setup.exe](https://github.com/kennesaw-state-university/braincells/releases/latest) |
| Linux | [Braincells_x.x.x_amd64.AppImage](https://github.com/kennesaw-state-university/braincells/releases/latest) |

### Build from Source

Requirements:
- Node.js 20.19+ or 22.12+
- pnpm 9+
- Rust 1.77.2+

```bash
git clone https://github.com/kennesaw-state-university/braincells.git
cd braincells
pnpm install
pnpm tauri dev
```

## Quick Start

### 1. Configure an Inference Backend

Open **Settings** from the sidebar and configure one of the following:

#### Option A: Local Model (Recommended for Privacy)

1. Go to **Settings > Local Models**
2. Click **Download Model** and select a recommended model for your system
3. Wait for the download to complete
4. Click **Activate** to use the model

Recommended models by system RAM:
| RAM | Model |
|-----|-------|
| 8GB | Qwen2.5-3B-Instruct (Q4_K_M) |
| 16GB | Llama-3.2-8B-Instruct (Q4_K_M) |
| 32GB+ | Qwen2.5-14B-Instruct (Q4_K_M) |

#### Option B: Cloud Provider

1. Go to **Settings > Cloud Providers**
2. Enter your API key for one of:
   - **HuggingFace** - Get key at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - **OpenRouter** - Get key at [openrouter.ai/keys](https://openrouter.ai/keys)
   - **OpenAI** - Get key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
3. Select a model and click **Activate**

### 2. Create Your First AI Column

1. Click **+ Add Column** in your spreadsheet
2. Select **AI Generated** as the column type
3. Write a prompt using `{{column_name}}` to reference other columns
4. Click **Generate** to process all rows

Example prompt:
```
Summarize the following text in one sentence: {{description}}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Braincells Desktop App                    │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Qwik + TypeScript)                               │
│  ├── Spreadsheet UI                                         │
│  ├── Settings Panel                                         │
│  └── Inference Bridge ─────────────────┐                    │
├─────────────────────────────────────────┼───────────────────┤
│  Tauri IPC Layer                        │                   │
├─────────────────────────────────────────┼───────────────────┤
│  Backend (Rust)                         ▼                   │
│  ├── Inference Pool (rate limiting, batching)               │
│  ├── Local Engine (llama.cpp)                               │
│  │   ├── Metal acceleration (macOS)                         │
│  │   ├── CUDA acceleration (Windows/Linux)                  │
│  │   └── CPU fallback                                       │
│  └── Cloud Engine                                           │
│      ├── HuggingFace Inference API                          │
│      ├── OpenRouter                                         │
│      └── OpenAI                                             │
└─────────────────────────────────────────────────────────────┘
```

## Development

### Commands

```bash
# Development (Tauri + Vite)
pnpm tauri dev

# Build production app
pnpm tauri build

# Frontend only (web mode)
pnpm dev

# Run tests
pnpm test

# Lint & format
pnpm lint
pnpm fmt

# Type check
pnpm build.types
```

### Project Structure

```
braincells/
├── src/                    # Frontend (Qwik)
│   ├── components/         # Reusable UI components
│   ├── features/           # Feature modules (table, execution, etc.)
│   ├── routes/             # QwikCity file-based routing
│   ├── services/           # External integrations
│   │   ├── inference/      # Inference routing logic
│   │   └── tauri/          # Tauri IPC bindings
│   └── usecases/           # Application logic
├── src-tauri/              # Backend (Rust)
│   ├── src/
│   │   ├── commands/       # Tauri IPC commands
│   │   ├── llm/            # LLM engine implementations
│   │   │   ├── local.rs    # llama.cpp backend
│   │   │   ├── cloud.rs    # Cloud provider backends
│   │   │   └── pool.rs     # Concurrency management
│   │   └── models/         # Model management
│   └── Cargo.toml
└── .github/workflows/      # CI/CD
```

## Releasing

### Setup (One-Time)

1. Generate signing keys:
   ```bash
   cd src-tauri
   npx @tauri-apps/cli signer generate -w ~/.tauri/braincells.key
   ```

2. Add secrets to GitHub repository:
   - `TAURI_SIGNING_PRIVATE_KEY` - Contents of `~/.tauri/braincells.key`
   - `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` - Your password

3. Update `src-tauri/tauri.conf.json` line 63 with the public key from step 1

### Creating a Release

1. Update version in `src-tauri/tauri.conf.json` and `src-tauri/Cargo.toml`
2. Commit and tag:
   ```bash
   git add .
   git commit -m "Release v1.0.0"
   git tag v1.0.0
   git push origin main --tags
   ```
3. GitHub Actions will build and publish the release automatically

## Configuration

### Environment Variables (Web Mode Only)

When running in web mode (not desktop), these environment variables apply:

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | - |
| `MODEL_ENDPOINT_URL` | Custom OpenAI-compatible endpoint | - |
| `MODEL_ENDPOINT_NAME` | Model name for custom endpoint | - |
| `DATA_DIR` | Data storage directory | `./data` |

### Desktop Settings

All settings are configured through the Settings panel in the app and stored in:
- **macOS**: `~/Library/Application Support/braincells/settings.json`
- **Windows**: `%APPDATA%/braincells/settings.json`
- **Linux**: `~/.config/braincells/settings.json`

## Troubleshooting

### Local Model Not Loading

1. Ensure you have enough RAM for the model
2. Check that the model file downloaded completely
3. Try a smaller quantization (Q4_K_M instead of Q8_0)

### Slow Inference

1. **macOS**: Metal acceleration is automatic on Apple Silicon
2. **Windows/Linux**: Install CUDA toolkit for GPU acceleration
3. Reduce `max_concurrent_requests` in Settings if hitting memory limits

### App Crashes on Startup

1. Delete settings file (see paths above)
2. Restart the app - it will recreate default settings

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Credits

Braincells is developed by the [Kennesaw State University Office of Research](https://research.kennesaw.edu/).

Built on top of [Hugging Face AI Sheets](https://github.com/huggingface/aisheets) with significant enhancements for desktop deployment and local LLM inference.
