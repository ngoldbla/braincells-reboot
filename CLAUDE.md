# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hugging Face AI Sheets is an open-source tool for building, enriching, and transforming datasets using AI models with no code. It integrates with the Hugging Face Hub for access to thousands of open models via Inference Providers, and supports custom/local LLMs (Ollama, etc.) via OpenAI-compatible API endpoints.

## Development Commands

```bash
# Development (Vite dev server with SSR on port 5173)
pnpm dev

# Production build
pnpm build

# Serve production build (Express on port 3000)
pnpm serve

# Testing
pnpm test                # Run vitest
pnpm test.unit.ui        # Run vitest with UI

# Linting & Formatting (Biome)
pnpm lint                # Lint with auto-fix
pnpm fmt                 # Format code
pnpm lint.check          # Check only (CI)
pnpm fmt.check           # Check only (CI)

# Type checking only
pnpm build.types
```

## Environment Setup

Create `.env` in root with at minimum:
```
HF_TOKEN=your_hugging_face_token
```

Optional for custom LLMs:
```
MODEL_ENDPOINT_URL=http://localhost:11434
MODEL_ENDPOINT_NAME=llama3
```

## Architecture

**Stack:** Qwik + QwikCity (SSR framework), Express.js (production server), Vite (build), Tailwind CSS

**Key directories:**
- `src/routes/` - QwikCity file-based routing. `index.tsx` = pages, `index.ts` = API endpoints
- `src/components/` - Stateless reusable UI components
- `src/features/` - Feature modules with business logic (table, datasets, execution, export/import, etc.)
- `src/services/` - External integrations (auth, db, inference, repository, websearch, cache)
- `src/usecases/` - High-level application logic orchestrating services (generate-cells, export-to-hub, import-from-hub, etc.)
- `src/state/` - Global Qwik state management
- `src/loaders/` - QwikCity route loaders for data fetching
- `src/config.ts` - All environment variable configuration

**Data layer:**
- SQLite (via Sequelize) - primary local storage
- DuckDB - SQL analytics
- LanceDB - vector embeddings for semantic search

**ML/AI integration:**
- `@huggingface/hub` - Hub dataset operations
- `@huggingface/inference` - Inference API calls
- `@huggingface/transformers` - Local embeddings

## Code Style

- **Formatter:** Biome - single quotes, 2-space indent, 80 char line width, semicolons required
- **Linting:** Biome with `noUnusedImports: error`
- **TypeScript:** Strict mode, path alias `~/*` → `./src/*`
- **JSX quotes:** Double quotes in JSX, single quotes elsewhere

## Key Patterns

- Use cases in `src/usecases/` orchestrate complex operations across multiple services
- Features in `src/features/` are self-contained modules with their own components and logic
- Services in `src/services/` handle external integrations and should be stateless
- Route loaders (`src/loaders/`) fetch data server-side for SSR
