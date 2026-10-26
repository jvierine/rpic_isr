#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
cargo test --manifest-path wasm/Cargo.toml
cargo build --manifest-path wasm/Cargo.toml --release --target wasm32-unknown-unknown
wasm-bindgen wasm/target/wasm32-unknown-unknown/release/picisr_web.wasm --target web --out-dir public/pkg
node stamp.mjs
