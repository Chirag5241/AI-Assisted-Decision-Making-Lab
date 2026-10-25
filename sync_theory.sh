#!/usr/bin/env bash
# Synchronize Obsidian vault into theory/ folder
VAULT_DIR="/Users/eeshanzele/Documents/Obsidian Vault/CS 598 Project"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THEORY_DIR="$REPO_DIR/theory"

if [ ! -d "$VAULT_DIR" ]; then
    echo "Error: Obsidian Vault directory not found at $VAULT_DIR"
    exit 1
fi

echo "Syncing from Obsidian Vault -> $THEORY_DIR..."
mkdir -p "$THEORY_DIR"
rsync -av --delete --exclude '.DS_Store' "$VAULT_DIR/" "$THEORY_DIR/"
echo "Sync complete!"
