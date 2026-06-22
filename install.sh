#!/usr/bin/env bash
# ppagent installer for macOS
# Usage: curl -fsSL <url> | bash   (or just: bash install.sh)

set -euo pipefail

# ─── Colors ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[info]${NC}  $*"; }
success() { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
error()   { echo -e "${RED}[error]${NC} $*"; }
header()  { echo -e "\n${BOLD}$*${NC}"; }

# ─── Preflight ───────────────────────────────────────────────────────────────

if [[ "$(uname)" != "Darwin" ]]; then
    error "This installer is for macOS only. Detected: $(uname)"
    exit 1
fi

echo -e "${BOLD}"
echo "  ╔═══════════════════════════════════╗"
echo "  ║      ppagent installer (macOS)    ║"
echo "  ╚═══════════════════════════════════╝"
echo -e "${NC}"

# ─── Homebrew ────────────────────────────────────────────────────────────────

header "Checking Homebrew..."
if command -v brew &>/dev/null; then
    success "Homebrew found: $(brew --prefix)"
else
    info "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add to PATH for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    success "Homebrew installed"
fi

# ─── Python 3.12+ ────────────────────────────────────────────────────────────

header "Checking Python..."

PYTHON_CMD=""
for cmd in python3.12 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [[ "$major" -ge 3 && "$minor" -ge 12 ]]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [[ -n "$PYTHON_CMD" ]]; then
    success "Python found: $($PYTHON_CMD --version)"
else
    info "Python 3.12+ not found. Installing via Homebrew..."
    brew install python@3.12
    PYTHON_CMD="python3.12"
    success "Python 3.12 installed"
fi

# ─── uv ──────────────────────────────────────────────────────────────────────

header "Checking uv..."
if command -v uv &>/dev/null; then
    success "uv found: $(uv --version)"
else
    info "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the shell additions
    if [[ -f "$HOME/.local/bin/env" ]]; then
        source "$HOME/.local/bin/env"
    elif [[ -f "$HOME/.cargo/env" ]]; then
        source "$HOME/.cargo/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
    success "uv installed: $(uv --version)"
fi

# ─── HuggingFace CLI ─────────────────────────────────────────────────────────

header "Checking HuggingFace CLI..."
if command -v hf &>/dev/null; then
    success "hf CLI found: $(hf --version 2>/dev/null || echo 'installed')"
else
    info "hf CLI not found. Installing..."
    uv tool install "huggingface_hub[cli]"
    success "hf CLI installed"
fi

# ─── Clone / Install ppagent ─────────────────────────────────────────────────

INSTALL_DIR="${PPAGENT_DIR:-$HOME/ppagent}"

header "Installing ppagent..."
if [[ -d "$INSTALL_DIR/pyproject.toml" ]] || [[ -f "$INSTALL_DIR/pyproject.toml" ]]; then
    warn "ppagent directory already exists at $INSTALL_DIR"
    info "Updating dependencies..."
    cd "$INSTALL_DIR"
    uv sync
    success "Dependencies updated"
else
    # Check if we're running from within the repo
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/pyproject.toml" ]] && grep -q "ppagent" "$SCRIPT_DIR/pyproject.toml"; then
        info "Running from source directory: $SCRIPT_DIR"
        INSTALL_DIR="$SCRIPT_DIR"
        cd "$INSTALL_DIR"
    else
        info "Installing to: $INSTALL_DIR"
        if command -v git &>/dev/null; then
            git clone "${PPAGENT_REPO:-https://github.com/PhDeasy-org/your-paper-reading-agent.git}" "$INSTALL_DIR" 2>/dev/null || {
                warn "Git clone failed. Creating project directory..."
                mkdir -p "$INSTALL_DIR"
            }
        else
            warn "Git not found. Creating project directory..."
            mkdir -p "$INSTALL_DIR"
        fi
        cd "$INSTALL_DIR"
    fi
    uv sync
    success "ppagent installed"
fi

# ─── Config setup ────────────────────────────────────────────────────────────

header "Setting up configuration..."

# Config lives entirely in ~/.config/ppagent/ (outside the project tree so it
# survives reinstalls). The installer only seeds that path when empty — it
# never writes config/profile files into the project directory.
CFG_DIR="$HOME/.config/ppagent"
if [[ -f "$CFG_DIR/settings.toml" ]]; then
    success "Config already exists (skipping init)"
else
    mkdir -p "$CFG_DIR"
    uv run ppagent config init
fi

if [[ ! -f "$CFG_DIR/profile.md" ]]; then
    warn "No research profile found. Run ppagent config or edit $CFG_DIR/profile.md to set your interests."
fi

# ─── Shell integration ───────────────────────────────────────────────────────

header "Setting up shell integration..."

PPAGENT_BIN="$INSTALL_DIR/.venv/bin"

# Detect shell
if [[ -n "${ZSH_VERSION:-}" ]] || [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ -n "${BASH_VERSION:-}" ]] || [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
else
    SHELL_RC="$HOME/.profile"
fi

ALIAS_LINE="export PATH=\"$PPAGENT_BIN:\$PATH\"  # ppagent CLI"

if [[ -f "$SHELL_RC" ]] && grep -q "ppagent" "$SHELL_RC"; then
    success "PATH already configured in $SHELL_RC"
else
    echo "" >> "$SHELL_RC"
    echo "# ppagent — paper reading agent CLI" >> "$SHELL_RC"
    echo "$ALIAS_LINE" >> "$SHELL_RC"
    success "Added ppagent to PATH in $SHELL_RC"
fi

# ─── Shell completion ───────────────────────────────────────────────────────

header "Setting up shell completion..."

COMPLETION_INSTALLED=false

# Detect shell and install completion via Typer's built-in mechanism
if [[ -n "${ZSH_VERSION:-}" ]] || [[ "$SHELL" == *"zsh"* ]]; then
    # zsh completion — generate via Python so shellingham can't mis-detect bash
    # (install.sh itself runs under bash, which would cause --show-completion to
    # emit bash-format code into the zsh completion file).
    COMPLETION_FILE="$HOME/.zfunc/_ppagent"
    if [[ -f "$COMPLETION_FILE" ]]; then
        success "zsh completion already installed"
        COMPLETION_INSTALLED=true
    else
        mkdir -p "$HOME/.zfunc"
        if uv run python -c "
from typer._completion_shared import get_completion_script
print(get_completion_script(prog_name='ppagent', complete_var='_PPAGENT_COMPLETE', shell='zsh'))
" > "$COMPLETION_FILE" 2>/dev/null; then
            # Ensure .zfunc is in fpath
            if ! grep -q '.zfunc' "$SHELL_RC" 2>/dev/null; then
                echo "" >> "$SHELL_RC"
                echo "# ppagent shell completion" >> "$SHELL_RC"
                echo 'fpath=(~/.zfunc $fpath)' >> "$SHELL_RC"
                echo 'autoload -Uz compinit && compinit' >> "$SHELL_RC"
            fi
            success "zsh completion installed at $COMPLETION_FILE"
            COMPLETION_INSTALLED=true
        else
            warn "Could not generate zsh completion (will work after 'source $SHELL_RC')"
        fi
    fi
elif [[ -n "${BASH_VERSION:-}" ]] || [[ "$SHELL" == *"bash"* ]]; then
    # bash completion
    COMPLETION_FILE="$HOME/.bash_completions/ppagent.bash"
    if [[ -f "$COMPLETION_FILE" ]]; then
        success "bash completion already installed"
        COMPLETION_INSTALLED=true
    else
        mkdir -p "$HOME/.bash_completions"
        if uv run python -c "
from typer._completion_shared import get_completion_script
print(get_completion_script(prog_name='ppagent', complete_var='_PPAGENT_COMPLETE', shell='bash'))
" > "$COMPLETION_FILE" 2>/dev/null; then
            if ! grep -q 'ppagent.bash' "$SHELL_RC" 2>/dev/null; then
                echo "" >> "$SHELL_RC"
                echo "# ppagent shell completion" >> "$SHELL_RC"
                echo 'source ~/.bash_completions/ppagent.bash' >> "$SHELL_RC"
            fi
            success "bash completion installed at $COMPLETION_FILE"
            COMPLETION_INSTALLED=true
        else
            warn "Could not generate bash completion (will work after 'source $SHELL_RC')"
        fi
    fi
elif [[ -n "${FISH_VERSION:-}" ]] || [[ "$SHELL" == *"fish"* ]]; then
    # fish completion
    COMPLETION_FILE="$HOME/.config/fish/completions/ppagent.fish"
    if [[ -f "$COMPLETION_FILE" ]]; then
        success "fish completion already installed"
        COMPLETION_INSTALLED=true
    else
        mkdir -p "$HOME/.config/fish/completions"
        if uv run python -c "
from typer._completion_shared import get_completion_script
print(get_completion_script(prog_name='ppagent', complete_var='_PPAGENT_COMPLETE', shell='fish'))
" > "$COMPLETION_FILE" 2>/dev/null; then
            success "fish completion installed at $COMPLETION_FILE"
            COMPLETION_INSTALLED=true
        else
            warn "Could not generate fish completion"
        fi
    fi
else
    warn "Unknown shell. Install manually: ppagent --install-completion"
fi

if [[ "$COMPLETION_INSTALLED" == false ]]; then
    info "You can install completion later with: ppagent --install-completion"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${GREEN}Installation complete!${NC}"
echo ""
echo -e "  Run ${BLUE}ppagent config${NC} to set up your API keys and research profile."
echo ""
