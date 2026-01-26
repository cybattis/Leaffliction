#!/bin/bash

set -e

# --- Styles ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to clean the project directory
clean_project() {
    echo -e "${BLUE}--- Cleaning Project Directory ---${NC}"

    if command -v deactivate &> /dev/null; then
        echo "🔌 Deactivating virtual environment..."
        deactivate
    fi

    echo "🗑️  Removing virtual environment (.venv)..."
    rm -rf .venv

    echo "🗑️  Removing build files (egg-info, build)..."
    rm -rf build ./*.egg-info

    echo "🗑️  Removing Python cache files (__pycache__, *.pyc)..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

    echo -e "\n${GREEN}✅ Project cleaned successfully.${NC}"
    echo -e "${BLUE}------------------------------------${NC}"
}

# Function to display a formatted help message
show_help() {
    printf "\n"
    printf "${BLUE}Usage:${NC}\n"
    printf "  %s [command]\n\n" "$0"
    printf "${BLUE}Description:${NC}\n"
    printf "  A script to manage the project's virtual environment and dependencies.\n\n"
    printf "${BLUE}Available Commands:${NC}\n"
    printf "  ${GREEN}install, -i${NC}   Sets up the environment and installs packages (default action).\n"
    printf "  ${GREEN}clean,   -c${NC}   Deletes the virtual environment and build artifacts.\n"
    printf "  ${GREEN}help,    -h${NC}   Displays this help message.\n"
    printf "\n"
}

# function to check installation of a package with argument and visual feedback
check_package_installation() {
    PACKAGE_NAME=$1
    echo -n "🔍 Checking installation of package '$PACKAGE_NAME'... "
    # Use the discovered python executable
    if "$PYTHON_EXEC" -c "import $PACKAGE_NAME" &>/dev/null; then
        echo -e "${GREEN}Installed ✅${NC}"
    else
        echo -e "${RED}Not Installed ❌${NC}"
        exit 1
    fi
}

case "$1" in
    install|"-i"|"")
        ;;
    clean|"-c")
        clean_project
        exit 0
        ;;
    help|-h)
        show_help
        exit 0
        ;;
    *)
        printf "Unknown command: %s \n" "$1"
        exit 1
        ;;
esac

echo -e "${BLUE}--- Starting Leaffliction Project Setup ---${NC}"

echo "🔎 Searching for most recent compatible Python version..."

# List of potential python executables to check, from newest to oldest
CANDIDATES=("python3")
PYTHON_EXEC=""

for cmd in "${CANDIDATES[@]}"; do
    if command -v "$cmd" &> /dev/null; then
        # Check if version is >= 3.11
        if "$cmd" -c 'import sys; assert sys.version_info >= (3, 10)' &>/dev/null; then
            PYTHON_EXEC="$cmd"
            VERSION_STR=$($cmd --version)
            echo -e "${GREEN}✅ Selected $VERSION_STR ($cmd)${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_EXEC" ]; then
    echo -e "${RED}ERROR: No compatible Python version found (>= 3.10).${NC}"
    echo "Please install Python 3.10 or higher."
    exit 1
fi

VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo "♻️  Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "🐍 Creating virtual environment in '$VENV_DIR' using $PYTHON_EXEC..."
    "$PYTHON_EXEC" -m venv "$VENV_DIR"
    echo -e "${GREEN}✅ Virtual environment created.${NC}"
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
else
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
fi

echo "🚀 Activating virtual environment..."
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

echo "📦 Installing project dependencies from pyproject.toml..."
pip install --upgrade pip > /dev/null
# This installs the package itself plus the [dev] optional dependencies
pip install ".[dev]" > /dev/null
echo -e "${GREEN}✅ Dependencies installed successfully.${NC}"

echo "🔬 Verifying installation..."
# Note: check_package_installation now uses $PYTHON_EXEC, but inside the venv
# 'python3' and '$PYTHON_EXEC' might point to the venv python depending on linking.
# However, inside venv, we usually just want 'python' or 'python3'.
# Resetting PYTHON_EXEC to 'python' ensures we check packages inside the active venv.
PYTHON_EXEC="python"

check_package_installation "numpy"
check_package_installation "matplotlib"
check_package_installation "cv2"
check_package_installation "plantcv"
check_package_installation "sklearn"
check_package_installation "skimage"
check_package_installation "flake8"
check_package_installation "tensorflow"
check_package_installation "keras"

echo "🔗 Checking for dependency conflicts..."
if ! pip check &>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Dependency conflicts detected. Run 'pip check' for details.${NC}"
else
    echo -e "${GREEN}✅ No dependency conflicts found.${NC}"
fi

echo -e "\n${GREEN}🎉 Setup complete! The virtual environment is ready and active.${NC}\n"
echo ""
echo "To deactivate the virtual environment later, simply run:"
echo -e "  ${YELLOW}deactivate${NC}"
echo "To reactivate it, run:"
echo -e "  ${YELLOW}source $ACTIVATE_SCRIPT${NC}"
echo -e "\n${BLUE}------------------------------------${NC}"