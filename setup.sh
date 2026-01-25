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
    # Using printf for aligned columns. The %-20s reserves 20 characters for the command string.
    printf "\n"
    printf "${BLUE}%s${NC}\n" "Usage:"
    printf "  %s [command]\n\n" "$0"
    printf "${BLUE}%s${NC}\n" "Description:"
    printf "%s\n\n" "  A script to manage the project's virtual environment and dependencies."
    printf "${BLUE}%s${NC}\n" "Available Commands:"
    printf "${GREEN}%s${NC} %s\n" "  install, -i" " Sets up the environment and installs packages (default action)."
    printf "${GREEN}%s${NC} %s\n" "  clean,   -c" " Deletes the virtual environment and build artifacts."
    printf "${GREEN}%s${NC} %s\n" "  help,    -h" " Displays this help message."
    printf "\n"
}

# function to check installation of a package with argument and visual feedback
check_package_installation() {
    PACKAGE_NAME=$1
    echo -n "🔍 Checking installation of package '$PACKAGE_NAME'... "
    if python3 -c "import $PACKAGE_NAME" &>/dev/null; then
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

echo -e "${BLUE}--- Starting LEAFFLICTION Project Setup ---${NC}"

echo "🔎 Checking Python version..."
if ! python3 -c 'import sys; assert sys.version_info >= (3, 10)' &>/dev/null; then
    echo -e "${RED}ERROR: Python 3.10 or higher is required.${NC}"
    echo "Please install a compatible Python version and try again."
    exit 1
fi
echo -e "${GREEN}✅ Python version is compatible.${NC}"

VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo "♻️  Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "🐍 Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
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
pip install . > /dev/null
echo -e "${GREEN}✅ Dependencies installed successfully.${NC}"

echo "🔬 Verifying installation..."
check_package_installation "numpy"
check_package_installation "matplotlib"
check_package_installation "cv2"
check_package_installation "plantcv"
check_package_installation "sklearn"
check_package_installation "skimage"

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
