#!/bin/bash
# FASGA Docker Wrapper Script
# Makes it easy to run FASGA in Docker with GPU support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker is installed"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    print_success "Docker Compose is installed"
}

# Check if NVIDIA Container Toolkit is installed
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Container Toolkit may not be installed"
        echo "For GPU support, install nvidia-container-toolkit:"
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
        echo "Continuing anyway..."
        return 1
    fi
    print_success "NVIDIA Container Toolkit is installed"
    return 0
}

# Build Docker image
build_image() {
    print_header "Building FASGA Docker Image"
    
    check_docker
    check_docker_compose
    
    echo "Building image (this may take a few minutes)..."
    docker compose build
    
    print_success "Image built successfully"
}

# Check GPU support
check_gpu() {
    print_header "Checking GPU Support"
    
    check_docker
    
    if check_nvidia_docker; then
        echo ""
        echo "Running GPU diagnostic..."
        docker compose run --rm fasga uv run python check_cuda.py
    else
        print_warning "Cannot check GPU - NVIDIA Docker runtime not available"
        exit 1
    fi
}

# Run FASGA
run_fasga() {
    check_docker
    
    if [ $# -lt 3 ]; then
        print_error "Usage: $0 run <audio_file> <text_file> <output_file> [options]"
        echo ""
        echo "Example:"
        echo "  $0 run audio.mp3 text.txt output.srt"
        echo "  $0 run audio.mp3 text.txt output.srt --language es --verbose"
        exit 1
    fi
    
    AUDIO_FILE="$1"
    TEXT_FILE="$2"
    OUTPUT_FILE="$3"
    shift 3
    OPTIONS="$@"
    
    # Check if files exist
    if [ ! -f "$AUDIO_FILE" ]; then
        print_error "Audio file not found: $AUDIO_FILE"
        exit 1
    fi
    
    if [ ! -f "$TEXT_FILE" ]; then
        print_error "Text file not found: $TEXT_FILE"
        exit 1
    fi
    
    # Get absolute paths
    AUDIO_ABS=$(realpath "$AUDIO_FILE")
    TEXT_ABS=$(realpath "$TEXT_FILE")
    OUTPUT_ABS=$(realpath "$OUTPUT_FILE" 2>/dev/null || echo "$(pwd)/$OUTPUT_FILE")
    OUTPUT_DIR=$(dirname "$OUTPUT_ABS")
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    print_header "Running FASGA"
    echo "Audio:  $AUDIO_FILE"
    echo "Text:   $TEXT_FILE"
    echo "Output: $OUTPUT_FILE"
    echo ""
    
    docker run --rm --gpus all \
        -v "$AUDIO_ABS:/data/audio:ro" \
        -v "$TEXT_ABS:/data/text:ro" \
        -v "$OUTPUT_DIR:/data/output" \
        fasga:latest \
        uv run fasga /data/audio /data/text -o /data/output/$(basename "$OUTPUT_FILE") $OPTIONS
    
    if [ $? -eq 0 ]; then
        print_success "Subtitles generated successfully: $OUTPUT_FILE"
    else
        print_error "Processing failed"
        exit 1
    fi
}

# Interactive shell
shell() {
    print_header "Opening Interactive Shell"
    
    check_docker
    
    # Create data directory if it doesn't exist
    mkdir -p data
    
    echo "Data directory: $(pwd)/data"
    echo "Place your audio and text files in the 'data' directory"
    echo ""
    
    docker compose run --rm fasga bash
}

# Show help
show_help() {
    cat << EOF
FASGA Docker Wrapper - Easy GPU-accelerated subtitle generation

Usage: $0 <command> [options]

Commands:
  build                     Build the Docker image
  check                     Check GPU support and CUDA availability
  run <audio> <text> <out>  Process audiobook and generate subtitles
  shell                     Open interactive shell in container
  help                      Show this help message

Examples:
  # First time setup
  $0 build
  $0 check

  # Generate subtitles
  $0 run audio.mp3 book.txt subtitles.srt
  
  # With options
  $0 run audio.mp3 book.txt out.srt --language es --verbose
  $0 run audio.mp3 book.txt out.srt --whisper-model medium --device cpu
  
  # Interactive mode
  $0 shell
  > uv run fasga /data/audio.mp3 /data/text.txt -o /data/output.srt

For detailed documentation, see DOCKER.md

EOF
}

# Main script
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    COMMAND="$1"
    shift
    
    case "$COMMAND" in
        build)
            build_image
            ;;
        check)
            check_gpu
            ;;
        run)
            run_fasga "$@"
            ;;
        shell)
            shell
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"

