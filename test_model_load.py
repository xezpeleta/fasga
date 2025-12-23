#!/usr/bin/env python3
"""Quick test to verify Whisper model loading works with the patches."""

import sys
from fasga.transcriber import WhisperTranscriber

print("Testing Whisper model loading with PyTorch 2.6+ compatibility patches...")
print("This will download models on first run (may take a few minutes).\n")

try:
    # Create a small transcriber instance
    transcriber = WhisperTranscriber(
        model_size="tiny",  # Use smallest model for quick test
        device="cpu",       # Use CPU to avoid GPU requirements
        compute_type="float32"
    )
    
    print("✓ WhisperTranscriber initialized")
    
    # Try to load the model
    transcriber._load_model()
    
    print("✓ Whisper model loaded successfully!")
    print("\n🎉 All compatibility patches are working correctly!")
    
    # Cleanup
    transcriber.cleanup()
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


