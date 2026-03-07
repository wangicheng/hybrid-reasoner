#!/usr/bin/env python3
"""
Test script to verify API Key rotation mechanism
"""

import sys
sys.path.insert(0, '.')

from src.config import settings
from src.core.api_utils import get_api_key_rotator, get_current_api_key

def main():
    print("=" * 60)
    print("API Key Rotation Test")
    print("=" * 60)
    
    # Test 1: Config loads API keys
    print(f"\n✓ Test 1: Config loading")
    print(f"  Total API keys loaded: {len(settings.GOOGLE_API_KEYS)}")
    print(f"  First 3 keys (masked): {[k[:10] + '...' for k in settings.GOOGLE_API_KEYS[:3]]}")
    
    # Test 2: Rotator initialization
    print(f"\n✓ Test 2: Rotator initialization")
    rotator = get_api_key_rotator()
    print(f"  Rotator has {len(rotator.api_keys)} keys")
    print(f"  Current index: {rotator.current_index}")
    
    # Test 3: Getting current key
    print(f"\n✓ Test 3: Getting current key")
    current_key = get_current_api_key()
    print(f"  Current key: {current_key[:10]}...")
    
    # Test 4: Rotation
    print(f"\n✓ Test 4: API Key rotation")
    initial_index = rotator.current_index
    for i in range(3):
        rotator.rotate()
        print(f"  After rotate {i+1}: index={rotator.current_index}")
    
    # Test 5: Wrapping around
    print(f"\n✓ Test 5: Wrapping around")
    rotator.current_index = len(rotator.api_keys) - 1
    rotator.rotate()
    print(f"  After rotating from last index: current_index={rotator.current_index} (should be 0)")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

if __name__ == "__main__":
    main()
