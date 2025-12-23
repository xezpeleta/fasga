#!/usr/bin/env python3
"""Debug script to analyze SRT timestamps."""

import sys

def parse_srt_time(time_str):
    """Parse SRT timestamp to seconds."""
    hours, minutes, rest = time_str.split(':')
    seconds, milliseconds = rest.split(',')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def analyze_srt(filepath):
    """Analyze SRT file timestamps."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    timestamps = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            start_str, end_str = line.split(' --> ')
            start = parse_srt_time(start_str)
            end = parse_srt_time(end_str)
            timestamps.append((start, end))
        i += 1
    
    if timestamps:
        print(f"Total subtitles: {len(timestamps)}")
        print(f"\nFirst 10 timestamps (in seconds):")
        for i, (start, end) in enumerate(timestamps[:10], 1):
            print(f"  {i}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        
        print(f"\nTimestamp range:")
        print(f"  First subtitle starts: {timestamps[0][0]:.2f}s ({timestamps[0][0]/3600:.2f}h)")
        print(f"  Last subtitle ends: {timestamps[-1][1]:.2f}s ({timestamps[-1][1]/3600:.2f}h)")
        
        # Check if all timestamps are in a weird range
        if timestamps[0][0] > 3600:
            print(f"\n⚠️  WARNING: First subtitle starts at {timestamps[0][0]/3600:.2f} hours!")
            print(f"  This suggests timestamps are offset or incorrect.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_srt.py <srt_file>")
        sys.exit(1)
    
    analyze_srt(sys.argv[1])

