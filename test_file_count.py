#!/usr/bin/env python3
"""
Quick test to verify the pipeline will generate 1000+ audio files.
"""

# Configuration
TOP_N_HOTSPOTS = 50
EVENTS_PER_HOTSPOT = 5
RECIPES_PER_EVENT = 4

# Calculate expected outputs
total_files = TOP_N_HOTSPOTS * EVENTS_PER_HOTSPOT * RECIPES_PER_EVENT

print("=" * 70)
print("ACOUSTIC SHIELD - FILE GENERATION ESTIMATE")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  • Top hotspots: {TOP_N_HOTSPOTS}")
print(f"  • Events per hotspot: {EVENTS_PER_HOTSPOT}")
print(f"  • Recipe variations per event: {RECIPES_PER_EVENT}")
print(f"\nCalculation:")
print(f"  {TOP_N_HOTSPOTS} hotspots × {EVENTS_PER_HOTSPOT} events × {RECIPES_PER_EVENT} recipes = {total_files} audio files")
print(f"\n✓ Target met: {total_files >= 1000}")
print(f"  Expected output: {total_files:,} WAV files")
print("=" * 70)

# Class distribution estimate (based on typical distribution)
# Assuming: 30% Normal, 30% TireSkid, 25% EmergencyBraking, 15% CollisionImminent
distributions = {
    'Normal': 0.30,
    'TireSkid': 0.30,
    'EmergencyBraking': 0.25,
    'CollisionImminent': 0.15
}

print(f"\nEstimated class distribution:")
for class_name, pct in distributions.items():
    count = int(total_files * pct)
    print(f"  • {class_name}: ~{count:,} files ({pct*100:.0f}%)")

print("\n" + "=" * 70)
