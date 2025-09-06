"""Periodicity strength example - measures strength of periodic patterns."""

# to run python videokurt/smoke_tests/feat_test_periodicity_strength.py
from videokurt import VideoKurt
import numpy as np

# Create VideoKurt instance
vk = VideoKurt()

# Configure for periodicity analysis
vk.configure(frame_step=1, resolution_scale=0.5)

# Add periodicity_strength feature
vk.add_feature('periodicity_strength',
               min_frequency=0.1,  # Minimum frequency (Hz)
               max_frequency=10.0)  # Maximum frequency (Hz)

print("Analyzing periodicity strength in video...")
print("This measures how strongly periodic/repetitive the content is")
print()

results = vk.analyze('sample_recording.MP4')

# Get periodicity results
periodicity = results.features['periodicity_strength'].data

print("\nPeriodicity Strength Results:")
print(f"  Data type: {type(periodicity)}")

if isinstance(periodicity, dict):
    # Dominant period
    if 'dominant_period' in periodicity:
        print(f"\n  Dominant period: {periodicity['dominant_period']} frames")
        
        # Convert to time if we know frame rate (assume 30 fps)
        fps = 30
        period_seconds = periodicity['dominant_period'] / fps
        print(f"  Period duration: {period_seconds:.2f} seconds")
    
    # Periodicity strength
    if 'strength' in periodicity:
        strength = periodicity['strength']
        print(f"\n  Periodicity strength: {strength:.3f}")
        
        if strength > 0.8:
            print("  Pattern: Very strong periodicity")
        elif strength > 0.5:
            print("  Pattern: Moderate periodicity")
        elif strength > 0.3:
            print("  Pattern: Weak periodicity")
        else:
            print("  Pattern: No clear periodicity")
    
    # Detected periods
    if 'detected_periods' in periodicity:
        periods = periodicity['detected_periods']
        if periods:
            print(f"\n  All detected periods:")
            for period_info in periods[:5]:
                period = period_info.get('period', 0)
                strength = period_info.get('strength', 0)
                confidence = period_info.get('confidence', 0)
                
                print(f"    Period {period} frames:")
                print(f"      Strength: {strength:.3f}")
                print(f"      Confidence: {confidence:.1%}")
    
    # Autocorrelation analysis
    if 'autocorrelation' in periodicity:
        autocorr = periodicity['autocorrelation']
        if isinstance(autocorr, (list, np.ndarray)) and len(autocorr) > 0:
            print(f"\n  Autocorrelation Analysis:")
            print(f"    Max correlation: {np.max(autocorr):.3f}")
            print(f"    Mean correlation: {np.mean(autocorr):.3f}")
            
            # Find peaks
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    if autocorr[i] > 0.3:  # Threshold
                        peaks.append(i)
            
            if peaks:
                print(f"    Correlation peaks at lags: {peaks[:5]}")
    
    # Frequency spectrum
    if 'frequency_spectrum' in periodicity:
        spectrum = periodicity['frequency_spectrum']
        if 'dominant_frequency' in spectrum:
            print(f"\n  Frequency Analysis:")
            print(f"    Dominant frequency: {spectrum['dominant_frequency']:.3f} Hz")
            
            if 'harmonics' in spectrum:
                print(f"    Harmonics detected: {spectrum['harmonics'][:3]}")
            
            if 'spectral_energy' in spectrum:
                print(f"    Spectral energy: {spectrum['spectral_energy']:.3f}")

# Analyze pattern types
print("\n" + "="*50)
print("Pattern Type Analysis:")

if isinstance(periodicity, dict):
    strength = periodicity.get('strength', 0)
    period = periodicity.get('dominant_period', 0)
    
    if strength > 0.5 and period > 0:
        # Classify by period length
        if period < 10:
            print(f"  Pattern: Fast repetition ({period} frames)")
            print("  Interpretation: Rapid cyclic behavior or flashing")
        elif period < 30:
            print(f"  Pattern: Medium-speed repetition ({period} frames)")
            print("  Interpretation: Regular animation or UI cycling")
        elif period < 90:
            print(f"  Pattern: Slow repetition ({period} frames)")
            print("  Interpretation: Slow loops or gradual cycles")
        else:
            print(f"  Pattern: Very slow repetition ({period} frames)")
            print("  Interpretation: Long-form patterns or scene loops")
        
        # Check for multiple periods
        if 'detected_periods' in periodicity:
            periods = periodicity['detected_periods']
            if len(periods) > 1:
                print(f"\n  Multiple periodicities detected:")
                print("  Interpretation: Complex pattern with nested cycles")
    else:
        print("  Pattern: Non-periodic")
        print("  Interpretation: Irregular or non-repeating content")

# Combine with other temporal features
print("\n" + "="*50)
print("Combined Temporal Analysis:")

vk2 = VideoKurt()
vk2.configure(frame_step=1, resolution_scale=0.5)

# Add multiple temporal features
vk2.add_feature('periodicity_strength')
vk2.add_feature('temporal_activity_patterns', window_size=30)

print("\nProcessing with combined temporal features...")
results2 = vk2.analyze('sample_recording.MP4')

periodicity = results2.features['periodicity_strength'].data
temporal = results2.features['temporal_activity_patterns'].data

if periodicity and temporal:
    print(f"\n  Combined Analysis:")
    
    period_strength = periodicity.get('strength', 0)
    activity_variance = temporal.get('activity_variance', 0)
    
    if period_strength > 0.5 and activity_variance > 0:
        print(f"    Periodicity: {period_strength:.3f}")
        print(f"    Activity variance: {activity_variance:.3f}")
        
        if activity_variance < 0.1:
            print("    Pattern: Periodic with stable amplitude")
        else:
            print("    Pattern: Periodic with varying intensity")
    
    # Check phase alignment
    if 'pattern_changes' in temporal:
        changes = temporal['pattern_changes']
        period = periodicity.get('dominant_period', 0)
        
        if changes and period > 0:
            # Check if pattern changes align with period
            change_frames = [c['frame'] for c in changes]
            if change_frames:
                aligned = sum(1 for f in change_frames if f % period < 2)
                alignment = aligned / len(change_frames) if change_frames else 0
                
                print(f"\n    Pattern change alignment: {alignment:.1%}")
                if alignment > 0.7:
                    print("    Changes align with periodic cycle")

# Detect specific patterns
print("\n" + "="*50)
print("Specific Pattern Detection:")

if isinstance(periodicity, dict):
    period = periodicity.get('dominant_period', 0)
    strength = periodicity.get('strength', 0)
    
    # Common pattern detection
    fps = 30  # Assumed frame rate
    
    if strength > 0.5:
        if 28 <= period <= 32:  # ~1 second
            print("  Detected: ~1 second cycle")
            print("  Possible: Progress indicator or timer")
        elif 58 <= period <= 62:  # ~2 seconds
            print("  Detected: ~2 second cycle")
            print("  Possible: Breathing animation or pulse effect")
        elif 88 <= period <= 92:  # ~3 seconds
            print("  Detected: ~3 second cycle")
            print("  Possible: Notification cycle or slideshow")
        elif 148 <= period <= 152:  # ~5 seconds
            print("  Detected: ~5 second cycle")
            print("  Possible: Screen saver or demo loop")
        
        # Check for beat patterns (music visualization)
        if 12 <= period <= 18:  # ~0.5 seconds (120 BPM)
            print("\n  Possible beat pattern detected")
            print("  Could be: Music visualization or rhythmic animation")
        
        # Check for scrolling patterns
        if period < 5 and strength > 0.7:
            print("\n  High-frequency repetition detected")
            print("  Could be: Scrolling text or rapid animation")

print("\n✓ Periodicity strength analysis complete")