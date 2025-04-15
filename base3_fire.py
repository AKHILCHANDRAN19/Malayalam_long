# -*- coding: utf-8 -*- # Recommended for path handling if non-ASCII chars are possible
import cv2
import numpy as np
import os
import subprocess
import math # Needed for ceiling function in safe slicing

# =====================================================
# Configuration
# =====================================================

input_folder = "/storage/emulated/0/INPUT"  # adjust this folder path
# Ensure input_folder uses the correct separator for the OS if needed
# input_folder = os.path.normpath(input_folder)

num_segments = 4
clip_duration = 4.0        # seconds per clip
fps = 30                   # frames per second (fixed)
total_frames = int(clip_duration * fps)  # should be 120 frames for 4s at 30fps
output_width = 1280        # target width for 16:9 (e.g. YouTube)
output_height = 720        # target height for 16:9

# Define an equal margin to leave on all sides for the overlay image.
margin = int(0.10 * output_width)

# --- Animation Configuration ---
animation_duration_seconds = 0.4 # Duration of the slide animation
if animation_duration_seconds >= clip_duration:
    print("Warning: Animation duration is equal or greater than clip duration. Setting to 0.")
    animation_duration_seconds = 0
elif animation_duration_seconds < 0:
     print("Warning: Animation duration cannot be negative. Setting to 0.")
     animation_duration_seconds = 0
animation_duration_frames = int(animation_duration_seconds * fps)
# --- End Animation Configuration ---

sound_file = os.path.join(input_folder, "Whoos 3.mp3")
fire_overlay_video_path = os.path.join(input_folder, "Fire.mp4") # <<< Path to fire overlay video

# Create input folder if it doesn't exist
if not os.path.exists(input_folder):
    try:
        os.makedirs(input_folder)
        print(f"Created input folder: {input_folder}")
        print("Please place your 1.mp4, 1.png, ..., N.mp4, N.png, Fire.mp4, and the sound file in this folder.")
    except OSError as e:
        print(f"ERROR: Could not create input folder {input_folder}: {e}")
        exit() # Exit if we can't create the essential folder

# --- Check for essential files ---
essential_files_exist = True
if not os.path.exists(sound_file):
    print(f"ERROR: Sound file not found: {sound_file}")
    essential_files_exist = False
if not os.path.exists(fire_overlay_video_path):
    print(f"ERROR: Fire overlay video not found: {fire_overlay_video_path}")
    essential_files_exist = False

# Check for segment files (optional, loop will handle missing ones)
# for i in range(1, num_segments + 1):
#     if not os.path.exists(os.path.join(input_folder, f"{i}.mp4")): print(f"Warning: Missing {i}.mp4")
#     if not os.path.exists(os.path.join(input_folder, f"{i}.png")): print(f"Warning: Missing {i}.png")

if not essential_files_exist:
     print("Essential files missing. Please check paths and files. Exiting.")
     exit()
# --- End Check ---


# =====================================================
# Part 1: Create 4-Second Silent Clips with Overlay, Animation & Fire
# =====================================================
print("\n--- Part 1: Generating Silent Clips ---")
silent_clip_paths = []

for i in range(1, num_segments + 1):
    bg_video_path = os.path.join(input_folder, f"{i}.mp4")
    overlay_img_path = os.path.join(input_folder, f"{i}.png")
    silent_clip = os.path.join(input_folder, f"clip_{i}_silent.mp4")

    print(f"\nProcessing Segment {i}...")
    # --- Input File Checks ---
    if not os.path.exists(bg_video_path):
        print(f"ERROR: Background video not found: {bg_video_path}. Skipping segment {i}.")
        continue
    if not os.path.exists(overlay_img_path):
        print(f"ERROR: Overlay image not found: {overlay_img_path}. Skipping segment {i}.")
        continue

    # --- Open Video Captures ---
    cap = cv2.VideoCapture(bg_video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open background video: {bg_video_path}. Skipping segment {i}.")
        continue
    bg_fps = cap.get(cv2.CAP_PROP_FPS)
    bg_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f" Background: {os.path.basename(bg_video_path)} ({bg_frame_count} frames @ {bg_fps:.2f} fps)")


    # Open Fire Overlay Video Capture for this segment
    fire_cap = cv2.VideoCapture(fire_overlay_video_path)
    if not fire_cap.isOpened():
        print(f"ERROR: Cannot open fire overlay video: {fire_overlay_video_path}. Skipping segment {i}.")
        cap.release()
        continue
    fire_fps = fire_cap.get(cv2.CAP_PROP_FPS)
    fire_frame_count = int(fire_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fire_frame_count <= 0:
         print(f"ERROR: Fire overlay video {fire_overlay_video_path} has no frames or count is invalid. Skipping segment {i}.")
         cap.release()
         fire_cap.release()
         continue
    print(f" Fire Overlay: {os.path.basename(fire_overlay_video_path)} ({fire_frame_count} frames @ {fire_fps:.2f} fps)")


    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # H.264 alternative: *'avc1' or *'h264'
    out = cv2.VideoWriter(silent_clip, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        print(f"ERROR: Failed to open VideoWriter for {silent_clip}. Check permissions or codec support. Skipping segment {i}.")
        cap.release()
        fire_cap.release()
        continue

    # --- Load and Prepare Overlay Image ---
    overlay_orig = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    if overlay_orig is None:
        print(f"ERROR: Cannot load overlay image: {overlay_img_path}. Skipping segment {i}.")
        cap.release()
        fire_cap.release()
        out.release()
        continue

    if overlay_orig.shape[2] == 3: # Add alpha channel if missing
         overlay_orig = cv2.cvtColor(overlay_orig, cv2.COLOR_BGR2BGRA)

    # --- Resizing Logic ---
    max_width = output_width - 2 * margin
    max_height = output_height - 2 * margin
    orig_h, orig_w = overlay_orig.shape[:2]
    if orig_w == 0 or orig_h == 0:
        print(f"ERROR: Invalid overlay image dimensions for {overlay_img_path}. Skipping segment {i}.")
        cap.release()
        fire_cap.release()
        out.release()
        continue
    scale_factor = min(max_width / orig_w, max_height / orig_h)
    new_w = max(1, int(orig_w * scale_factor))
    new_h = max(1, int(orig_h * scale_factor))
    try:
        overlay_resized = cv2.resize(overlay_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"ERROR: Failed to resize overlay image {overlay_img_path}: {e}. Skipping segment {i}.")
        cap.release()
        fire_cap.release()
        out.release()
        continue
    print(f" Overlay Image: {os.path.basename(overlay_img_path)} resized to {new_w}x{new_h}")

    # --- Animation Setup ---
    target_x = margin + (max_width - new_w) // 2
    target_y = margin + (max_height - new_h) // 2
    animation_type = (i - 1) % 4
    start_x, start_y = target_x, target_y
    if animation_duration_frames > 0:
        anim_map = {0: "Top", 1: "Left", 2: "Bottom", 3: "Right"}
        print(f" Animating image slide from {anim_map[animation_type]} ({animation_duration_seconds}s)")
        if animation_type == 0: start_y = -new_h
        elif animation_type == 1: start_x = -new_w
        elif animation_type == 2: start_y = output_height
        elif animation_type == 3: start_x = output_width

    # --- Frame Processing Loop ---
    processed_frames = 0
    last_good_bg_frame = None
    fire_frame_index = -1 # To handle looping

    while processed_frames < total_frames:
        # --- Read Background Frame ---
        ret_bg, bg_frame = cap.read()
        if not ret_bg:
            if last_good_bg_frame is None:
                print(f"ERROR: Background video {bg_video_path} couldn't read first frame. Cannot create clip.")
                break # Exit loop for this segment
            else:
                # Pad with the last good frame
                bg_frame = last_good_bg_frame.copy()
        else:
             # Resize the background frame immediately
             bg_frame_resized = cv2.resize(bg_frame, (output_width, output_height))
             last_good_bg_frame = bg_frame_resized.copy() # Store last good one

        # --- Read Fire Frame (Looping) ---
        fire_frame_index += 1
        if fire_frame_index >= fire_frame_count:
            fire_frame_index = 0 # Reset index for looping
            fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Go back to start of fire video

        ret_fire, fire_frame = fire_cap.read()
        if not ret_fire:
             # If reading fails even after reset (e.g., video corrupted), try reading first frame again
             print(f"Warning: Failed to read frame {fire_frame_index} from fire video. Attempting reset/reread.")
             fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             ret_fire, fire_frame = fire_cap.read()
             if not ret_fire:
                 print("ERROR: Cannot read from fire video even after reset. Disabling fire overlay for remaining frames.")
                 fire_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8) # Use black frame
             else:
                  fire_frame_index = 0 # Successfully read first frame

        # Resize fire frame to output dimensions
        try:
            fire_frame_resized = cv2.resize(fire_frame, (output_width, output_height))
        except Exception as e:
            print(f"Warning: Could not resize fire frame {fire_frame_index}. Using black frame. Error: {e}")
            fire_frame_resized = np.zeros_like(last_good_bg_frame) # Use black frame of correct size


        # --- Start Compositing ---
        # Initialize composite frame with the (potentially padded) background
        composite = last_good_bg_frame.copy()

        # --- Calculate Image Overlay Position (Animation/Static) ---
        current_x = target_x
        current_y = target_y
        if processed_frames < animation_duration_frames and animation_duration_frames > 0:
            progress = min(1.0, (processed_frames + 1) / animation_duration_frames)
            current_x = int(start_x + progress * (target_x - start_x))
            current_y = int(start_y + progress * (target_y - start_y))

        # --- Place Image Overlay (Safe Placement with Alpha Blending) ---
        x1, y1 = current_x, current_y
        x2, y2 = current_x + new_w, current_y + new_h
        frame_x1, frame_y1 = max(x1, 0), max(y1, 0)
        frame_x2, frame_y2 = min(x2, output_width), min(y2, output_height)

        if frame_x1 < frame_x2 and frame_y1 < frame_y2:
            overlay_x1 = frame_x1 - x1
            overlay_y1 = frame_y1 - y1
            overlay_w = frame_x2 - frame_x1
            overlay_h = frame_y2 - frame_y1
            overlay_x2 = overlay_x1 + overlay_w
            overlay_y2 = overlay_y1 + overlay_h

            overlay_x1, overlay_y1 = max(0, overlay_x1), max(0, overlay_y1)
            overlay_x2, overlay_y2 = min(new_w, overlay_x2), min(new_h, overlay_y2)
            overlay_w, overlay_h = overlay_x2 - overlay_x1, overlay_y2 - overlay_y1
            frame_x2, frame_y2 = frame_x1 + overlay_w, frame_y1 + overlay_h

            if overlay_w > 0 and overlay_h > 0:
                try:
                    roi = composite[frame_y1:frame_y2, frame_x1:frame_x2]
                    overlay_part = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

                    if roi.shape[:2] == overlay_part.shape[:2]:
                        alpha = overlay_part[:, :, 3] / 255.0
                        overlay_colors = overlay_part[:, :, :3]
                        alpha_mask = cv2.cvtColor(alpha.astype(np.float32), cv2.COLOR_GRAY2BGR)
                        blended_roi = (overlay_colors * alpha_mask) + (roi * (1.0 - alpha_mask))
                        composite[frame_y1:frame_y2, frame_x1:frame_x2] = blended_roi.astype(np.uint8)
                except Exception as e:
                     if processed_frames % 30 == 0: # Print less often
                         print(f"Warning: Error during image overlay blending frame {processed_frames}: {e}")

        # --- Add Fire Overlay ---
        # Use cv2.add for additive blending (good for bright particles on dark bg)
        # Assumes fire_frame_resized is 3-channel BGR
        try:
             # Ensure composite is also 3 channels before adding
             if composite.shape[2] == 4:
                 composite = cv2.cvtColor(composite, cv2.COLOR_BGRA2BGR)

             composite = cv2.add(composite, fire_frame_resized)
             # Alternative: Screen blend mode (more complex, handles bright backgrounds better)
             # composite_norm = composite.astype(np.float32) / 255.0
             # fire_norm = fire_frame_resized.astype(np.float32) / 255.0
             # blended_norm = 1.0 - (1.0 - composite_norm) * (1.0 - fire_norm)
             # composite = np.clip(blended_norm * 255.0, 0, 255).astype(np.uint8)
        except Exception as e:
             if processed_frames % 30 == 0: # Print less often
                 print(f"Warning: Error during fire overlay blending frame {processed_frames}: {e}")


        # --- Write Frame ---
        out.write(composite)
        processed_frames += 1

    # --- End Frame Processing Loop for Segment ---
    print(f" Processed {processed_frames} frames for segment {i}.")
    cap.release()
    fire_cap.release()
    out.release()

    if processed_frames == total_frames:
        print(f" Silent clip with overlays saved: {silent_clip}")
        silent_clip_paths.append(silent_clip)
    else:
        print(f" Failed or incomplete silent clip for segment {i} ({processed_frames}/{total_frames}). Not adding to list.")
        if os.path.exists(silent_clip):
            try: os.remove(silent_clip)
            except OSError as e: print(f" Error removing incomplete file {silent_clip}: {e}")


# =====================================================
# Part 2: Add Sound Effect Using FFmpeg
# =====================================================
print("\n--- Part 2: Adding Audio ---")
final_clip_paths = []
ffmpeg_executable = "ffmpeg" # Or provide full path

# Check FFmpeg availability once
try:
     result = subprocess.run([ffmpeg_executable, "-version"], check=True, capture_output=True, text=True)
     print("FFmpeg found:", result.stdout.split('\n', 1)[0]) # Print first line of version info
except (FileNotFoundError, subprocess.CalledProcessError) as e:
     print(f"ERROR: '{ffmpeg_executable}' command not found or not executable: {e}")
     print("Make sure FFmpeg is installed and in your system's PATH. Cannot proceed.")
     # Clear silent clips list to prevent further steps
     silent_clip_paths = []


if not silent_clip_paths:
     print("No silent clips were successfully created in Part 1 or FFmpeg not found. Skipping audio addition.")
else:
    for idx, silent_clip in enumerate(silent_clip_paths):
        try: # Extract original segment number from filename for consistent output naming
            original_segment_num = int(os.path.basename(silent_clip).split('_')[1])
        except (IndexError, ValueError):
            original_segment_num = idx + 1 # Fallback if parsing fails (use 1-based index)
            print(f"Warning: Could not parse segment number from {silent_clip}. Using index {original_segment_num}.")

        final_clip = os.path.join(input_folder, f"final_{original_segment_num}.mp4")

        ffmpeg_cmd = [
            ffmpeg_executable, "-y", # Overwrite output without asking
            "-i", silent_clip,     # Input 0: Silent video clip
            "-i", sound_file,      # Input 1: Sound effect
            "-filter_complex",
            # Trim sound to 1s, generate 3s silence, concatenate to match 4s video
            "[1:a]atrim=duration=1,asetpts=PTS-STARTPTS[sound]; " # Take 1s from input 1 audio
            "anullsrc=r=44100:cl=stereo,atrim=duration=3,asetpts=PTS-STARTPTS[silence]; " # Generate 3s stereo silence
            "[sound][silence]concat=n=2:v=0:a=1[a]", # Concatenate sound and silence
            "-map", "0:v:0",       # Map video from input 0
            "-map", "[a]",         # Map the filtered audio stream
            "-c:v", "copy",        # Copy video codec (fast)
            "-c:a", "aac",         # Encode audio to AAC
            "-b:a", "128k",        # Set audio bitrate
            "-shortest",           # Ensure output duration matches shortest input (the 4s video)
            final_clip
        ]
        print(f" Adding sound to clip {original_segment_num} -> {os.path.basename(final_clip)}...")
        try:
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            print(f"  Success.")
            final_clip_paths.append(final_clip)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR running FFmpeg for clip {original_segment_num}:")
            print(f"  Command: {' '.join(e.cmd)}")
            print(f"  Stderr: {e.stderr}")
            if os.path.exists(final_clip): os.remove(final_clip) # Clean up failed output


# =====================================================
# Part 3: Combine Final Clips (Video Only) Using OpenCV
# =====================================================
print("\n--- Part 3: Combining Video Tracks ---")
combined_video_silent = os.path.join(input_folder, "combined_video_silent.mp4")
combiner_initialized = False
out_combined = None # Define scope outside try block

if not final_clip_paths:
    print("No final clips with audio were generated. Skipping video combination.")
else:
    try:
        # Sort final clips by number in filename to ensure correct concatenation order
        final_clip_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        print(" Combining video from clips:", [os.path.basename(p) for p in final_clip_paths])

        first_clip_path = final_clip_paths[0]
        cap_check = cv2.VideoCapture(first_clip_path)
        if not cap_check.isOpened():
            raise IOError(f"Could not open first final clip {first_clip_path} to get parameters.")

        comb_width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        comb_height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        comb_fps = cap_check.get(cv2.CAP_PROP_FPS)
        if not (0 < comb_fps < 200):
             print(f"Warning: Read invalid FPS ({comb_fps}) from {first_clip_path}. Falling back to configured fps ({fps}).")
             comb_fps = fps
        cap_check.release()

        print(f" Combining videos with parameters: {comb_width}x{comb_height} @ {comb_fps:.2f} fps")
        fourcc_comb = cv2.VideoWriter_fourcc(*"mp4v")
        out_combined = cv2.VideoWriter(combined_video_silent, fourcc_comb, comb_fps, (comb_width, comb_height))
        if not out_combined.isOpened():
             raise IOError(f"Could not open VideoWriter for {combined_video_silent}")
        combiner_initialized = True

        total_combined_frames = 0
        for clip_path in final_clip_paths:
            clip_frame_count = 0
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                print(f"ERROR: Cannot open {clip_path} for combination. Skipping.")
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w != comb_width or h != comb_height:
                print(f"ERROR: Dimension mismatch in {os.path.basename(clip_path)} ({w}x{h}). Skipping.")
                cap.release()
                continue

            while True:
                ret, frame = cap.read()
                if not ret: break
                out_combined.write(frame)
                clip_frame_count += 1
            cap.release()
            total_combined_frames += clip_frame_count
            print(f"  Appended {clip_frame_count} frames from {os.path.basename(clip_path)}")

        print(f" Combined silent video saved: {combined_video_silent} ({total_combined_frames} frames)")

    except Exception as e:
        print(f"Error during video combination: {e}")
        combiner_initialized = False # Mark as failed
        combined_video_silent = None # Prevent further use
    finally:
         if out_combined is not None and out_combined.isOpened():
              out_combined.release() # Ensure writer is released


# =====================================================
# Part 4: Extract Audio, Concatenate, Merge
# =====================================================
print("\n--- Part 4: Processing Audio and Final Merge ---")

if not final_clip_paths:
    print("No final clips available. Skipping audio extraction and merge.")
elif not combiner_initialized or combined_video_silent is None or not os.path.exists(combined_video_silent):
     print("Combined silent video was not created successfully. Skipping audio extraction and merge.")
else:
    audio_files = []
    audio_list_file = os.path.join(input_folder, "audio_list.txt")
    combined_audio = os.path.join(input_folder, "combined_audio.aac")
    final_output = os.path.join(input_folder, "final_output.mp4")
    extraction_successful = True

    print(" Extracting audio streams...")
    for idx, clip_path in enumerate(final_clip_paths): # Use sorted list from Part 3
        try:
            original_segment_num = int(os.path.basename(clip_path).split('_')[1].split('.')[0])
        except:
            original_segment_num = idx + 1 # Fallback

        audio_file = os.path.join(input_folder, f"audio_{original_segment_num}.aac")
        extract_cmd = [ffmpeg_executable, "-y", "-i", clip_path, "-vn", "-acodec", "copy", audio_file]
        try:
            result = subprocess.run(extract_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                 audio_files.append(audio_file)
                 print(f"  Extracted audio from clip {original_segment_num} -> {os.path.basename(audio_file)}")
            else:
                 print(f"  ERROR: Audio extraction from {clip_path} failed or produced empty file.")
                 extraction_successful = False; break
        except subprocess.CalledProcessError as e:
            print(f"  ERROR extracting audio from {clip_path}:")
            print(f"  Command: {' '.join(e.cmd)}")
            print(f"  Stderr: {e.stderr}")
            extraction_successful = False; break # Stop extraction if one fails

    # Proceed only if all audio extractions were successful
    if extraction_successful and len(audio_files) == len(final_clip_paths):
        try:
            # Create audio list file (using sorted audio_files list)
            with open(audio_list_file, "w", encoding='utf-8') as f:
                for af in audio_files: # Should already be sorted by previous step
                    f.write(f"file '{os.path.basename(af)}'\n")
            print(f" Created audio list file: {audio_list_file}")

            # Concatenate audio files
            concat_audio_cmd = [ffmpeg_executable, "-y", "-f", "concat", "-safe", "0", "-i", audio_list_file, "-c", "copy", combined_audio]
            print(" Concatenating audio files...")
            result = subprocess.run(concat_audio_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            print(f"  Combined audio saved: {combined_audio}")

            # Merge final video and audio
            merge_cmd = [ffmpeg_executable, "-y", "-i", combined_video_silent, "-i", combined_audio, "-c:v", "copy", "-c:a", "copy", "-shortest", "-map", "0:v:0", "-map", "1:a:0", final_output]
            print(" Merging final video and audio...")
            result = subprocess.run(merge_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            expected_duration_seconds = len(final_clip_paths) * clip_duration
            print(f"\n Final video (expected duration: {expected_duration_seconds:.1f}s) saved as: {final_output}")

            # --- Cleanup ---
            print("\n Cleaning up intermediate files...")
            cleanup_files = silent_clip_paths + final_clip_paths + audio_files
            if os.path.exists(audio_list_file): cleanup_files.append(audio_list_file)
            if os.path.exists(combined_video_silent): cleanup_files.append(combined_video_silent)
            if os.path.exists(combined_audio): cleanup_files.append(combined_audio)
            removed_count = 0
            for f in cleanup_files:
                 if os.path.exists(f):
                     try: os.remove(f); removed_count += 1
                     except OSError as e: print(f"  Warning: Could not remove {f}: {e}")
            print(f" Cleanup complete (removed {removed_count} files).")
            # --- End Cleanup ---

        except subprocess.CalledProcessError as e:
            print(f"ERROR running FFmpeg during audio concatenation or final merge:")
            print(f"  Command: {' '.join(e.cmd)}")
            print(f"  Stderr: {e.stderr}")
        except Exception as e:
             print(f"An error occurred during audio processing or final merge: {e}")
    elif not extraction_successful:
        print("Skipping audio concatenation and final merge due to audio extraction errors.")
    else: # Mismatch in count
        print(f"Skipping audio concatenation and final merge. Expected {len(final_clip_paths)} audio files, but found {len(audio_files)}.")


print("\n--- Script finished ---")
