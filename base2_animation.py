import cv2
import numpy as np
import os
import subprocess
import math # Needed for ceiling function in safe slicing

# =====================================================
# Configuration
# =====================================================

input_folder = "/storage/emulated/0/INPUT"  # adjust this folder path
num_segments = 4
clip_duration = 4.0        # seconds per clip
fps = 30                   # frames per second (fixed)
total_frames = int(clip_duration * fps)  # should be 120 frames for 4s at 30fps
output_width = 1280        # target width for 16:9 (e.g. YouTube)
output_height = 720        # target height for 16:9

# Define an equal margin to leave on all sides for the overlay image.
# This margin is in pixels. (Adjust as needed; here we use 10% of width.)
margin = int(0.10 * output_width)

# --- Animation Configuration ---
# <<< CHANGE MADE HERE >>>
animation_duration_seconds = 0.4 # Duration of the slide animation (must be less than clip_duration)
# <<< END OF CHANGE >>>

if animation_duration_seconds >= clip_duration:
    print("Warning: Animation duration is equal or greater than clip duration. Setting to 0.")
    animation_duration_seconds = 0
# Ensure animation duration is not negative
elif animation_duration_seconds < 0:
     print("Warning: Animation duration cannot be negative. Setting to 0.")
     animation_duration_seconds = 0


animation_duration_frames = int(animation_duration_seconds * fps)
# --- End Animation Configuration ---

sound_file = os.path.join(input_folder, "Whoos 3.mp3")

# Create input folder if it doesn't exist (for demonstration)
if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    print(f"Created input folder: {input_folder}")
    print("Please place your 1.mp4, 1.png, 2.mp4, 2.png, ..., N.mp4, N.png and the sound file in this folder.")

# =====================================================
# Part 1: Create 4-Second Silent Clips with Overlay & Animation
# =====================================================
# For each segment:
# - Read a 4-second portion from the background video.
# - Resize the background to 1280x720.
# - Load the corresponding overlay image and scale it so that it fits inside
#   the region defined by (output_width - 2*margin, output_height - 2*margin),
#   preserving its aspect ratio.
# - Animate the overlay sliding in during the first 'animation_duration_frames'.
# - Center the overlay within that region after the animation.
# - Write the composite frames to a silent video file ("clip_i_silent.mp4").

silent_clip_paths = []

for i in range(1, num_segments + 1):
    bg_video_path = os.path.join(input_folder, f"{i}.mp4")
    overlay_img_path = os.path.join(input_folder, f"{i}.png")
    silent_clip = os.path.join(input_folder, f"clip_{i}_silent.mp4")

    if not os.path.exists(bg_video_path):
        print(f"ERROR: Background video not found: {bg_video_path}. Skipping segment {i}.")
        continue
    if not os.path.exists(overlay_img_path):
        print(f"ERROR: Overlay image not found: {overlay_img_path}. Skipping segment {i}.")
        continue

    cap = cv2.VideoCapture(bg_video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open background video: {bg_video_path}. Skipping segment {i}.")
        continue

    # Create VideoWriter for silent clip (4 seconds exactly)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(silent_clip, fourcc, fps, (output_width, output_height))

    # Load overlay image (in color, assuming it might have transparency later)
    # Use IMREAD_UNCHANGED to keep alpha channel if present, otherwise load as color
    overlay_orig = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    if overlay_orig is None:
        print(f"ERROR: Cannot load overlay image: {overlay_img_path}. Skipping segment {i}.")
        cap.release()
        out.release() # Release writer if overlay fails
        continue

    # Handle case where image might not have alpha channel
    if overlay_orig.shape[2] == 3: # Add alpha channel if missing
         overlay_orig = cv2.cvtColor(overlay_orig, cv2.COLOR_BGR2BGRA)

    # --- Resizing Logic ---
    max_width = output_width - 2 * margin
    max_height = output_height - 2 * margin
    orig_h, orig_w = overlay_orig.shape[:2]

    # Prevent division by zero if image dimensions are invalid
    if orig_w == 0 or orig_h == 0:
        print(f"ERROR: Invalid overlay image dimensions (0 width or height) for {overlay_img_path}. Skipping segment {i}.")
        cap.release()
        out.release()
        continue

    scale_factor = min(max_width / orig_w, max_height / orig_h)
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)

    # Ensure dimensions are at least 1 pixel
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    try:
        overlay_resized = cv2.resize(overlay_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"ERROR: Failed to resize overlay image {overlay_img_path} to ({new_w}, {new_h}): {e}. Skipping segment {i}.")
        cap.release()
        out.release()
        continue
    # --- End Resizing ---

    # --- Animation Setup ---
    # Final target position (centered within margins)
    target_x = margin + (max_width - new_w) // 2
    target_y = margin + (max_height - new_h) // 2

    # Determine animation type and starting position based on segment index
    animation_type = (i - 1) % 4  # Cycle through 0, 1, 2, 3
    start_x, start_y = target_x, target_y # Initialize with target

    if animation_duration_frames > 0:
        if animation_type == 0:  # Slide from Top
            start_y = -new_h # Start just above the frame
            print(f"Segment {i}: Animating slide from Top ({animation_duration_seconds}s)")
        elif animation_type == 1:  # Slide from Left
            start_x = -new_w # Start just left of the frame
            print(f"Segment {i}: Animating slide from Left ({animation_duration_seconds}s)")
        elif animation_type == 2:  # Slide from Bottom
            start_y = output_height # Start fully below the frame
            print(f"Segment {i}: Animating slide from Bottom ({animation_duration_seconds}s)")
        elif animation_type == 3:  # Slide from Right
            start_x = output_width # Start fully right of the frame
            print(f"Segment {i}: Animating slide from Right ({animation_duration_seconds}s)")
    # --- End Animation Setup ---

    frame_count = 0 # Total frames attempted to read
    processed_frames = 0 # Keep track of frames actually processed and written
    last_good_frame = None # To hold the last successfully read frame for padding

    while processed_frames < total_frames:
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            # Handle end of video: Pad with the last good frame if available
            if last_good_frame is None:
                print(f"Warning: Background video {bg_video_path} ended before the first frame. Cannot create clip for segment {i}.")
                break # Exit the loop for this segment
            else:
                if frame_count <= total_frames + 1: # Only print warning once or twice
                     print(f"Warning: Background video {bg_video_path} ended at frame {frame_count-1}. Padding remaining {total_frames - processed_frames} frames with last good frame.")
                frame = last_good_frame.copy() # Use the last good frame for padding
        else:
             # Resize the background frame
             bg_frame_resized = cv2.resize(frame, (output_width, output_height))
             last_good_frame = bg_frame_resized.copy() # Store last good frame

        # Create composite frame for this iteration
        composite = last_good_frame.copy() # Start with the (potentially padded) background

        # --- Calculate Current Overlay Position ---
        current_x = target_x
        current_y = target_y

        # Use processed_frames for animation timing
        if processed_frames < animation_duration_frames and animation_duration_frames > 0:
            # Animation is in progress
            # Ensure division by zero doesn't happen if duration is 0 frames
            progress = (processed_frames + 1) / animation_duration_frames
            progress = min(1.0, progress) # Clamp progress just in case

            # Ease-out interpolation (optional, makes animation smoother)
            # progress = 1 - (1 - progress)**2 # Example: Quadratic ease-out

            # Linear Interpolate position
            current_x = int(start_x + progress * (target_x - start_x))
            current_y = int(start_y + progress * (target_y - start_y))
        # --- End Position Calculation ---


        # --- Safe Overlay Placement (Handles Animation Clipping & Transparency) ---
        # Determine the bounding box of the overlay in the current frame coordinates
        x1, y1 = current_x, current_y
        x2, y2 = current_x + new_w, current_y + new_h

        # Calculate the intersection of the overlay bounding box and the frame boundaries
        frame_x1 = max(x1, 0)
        frame_y1 = max(y1, 0)
        frame_x2 = min(x2, output_width)
        frame_y2 = min(y2, output_height)

        # If there is an intersection (visible part of the overlay)
        if frame_x1 < frame_x2 and frame_y1 < frame_y2:
            # Calculate the corresponding region within the overlay_resized image
            # Use math.ceil for end coordinates in slicing to avoid off-by-one
            overlay_x1 = frame_x1 - x1
            overlay_y1 = frame_y1 - y1
            overlay_w = frame_x2 - frame_x1
            overlay_h = frame_y2 - frame_y1
            overlay_x2 = overlay_x1 + overlay_w
            overlay_y2 = overlay_y1 + overlay_h

            # Ensure calculated overlay region indices are within bounds
            overlay_x1 = max(0, overlay_x1)
            overlay_y1 = max(0, overlay_y1)
            overlay_x2 = min(new_w, overlay_x2) # Clip to overlay width
            overlay_y2 = min(new_h, overlay_y2) # Clip to overlay height

            # Recalculate width/height after clipping
            overlay_w = overlay_x2 - overlay_x1
            overlay_h = overlay_y2 - overlay_y1

            # Ensure frame region matches overlay region size *after* clipping
            frame_x2 = frame_x1 + overlay_w
            frame_y2 = frame_y1 + overlay_h

            # Proceed only if dimensions are valid (> 0)
            if overlay_w > 0 and overlay_h > 0:
                try:
                    # Get the region of interest (ROI) from the background frame
                    roi = composite[frame_y1:frame_y2, frame_x1:frame_x2]

                    # Get the corresponding part of the overlay image
                    overlay_part = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

                    # Final check for shape match before blending (should usually match now)
                    if roi.shape[:2] == overlay_part.shape[:2]:
                        # Separate alpha channel and color channels
                        alpha = overlay_part[:, :, 3] / 255.0  # Normalize alpha to 0.0-1.0
                        overlay_colors = overlay_part[:, :, :3]

                        # Expand alpha channel to 3 color channels for broadcasting
                        alpha_mask = cv2.cvtColor(alpha.astype(np.float32), cv2.COLOR_GRAY2BGR)

                        # Blend overlay onto the ROI
                        blended_roi = (overlay_colors * alpha_mask) + (roi * (1.0 - alpha_mask))

                        # Place the blended ROI back into the composite frame
                        composite[frame_y1:frame_y2, frame_x1:frame_x2] = blended_roi.astype(np.uint8)
                    else:
                         # This warning should be less frequent now with corrected slicing
                         if processed_frames % 10 == 0: # Print warning less often
                             print(f"Warning: Dimension mismatch during blending frame {processed_frames}. ROI: {roi.shape}, OverlayPart: {overlay_part.shape}")
                except IndexError as e:
                    print(f"Error during overlay slicing/blending frame {processed_frames}: {e}")
                    print(f"Frame region: y={frame_y1}:{frame_y2}, x={frame_x1}:{frame_x2}")
                    print(f"Overlay region: y={overlay_y1}:{overlay_y2}, x={overlay_x1}:{overlay_x2}")
                except Exception as e:
                     print(f"Unexpected error during overlay blending frame {processed_frames}: {e}")

        # --- End Safe Overlay Placement ---

        out.write(composite)
        processed_frames += 1 # Increment only after successfully processing and writing a frame

    # --- End Frame Processing Loop ---

    cap.release()
    out.release()
    if processed_frames == total_frames: # Only count as successful if all frames written
        print(f"Silent clip for segment {i} saved as: {silent_clip} ({processed_frames} frames)")
        silent_clip_paths.append(silent_clip)
    else:
        print(f"Failed or incomplete silent clip for segment {i} ({processed_frames}/{total_frames} frames). Not adding to list.")
        # Optionally remove the incomplete file
        if os.path.exists(silent_clip):
            try:
                os.remove(silent_clip)
                print(f"Removed incomplete file: {silent_clip}")
            except OSError as e:
                print(f"Error removing incomplete file {silent_clip}: {e}")


# =====================================================
# Part 2: Add a Single Sound Effect at the Beginning Using FFmpeg
# =====================================================
# (No changes needed in this part)
final_clip_paths = []
if not os.path.exists(sound_file):
    print(f"ERROR: Sound file not found: {sound_file}. Skipping audio addition.")
elif not silent_clip_paths: # Check if any silent clips were successfully created
     print("No silent clips were successfully created in Part 1. Skipping audio addition.")
else:
    print("\nAdding audio to silent clips...")
    for idx, silent_clip in enumerate(silent_clip_paths, start=1):
        # Determine index based on the original segment number embedded in the filename
        try:
            original_segment_num = int(os.path.basename(silent_clip).split('_')[1])
        except (IndexError, ValueError):
            print(f"Warning: Could not determine original segment number from {silent_clip}. Using sequential index {idx}.")
            original_segment_num = idx # Fallback

        final_clip = os.path.join(input_folder, f"final_{original_segment_num}.mp4")
        # Check if silent clip exists before trying to process it (redundant due to list check, but safe)
        if not os.path.exists(silent_clip):
            print(f"ERROR: Silent clip {silent_clip} not found. Skipping.")
            continue

        # Check FFmpeg availability once
        ffmpeg_executable = "ffmpeg" # Or provide full path if necessary
        try:
             subprocess.run([ffmpeg_executable, "-version"], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
             print(f"ERROR: '{ffmpeg_executable}' command not found or not executable.")
             print("Make sure FFmpeg is installed and in your system's PATH.")
             break # Stop processing if ffmpeg isn't found

        ffmpeg_cmd = [
            ffmpeg_executable, "-y",
            "-i", silent_clip,
            "-i", sound_file,
            "-filter_complex",
            # Ensure audio duration matches video (4s)
            # Trim sound to 1s, generate 3s silence, concat
            "[1]atrim=duration=1,asetpts=PTS-STARTPTS[sound]; anullsrc=r=44100:cl=stereo,atrim=duration=3,asetpts=PTS-STARTPTS[silence]; [sound][silence]concat=n=2:v=0:a=1[a]",
            "-map", "0:v:0", # Map video from first input (silent clip)
            "-map", "[a]",   # Map filtered audio
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-c:a", "aac",   # Encode audio to AAC
            "-b:a", "128k",  # Set audio bitrate (optional but good practice)
            "-shortest",     # Finish encoding when the shortest input ends (video is 4s)
            final_clip
        ]
        print(f"Adding sound effect to clip {original_segment_num} (sound plays for 1s at start)...")
        try:
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True) # Capture output as text
            print(f"Final clip with audio saved as: {final_clip}")
            final_clip_paths.append(final_clip)
        except subprocess.CalledProcessError as e:
            print(f"ERROR running FFmpeg for clip {original_segment_num}:")
            print("Command:", " ".join(e.cmd))
            print("Stderr:", e.stderr)
            # print("Stdout:", e.stdout) # Usually less helpful on error
            # Optionally remove failed output file
            if os.path.exists(final_clip): os.remove(final_clip)


# =====================================================
# Part 3: Combine Final Clips (Video Only) Using OpenCV
# =====================================================
# (No changes needed in this part, but added existence checks and parameter fetching)
combined_video_silent = os.path.join(input_folder, "combined_video_silent.mp4")
combiner_initialized = False

# Check if there are any final clips to combine
if not final_clip_paths:
    print("\nNo final clips with audio were generated. Skipping video combination.")
else:
    print("\nCombining video tracks...")
    # Use the dimensions and fps from the first successfully created clip
    # This assumes all clips were created with the same settings
    try:
        first_clip_path = final_clip_paths[0]
        cap_check = cv2.VideoCapture(first_clip_path)
        if cap_check.isOpened():
            comb_width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
            comb_height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
            comb_fps = cap_check.get(cv2.CAP_PROP_FPS)
            # Ensure fps is reasonable, fallback to config if needed
            if not (0 < comb_fps < 200): # Basic sanity check for FPS
                 print(f"Warning: Read invalid FPS ({comb_fps}) from {first_clip_path}. Falling back to configured fps ({fps}).")
                 comb_fps = fps

            cap_check.release()

            print(f"Combining videos with parameters: {comb_width}x{comb_height} @ {comb_fps:.2f} fps")
            fourcc_comb = cv2.VideoWriter_fourcc(*"mp4v") # or "avc1" for H.264 if supported
            out_combined = cv2.VideoWriter(combined_video_silent, fourcc_comb, comb_fps, (comb_width, comb_height))
            combiner_initialized = True

            total_combined_frames = 0
            for clip_path in final_clip_paths:
                clip_frame_count = 0
                cap = cv2.VideoCapture(clip_path)
                if not cap.isOpened():
                    print(f"ERROR: Cannot open final clip for combination: {clip_path}")
                    continue # Skip this clip

                # Verify dimensions match the first clip
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w != comb_width or h != comb_height:
                    print(f"ERROR: Dimension mismatch in {clip_path} ({w}x{h}) vs expected ({comb_width}x{comb_height}). Skipping.")
                    cap.release()
                    continue # Skip this clip

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out_combined.write(frame)
                    clip_frame_count += 1
                cap.release() # Release inside the loop
                total_combined_frames += clip_frame_count
                print(f" Appended {clip_frame_count} frames from {os.path.basename(clip_path)}")

            out_combined.release() # Release after the loop
            print(f"Combined silent video saved as: {combined_video_silent} ({total_combined_frames} frames)")
        else:
            print(f"ERROR: Could not open first final clip {first_clip_path} to get parameters. Skipping video combination.")
            combined_video_silent = None # Indicate failure
    except Exception as e:
        print(f"Error during video combination setup or processing: {e}")
        if combiner_initialized and out_combined.isOpened():
            out_combined.release() # Ensure writer is released on error
        combined_video_silent = None # Indicate failure
        combiner_initialized = False


# =====================================================
# Part 4: Extract Audio from Final Clips, Concatenate, and Merge with Video
# =====================================================
# (No changes needed in this part, but added existence checks)

# Check if prerequisites exist
if not final_clip_paths:
    print("\nNo final clips with audio were generated. Skipping audio extraction and merge.")
elif not combiner_initialized or combined_video_silent is None or not os.path.exists(combined_video_silent):
     print("\nCombined silent video was not created successfully. Skipping audio extraction and merge.")
else:
    audio_files = []
    print("\nExtracting audio streams...")
    ffmpeg_executable = "ffmpeg" # Re-check availability if needed
    try:
         subprocess.run([ffmpeg_executable, "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
         print(f"ERROR: '{ffmpeg_executable}' command not found or not executable. Cannot proceed.")
         final_clip_paths = [] # Prevent further processing

    extraction_successful = True
    for idx, clip_path in enumerate(final_clip_paths, start=1):
         # Determine index based on the original segment number embedded in the filename
        try:
            original_segment_num = int(os.path.basename(clip_path).split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            original_segment_num = idx # Fallback

        audio_file = os.path.join(input_folder, f"audio_{original_segment_num}.aac")
        extract_cmd = [
            ffmpeg_executable, "-y",
            "-i", clip_path,
            "-vn",           # No video output
            "-acodec", "copy", # Copy audio stream without re-encoding
            audio_file
        ]
        try:
            result = subprocess.run(extract_cmd, check=True, capture_output=True, text=True)
            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                 audio_files.append(audio_file)
                 print(f"Extracted audio from clip {original_segment_num} to {os.path.basename(audio_file)}")
            else:
                 print(f"ERROR: Audio extraction from {clip_path} failed silently or produced empty file.")
                 extraction_successful = False
                 if os.path.exists(audio_file): os.remove(audio_file) # Clean up empty file
                 break # Stop extraction if one fails

        except subprocess.CalledProcessError as e:
            print(f"ERROR extracting audio from {clip_path}:")
            print("Command:", " ".join(e.cmd))
            print("Stderr:", e.stderr)
            extraction_successful = False
            # Optionally remove the failed audio file if it was partially created
            if os.path.exists(audio_file): os.remove(audio_file)
            break # Stop extraction if one fails

    # Proceed only if audio extraction was successful for all intended clips
    if extraction_successful and len(audio_files) == len(final_clip_paths):
        # Create a text file list for audio concatenation
        audio_list_file = os.path.join(input_folder, "audio_list.txt")
        try:
            with open(audio_list_file, "w") as f:
                # Sort audio files based on the number in their filename to ensure correct order
                audio_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                for af in audio_files:
                    # Using just the filename is simplest if ffmpeg's working dir is input_folder or handles it
                    f.write(f"file '{os.path.basename(af)}'\n") # Use relative path for safety with concat demuxer
            print(f"Created audio list file: {audio_list_file}")

            combined_audio = os.path.join(input_folder, "combined_audio.aac")
            concat_audio_cmd = [
                ffmpeg_executable, "-y",
                "-f", "concat",   # Use the concat demuxer
                "-safe", "0",     # Allow relative paths in the list file
                "-i", audio_list_file,
                "-c", "copy",     # Copy the streams without re-encoding
                combined_audio
            ]
            print("Concatenating audio files...")
            result = subprocess.run(concat_audio_cmd, check=True, capture_output=True, text=True) # Capture output
            print(f"Combined audio saved as: {combined_audio}")

            # Finally, merge the combined silent video with the concatenated audio
            final_output = os.path.join(input_folder, "final_output.mp4")
            merge_cmd = [
                ffmpeg_executable, "-y",
                "-i", combined_video_silent, # Combined video (silent)
                "-i", combined_audio,      # Combined audio
                "-c:v", "copy",            # Copy video stream
                "-c:a", "copy",            # Copy audio stream (already AAC)
                # "-c:a", "aac", "-b:a", "128k", # Re-encode audio if needed
                "-shortest",               # Finish when the shorter stream ends
                "-map", "0:v:0",           # Explicitly map video from input 0
                "-map", "1:a:0",           # Explicitly map audio from input 1
                final_output
            ]
            print("Merging final video and audio...")
            result = subprocess.run(merge_cmd, check=True, capture_output=True, text=True) # Capture output

            # Calculate expected duration for final check
            expected_duration_seconds = len(final_clip_paths) * clip_duration
            print(f"\nFinal video (expected duration: {expected_duration_seconds:.1f} seconds) saved as: {final_output}")

            # --- Cleanup --- Optional: Remove intermediate files
            print("Cleaning up intermediate files...")
            cleanup_files = silent_clip_paths + final_clip_paths + audio_files
            if os.path.exists(audio_list_file): cleanup_files.append(audio_list_file)
            if os.path.exists(combined_video_silent): cleanup_files.append(combined_video_silent)
            if os.path.exists(combined_audio): cleanup_files.append(combined_audio)

            for f in cleanup_files:
                 if os.path.exists(f):
                     try:
                         os.remove(f)
                     except OSError as e:
                         print(f"Warning: Could not remove intermediate file {f}: {e}")
            print("Cleanup complete.")
            # --- End Cleanup ---

        except subprocess.CalledProcessError as e:
            print(f"ERROR running FFmpeg during audio concatenation or final merge:")
            print("Command:", " ".join(e.cmd))
            print("Stderr:", e.stderr)
        except Exception as e:
             print(f"An error occurred during audio processing or final merge: {e}")
    elif not extraction_successful:
        print("Skipping audio concatenation and final merge due to audio extraction errors.")
    else:
        print(f"Skipping audio concatenation and final merge. Expected {len(final_clip_paths)} audio files, but found {len(audio_files)}.")


print("\nScript finished.")
