# -*- coding: utf-8 -*- # Recommended for path handling if non-ASCII chars are possible
import cv2
import numpy as np
import os
import subprocess
import math # Needed for ceiling function in safe slicing
import re   # For sentence splitting
import time # For potential delays

# --- New Dependencies ---
# These imports NEED to be at the top
try:
    from g4f.client import Client
except ImportError:
    print("ERROR: g4f library not found. Please install it: pip install -U g4f")
    exit()

try:
    import pollinations
except ImportError:
    print("ERROR: pollinations library not found. Please install it: pip install pollinations")
    exit()
# --- End New Dependencies ---


# =====================================================
# Configuration
# =====================================================

input_folder = "/storage/emulated/0/INPUT"  # adjust this folder path
# Ensure input_folder uses the correct separator for the OS if needed
# input_folder = os.path.normpath(input_folder)

# --- Video Generation Config ---
# num_segments WILL BE DETERMINED BY THE NUMBER OF SENTENCES/IMAGES GENERATED
num_background_videos = 4 # Number of background videos (1.mp4, 2.mp4, etc.) available
clip_duration = 4.0        # seconds per clip
fps = 30                   # frames per second (fixed)
total_frames = int(clip_duration * fps)  # should be 120 frames for 4s at 30fps
output_width = 1280        # target width for 16:9 (e.g. YouTube)
output_height = 720        # target height for 16:9
margin = int(0.10 * output_width) # Margin for overlay image

# --- Animation Config ---
animation_duration_seconds = 0.4 # Duration of the slide animation
if animation_duration_seconds >= clip_duration:
    print("Warning: Animation duration is equal or greater than clip duration. Setting to 0.")
    animation_duration_seconds = 0
elif animation_duration_seconds < 0:
     print("Warning: Animation duration cannot be negative. Setting to 0.")
     animation_duration_seconds = 0
animation_duration_frames = int(animation_duration_seconds * fps)

# --- Resource Paths ---
sound_file = os.path.join(input_folder, "Whoos 3.mp3")
fire_overlay_video_path = os.path.join(input_folder, "Fire.mp4") # <<< Path to fire overlay video

# --- AI Generation Config (Using values from your Pollinations example) ---
g4f_model = "gpt-4o-mini" # From your g4f example
pollinations_model = "flux" # From your Pollinations example
pollinations_seed = "random" # From your Pollinations example
pollinations_width = 1920 # From your Pollinations example (will be resized later if needed)
pollinations_height = 1080 # From your Pollinations example (will be resized later if needed)
pollinations_enhance = False # From your Pollinations example
pollinations_nologo = True # From your Pollinations example
pollinations_private = True # From your Pollinations example - Requires API Key/Login if true
pollinations_safe = False # From your Pollinations example

# =====================================================
# Setup & Input Checks
# =====================================================

# Create input folder if it doesn't exist
if not os.path.exists(input_folder):
    try:
        os.makedirs(input_folder)
        print(f"Created input folder: {input_folder}")
        print(f"Please ensure your {num_background_videos} background MP4s (1.mp4, 2.mp4, ...), Fire.mp4, and {os.path.basename(sound_file)} are in this folder.")
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

# Check for background videos
for i in range(1, num_background_videos + 1):
    bg_vid_path = os.path.join(input_folder, f"{i}.mp4")
    if not os.path.exists(bg_vid_path):
        print(f"ERROR: Background video not found: {bg_vid_path}")
        essential_files_exist = False

if not essential_files_exist:
     print("Essential base files (sound, fire video, background videos) missing. Please check paths and files. Exiting.")
     exit()
# --- End Check ---

# --- Initialize AI Clients ---
# Initialize g4f client (using your sample code structure)
try:
    g4f_client = Client()
    print("g4f client initialized.")
    # Optional: Add a quick test call here if desired
except Exception as e:
    print(f"ERROR: Failed to initialize g4f client: {e}")
    exit()

# Initialize Pollinations client (using your sample code structure)
try:
    # Create an instance of the Pollinations image model with desired settings
    pollinations_image_model = pollinations.Image(
        model=pollinations_model,
        seed=pollinations_seed,
        width=pollinations_width,
        height=pollinations_height,
        enhance=pollinations_enhance,
        nologo=pollinations_nologo,
        private=pollinations_private, # Ensure you are logged in or have credits if True
        safe=pollinations_safe,
        referrer="pollinations.py" # From your example
    )
    print(f"Pollinations client initialized for model: {pollinations_model}")
    # Print the available models (from your example)
    # print("Available models:", pollinations.Image.models())
except Exception as e:
    print(f"ERROR: Failed to initialize Pollinations client: {e}")
    # If privacy is True, this might be an authentication error.
    if pollinations_private:
        print("NOTE: Pollinations 'private=True' was used. Ensure you are authenticated (e.g., via CLI login).")
    exit()


# =====================================================
# Part 0: Text Input, Sentence Splitting, Image Generation
# =====================================================
print("\n--- Part 0: Generating Images from Text ---")

# --- Get User Input ---
user_text = input("Enter the text (paragraphs separated by newlines, sentences ending with '.'):\n")
if not user_text:
    print("ERROR: No text provided. Exiting.")
    exit()

# --- Split into Sentences (using simple '.' delimiter as requested) ---
sentences = [s.strip() for s in user_text.split('.') if s and s.strip()]

if not sentences:
    print("ERROR: No valid sentences found (separated by '.') in the input text. Exiting.")
    exit()

print(f"\nFound {len(sentences)} sentences.")

# --- Generate Prompts and Images ---
generated_image_paths = [] # List to hold paths of successfully generated images
failed_sentences_indices = [] # Keep track of failures

# Get the *entire original script text* to pass to g4f as requested
# This is generally not ideal for prompt generation, but fulfilling the request.
try:
    with open(__file__, 'r', encoding='utf-8') as f_script:
        script_content_for_g4f = f_script.read()
except Exception as e:
    print(f"Warning: Could not read the current script file ({__file__}) to include in g4f prompt: {e}")
    print("Proceeding without including script content in the g4f prompt.")
    script_content_for_g4f = "Video generation script context: Creates 4-second clips combining background video, animated overlay image, fire effect, and sound." # Fallback context


for i, sentence in enumerate(sentences):
    segment_index = i # 0-based index for list access
    segment_num = i + 1 # 1-based index for user messages and filenames
    print(f"\nProcessing Sentence {segment_num}/{len(sentences)}: '{sentence[:60]}...'")

    # 1. Generate Image Prompt using g4f (using your sample code structure)
    image_gen_prompt = None
    try:
        print("  Requesting image prompt from g4f...")
        # Construct the specific prompt requested
        g4f_prompt_content = f"""Give me an image generation prompt for the following sentence:
"{sentence}"

The prompt should be suitable for the following Python script which will use it with the Pollinations library to generate an image. The script then combines this image with background video, sound effects, and a fire overlay. Keep the prompt focused on generating a compelling visual for the sentence.

--- SCRIPT CONTEXT ---
{script_content_for_g4f}
--- END SCRIPT CONTEXT ---

Output only the image generation prompt itself.
"""
        # Using the g4f client structure from your example
        response = g4f_client.chat.completions.create(
            model=g4f_model, # Use the configured model
            messages=[{"role": "user", "content": g4f_prompt_content}],
            web_search=False # As in your example
            # Consider adding a timeout: timeout=60
        )
        image_gen_prompt = response.choices[0].message.content.strip()
        # Basic cleaning: remove potential quotes or common instruction phrases if g4f includes them
        image_gen_prompt = image_gen_prompt.strip('"`*')
        if image_gen_prompt.lower().startswith("image prompt:") or image_gen_prompt.lower().startswith("prompt:"):
             image_gen_prompt = image_gen_prompt.split(":", 1)[-1].strip()
        print(f"  g4f generated prompt: {image_gen_prompt}")

    except Exception as e:
        print(f"  ERROR generating prompt with g4f for sentence {segment_num}: {e}")
        failed_sentences_indices.append(segment_index)
        continue # Skip to next sentence

    # 2. Generate Image using Pollinations (using your sample code structure)
    if image_gen_prompt:
        # Define the output path in the INPUT folder (NOT Downloads)
        # Use a consistent naming scheme, e.g., generated_img_1.png, generated_img_2.png
        image_file_path = os.path.join(input_folder, f"generated_img_{segment_num}.png")

        try:
            print(f"  Requesting image from Pollinations (Model: {pollinations_model})...")
            # Generate the image using the model instance and the g4f prompt
            # (Adapting your loop structure for a single image per sentence)
            image = pollinations_image_model(prompt=image_gen_prompt) # Generate one image

            # Save the image (using your save structure)
            image.save(file=image_file_path)

            # Check if saving worked
            if os.path.exists(image_file_path) and os.path.getsize(image_file_path) > 0:
                print(f"  Image saved successfully: {image_file_path}")
                generated_image_paths.append(image_file_path) # Add path to our list
                # Optional: Print details (from your example)
                # print(f"  Image {segment_num} details: Prompt: {image.prompt}, Response: {image.response}")
            else:
                 print(f"  ERROR: Pollinations generation/saving failed for {image_file_path}. File missing or empty.")
                 failed_sentences_indices.append(segment_index)
                 if os.path.exists(image_file_path): # Clean up empty file
                     try: os.remove(image_file_path)
                     except OSError: pass

        except Exception as e:
            print(f"  ERROR generating/saving image with Pollinations for sentence {segment_num}: {e}")
            failed_sentences_indices.append(segment_index)
            if os.path.exists(image_file_path): # Clean up potentially failed file
                try: os.remove(image_file_path)
                except OSError: pass
            continue # Skip to next sentence
    else:
        print(f"  Skipping image generation for sentence {segment_num} because prompt generation failed.")
        failed_sentences_indices.append(segment_index)

# --- Post-Generation Check ---
if not generated_image_paths:
    print("\nERROR: No images were successfully generated. Cannot proceed to video creation. Exiting.")
    exit()

if failed_sentences_indices:
    print(f"\nWarning: Failed to process {len(failed_sentences_indices)} sentence(s).")
    # Optionally list the failed sentences if needed

# Determine the number of segments based on successfully generated images
num_segments = len(generated_image_paths)
print(f"\nSuccessfully generated {num_segments} images. Proceeding to create {num_segments} video segments.")


# =====================================================
# Part 1: Create 4-Second Silent Clips with Overlay, Animation & Fire
# =====================================================
print("\n--- Part 1: Generating Silent Clips ---")
silent_clip_paths = []

# *** MODIFIED LOOP ***
# Loop through the paths of the successfully generated images from Part 0
for i, overlay_img_path in enumerate(generated_image_paths):
    # i is the 0-based index, overlay_img_path is the full path to the generated image
    segment_num = i + 1 # 1-based segment number for filenames and messages

    # *** Calculate Background Video Index (Looping 1 to num_background_videos) ***
    bg_video_index = (i % num_background_videos) + 1
    bg_video_path = os.path.join(input_folder, f"{bg_video_index}.mp4")

    # Define the output silent clip path using the segment number
    silent_clip = os.path.join(input_folder, f"clip_{segment_num}_silent.mp4")

    print(f"\nProcessing Segment {segment_num}/{num_segments} (Using Img: {os.path.basename(overlay_img_path)}, BG: {bg_video_index}.mp4)...")

    # --- Input File Checks ---
    # Background video existence was checked at the start. Re-check just in case.
    if not os.path.exists(bg_video_path):
        print(f"ERROR: Background video not found: {bg_video_path}. Skipping segment {segment_num}.")
        continue
    # Overlay image path comes from generated_image_paths, so it must exist if it's in the list.
    # We still need to load it below.

    # --- Open Video Captures ---
    cap = cv2.VideoCapture(bg_video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open background video: {bg_video_path}. Skipping segment {segment_num}.")
        continue
    bg_fps = cap.get(cv2.CAP_PROP_FPS)
    bg_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Minor optimization: print only if frame count > 0
    if bg_frame_count > 0 :
         print(f" Background: {os.path.basename(bg_video_path)} ({bg_frame_count} frames @ {bg_fps:.2f} fps)")
    else:
         print(f" Background: {os.path.basename(bg_video_path)} (Frame count issues, FPS: {bg_fps:.2f})")


    # Open Fire Overlay Video Capture for this segment
    fire_cap = cv2.VideoCapture(fire_overlay_video_path)
    if not fire_cap.isOpened():
        print(f"ERROR: Cannot open fire overlay video: {fire_overlay_video_path}. Skipping segment {segment_num}.")
        cap.release()
        continue
    fire_fps = fire_cap.get(cv2.CAP_PROP_FPS)
    fire_frame_count = int(fire_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fire_frame_count <= 0:
         print(f"ERROR: Fire overlay video {fire_overlay_video_path} has no frames or count is invalid. Skipping segment {segment_num}.")
         cap.release()
         fire_cap.release()
         continue
    print(f" Fire Overlay: {os.path.basename(fire_overlay_video_path)} ({fire_frame_count} frames @ {fire_fps:.2f} fps)")


    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # H.264 alternative: *'avc1' or *'h264'
    out = cv2.VideoWriter(silent_clip, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        print(f"ERROR: Failed to open VideoWriter for {silent_clip}. Check permissions or codec support. Skipping segment {segment_num}.")
        cap.release()
        fire_cap.release()
        continue

    # --- Load and Prepare Overlay Image (Using the path from the loop) ---
    overlay_orig = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    if overlay_orig is None:
        print(f"ERROR: Cannot load generated overlay image: {overlay_img_path}. Skipping segment {segment_num}.")
        cap.release()
        fire_cap.release()
        out.release()
        # Attempt to remove the potentially corrupted generated image? Optional.
        # try: os.remove(overlay_img_path)
        # except OSError: pass
        continue

    # Ensure overlay has alpha channel
    if len(overlay_orig.shape) == 2: # Grayscale
         overlay_orig = cv2.cvtColor(overlay_orig, cv2.COLOR_GRAY2BGRA)
    elif overlay_orig.shape[2] == 3: # BGR, add alpha
         overlay_orig = cv2.cvtColor(overlay_orig, cv2.COLOR_BGR2BGRA)
    # If shape[2] == 4, it's already BGRA, do nothing.

    # --- Resizing Logic (Same as original script) ---
    max_width = output_width - 2 * margin
    max_height = output_height - 2 * margin
    orig_h, orig_w = overlay_orig.shape[:2]
    if orig_w == 0 or orig_h == 0:
        print(f"ERROR: Invalid overlay image dimensions for {overlay_img_path} (after loading). Skipping segment {segment_num}.")
        cap.release()
        fire_cap.release()
        out.release()
        continue
    scale_factor = min(max_width / orig_w, max_height / orig_h)
    new_w = max(1, int(orig_w * scale_factor))
    new_h = max(1, int(orig_h * scale_factor))
    try:
        # Use INTER_LANCZOS4 for potentially better downscaling quality
        overlay_resized = cv2.resize(overlay_orig, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4 if scale_factor < 1.0 else cv2.INTER_LINEAR)
    except Exception as e:
        print(f"ERROR: Failed to resize overlay image {overlay_img_path}: {e}. Skipping segment {segment_num}.")
        cap.release()
        fire_cap.release()
        out.release()
        continue
    print(f" Overlay Image: {os.path.basename(overlay_img_path)} resized to {new_w}x{new_h}")

    # --- Animation Setup (Cycles 0, 1, 2, 3 based on segment number) ---
    target_x = margin + (max_width - new_w) // 2
    target_y = margin + (max_height - new_h) // 2
    # Use the 1-based segment_num for consistent animation cycling
    animation_type = (segment_num - 1) % 4
    start_x, start_y = target_x, target_y
    if animation_duration_frames > 0:
        anim_map = {0: "Top", 1: "Left", 2: "Bottom", 3: "Right"}
        print(f" Animating image slide from {anim_map[animation_type]} ({animation_duration_seconds}s)")
        if animation_type == 0: start_y = -new_h       # Start above screen
        elif animation_type == 1: start_x = -new_w       # Start left of screen
        elif animation_type == 2: start_y = output_height # Start below screen
        elif animation_type == 3: start_x = output_width  # Start right of screen

    # --- Frame Processing Loop (Identical logic to original script) ---
    processed_frames = 0
    last_good_bg_frame = None
    fire_frame_index = -1 # To handle looping

    while processed_frames < total_frames:
        # --- Read Background Frame ---
        ret_bg, bg_frame = cap.read()
        current_bg_frame_for_compositing = None # Frame to use for this iteration
        if not ret_bg:
            if last_good_bg_frame is None:
                print(f"\nERROR: Background video {bg_video_path} couldn't read first frame. Using black background.")
                # Create a black frame as fallback ONLY if the first frame fails
                last_good_bg_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                current_bg_frame_for_compositing = last_good_bg_frame.copy() # Use black frame now
            else:
                # Pad with the last good frame if not the first frame failing
                current_bg_frame_for_compositing = last_good_bg_frame.copy() # Use last good frame
                if processed_frames % fps == 0: # Print warning less often
                    print(f"\nWarning: Reached end of background video {bg_video_path}. Re-using last frame.", end='')
        else:
             # Resize the background frame immediately
             try:
                 bg_frame_resized = cv2.resize(bg_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                 last_good_bg_frame = bg_frame_resized # Store the successful resize (no copy needed yet)
                 current_bg_frame_for_compositing = bg_frame_resized # Use the newly read & resized frame
             except Exception as e:
                 if last_good_bg_frame is None:
                      print(f"\nERROR: Failed to resize first background frame for segment {segment_num}: {e}. Using black background.")
                      last_good_bg_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                      current_bg_frame_for_compositing = last_good_bg_frame.copy()
                 else:
                      if processed_frames % fps == 0: # Print warning less often
                         print(f"\nWarning: Failed to resize background frame {processed_frames} for segment {segment_num}: {e}. Using last good frame.", end='')
                      current_bg_frame_for_compositing = last_good_bg_frame.copy() # Use copy of last good one


        # --- Read Fire Frame (Looping) ---
        fire_frame_index += 1
        if fire_frame_index >= fire_frame_count:
            fire_frame_index = 0 # Reset index for looping
            fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Go back to start of fire video

        ret_fire, fire_frame = fire_cap.read()
        if not ret_fire:
             # If reading fails even after reset (e.g., video corrupted), try reading first frame again
             if processed_frames % fps == 0 : print(f"\nWarning: Failed to read frame {fire_frame_index} from fire video. Attempting reset/reread.", end='')
             fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             ret_fire, fire_frame = fire_cap.read()
             if not ret_fire:
                 print("\nERROR: Cannot read from fire video even after reset. Disabling fire overlay for remaining frames.")
                 # Use a black frame of correct size and type
                 fire_frame_resized = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                 # To prevent further errors, we can set a flag or just keep using this black frame
             else:
                  fire_frame_index = 0 # Successfully read first frame
                  # Need to resize this first frame
                  try:
                       fire_frame_resized = cv2.resize(fire_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                  except Exception as e:
                       print(f"\nWarning: Could not resize reset fire frame {fire_frame_index}. Using black frame. Error: {e}")
                       fire_frame_resized = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        else:
             # Resize fire frame to output dimensions if read successfully
            try:
                fire_frame_resized = cv2.resize(fire_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                if processed_frames % 60 == 0: # Print less often
                    print(f"\nWarning: Could not resize fire frame {fire_frame_index}. Using black frame. Error: {e}", end='')
                # Use a black frame if resize fails
                fire_frame_resized = np.zeros((output_height, output_width, 3), dtype=np.uint8)


        # --- Start Compositing ---
        # Initialize composite frame with the background frame determined above
        # Make a copy to avoid modifying last_good_bg_frame unintentionally if it was used
        composite = current_bg_frame_for_compositing.copy()

        # --- Calculate Image Overlay Position (Animation/Static) ---
        current_x = target_x
        current_y = target_y
        if processed_frames < animation_duration_frames and animation_duration_frames > 0:
            # Linear interpolation
            progress = min(1.0, (processed_frames + 1) / animation_duration_frames)
            # # Optional: Ease-out interpolation (smoother)
            # progress = 1 - (1 - progress)**2
            current_x = int(start_x + progress * (target_x - start_x))
            current_y = int(start_y + progress * (target_y - start_y))


        # --- Place Image Overlay (Safe Placement with Alpha Blending) ---
        # (Using the exact safe slicing logic from the original script)
        x1, y1 = current_x, current_y
        x2, y2 = current_x + new_w, current_y + new_h
        frame_x1, frame_y1 = max(x1, 0), max(y1, 0)
        frame_x2, frame_y2 = min(x2, output_width), min(y2, output_height)

        if frame_x1 < frame_x2 and frame_y1 < frame_y2: # Check if there is overlap
            overlay_x1 = frame_x1 - x1
            overlay_y1 = frame_y1 - y1
            overlay_w = frame_x2 - frame_x1
            overlay_h = frame_y2 - frame_y1
            overlay_x2 = overlay_x1 + overlay_w
            overlay_y2 = overlay_y1 + overlay_h

            # Clamp overlay coordinates to valid range within overlay_resized
            overlay_x1, overlay_y1 = max(0, overlay_x1), max(0, overlay_y1)
            overlay_x2, overlay_y2 = min(new_w, overlay_x2), min(new_h, overlay_y2)
            overlay_w, overlay_h = overlay_x2 - overlay_x1, overlay_y2 - overlay_y1

            # Recalculate frame bounds based on clamped overlay dimensions
            frame_x2 = frame_x1 + overlay_w
            frame_y2 = frame_y1 + overlay_h

            if overlay_w > 0 and overlay_h > 0: # Check if there's still a valid overlay part
                try:
                    roi = composite[frame_y1:frame_y2, frame_x1:frame_x2]
                    overlay_part = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

                    # Double check shapes before blending (can happen with off-screen animations)
                    if roi.shape[:2] == overlay_part.shape[:2]:
                        alpha = overlay_part[:, :, 3] / 255.0
                        overlay_colors = overlay_part[:, :, :3]
                        alpha_mask = cv2.cvtColor(alpha.astype(np.float32), cv2.COLOR_GRAY2BGR) # Create 3-channel mask

                        # Blend: (Overlay * alpha) + (Background * (1 - alpha))
                        blended_roi = (overlay_colors * alpha_mask) + (roi * (1.0 - alpha_mask))
                        composite[frame_y1:frame_y2, frame_x1:frame_x2] = blended_roi.astype(np.uint8)
                    # else:
                        # Optional: Log shape mismatch if debugging animation issues
                        # if processed_frames % 30 == 0: print(f" Shape mismatch ROI {roi.shape} vs Overlay {overlay_part.shape}")
                except IndexError as e:
                     # This might happen if calculations are slightly off, especially during animation start/end
                     if processed_frames % 60 == 0: # Print less often
                         print(f"\nWarning: Index error during overlay blending frame {processed_frames}: {e}. Coords: roi[{frame_y1}:{frame_y2}, {frame_x1}:{frame_x2}], overlay[{overlay_y1}:{overlay_y2}, {overlay_x1}:{overlay_x2}]")
                except Exception as e:
                     if processed_frames % 60 == 0: # Print less often
                         print(f"\nWarning: Error during image overlay blending frame {processed_frames}: {e}")


        # --- Add Fire Overlay ---
        # (Using the exact Additive Blending logic from the original script)
        try:
             # Ensure composite is 3 channels BGR before adding
             if composite.shape[2] == 4:
                 composite = cv2.cvtColor(composite, cv2.COLOR_BGRA2BGR)

             # Ensure fire_frame_resized is valid and 3 channels BGR
             if fire_frame_resized is not None and fire_frame_resized.shape[2] == 4:
                 fire_frame_resized = cv2.cvtColor(fire_frame_resized, cv2.COLOR_BGRA2BGR)

             # Final check for compatibility before adding (size and channels)
             if fire_frame_resized is not None and composite.shape == fire_frame_resized.shape:
                 composite = cv2.add(composite, fire_frame_resized)
             elif fire_frame_resized is not None and processed_frames % 60 == 0: # Log mismatch if fire frame exists but dimensions differ
                 print(f"\nWarning: Shape mismatch composite ({composite.shape}) vs fire ({fire_frame_resized.shape}). Skipping fire.", end='')

        except Exception as e:
             if processed_frames % 60 == 0: # Print less often
                 print(f"\nWarning: Error during fire overlay blending frame {processed_frames}: {e}", end='')


        # --- Write Frame ---
        try:
            out.write(composite)
        except Exception as e:
             print(f"\nFATAL ERROR: Failed to write frame {processed_frames} for segment {segment_num}: {e}")
             print("Aborting processing for this segment.")
             processed_frames = total_frames + 1 # Force loop exit
             break # Exit inner while loop

        processed_frames += 1
        # Update progress on the same line
        print(f"  Segment {segment_num}: Processing Frame {processed_frames}/{total_frames}...", end='\r')


    # --- End Frame Processing Loop for Segment ---
    print(f"  Segment {segment_num}: Processed {min(processed_frames, total_frames)} frames. {' ' * 20}") # Clear progress line
    cap.release()
    fire_cap.release()
    out.release() # Close the VideoWriter for this segment

    if processed_frames > total_frames : # Check if loop was aborted due to write error
        print(f" Failed silent clip generation for segment {segment_num} due to write error. Not adding to list.")
        if os.path.exists(silent_clip):
             try: os.remove(silent_clip)
             except OSError as e: print(f" Error removing incomplete file {silent_clip}: {e}")
    elif processed_frames == total_frames:
        print(f" Silent clip with overlays saved: {silent_clip}")
        silent_clip_paths.append(silent_clip)
    else: # Should not happen with current logic unless initial background read fails badly
        print(f" Incomplete silent clip for segment {segment_num} ({processed_frames}/{total_frames}). Not adding to list.")
        if os.path.exists(silent_clip):
            try: os.remove(silent_clip)
            except OSError as e: print(f" Error removing incomplete file {silent_clip}: {e}")


# =====================================================
# Part 2: Add Sound Effect Using FFmpeg
# (No changes needed in this section's logic, uses silent_clip_paths)
# =====================================================
print("\n--- Part 2: Adding Audio ---")
final_clip_paths = []
ffmpeg_executable = "ffmpeg" # Or provide full path

# Check FFmpeg availability once
ffmpeg_ok = False
try:
     # Use startupinfo on Windows to prevent console window popup
     startupinfo = None
     if os.name == 'nt':
         startupinfo = subprocess.STARTUPINFO()
         startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
     result = subprocess.run([ffmpeg_executable, "-version"], check=True, capture_output=True, text=True, startupinfo=startupinfo)
     print("FFmpeg found:", result.stdout.split('\n', 1)[0]) # Print first line of version info
     ffmpeg_ok = True
except FileNotFoundError:
     print(f"ERROR: '{ffmpeg_executable}' command not found.")
     print("Make sure FFmpeg is installed and in your system's PATH, or provide the full path.")
except subprocess.CalledProcessError as e:
     print(f"ERROR: FFmpeg found but '{ffmpeg_executable} -version' failed: {e}")
     print("FFmpeg might be installed incorrectly.")
except Exception as e:
    print(f"ERROR: An unexpected error occurred while checking for FFmpeg: {e}")


if not silent_clip_paths:
     print("No silent clips were successfully created in Part 1. Skipping audio addition.")
elif not ffmpeg_ok:
      print("FFmpeg not found or not working. Skipping audio addition.")
else:
    for idx, silent_clip in enumerate(silent_clip_paths):
        # Extract original segment number from filename (e.g., clip_N_silent.mp4)
        try:
            original_segment_num = int(os.path.basename(silent_clip).split('_')[1])
        except (IndexError, ValueError):
            original_segment_num = idx + 1 # Fallback if parsing fails (use 1-based index)
            print(f"Warning: Could not parse segment number from {silent_clip}. Using index {original_segment_num}.")

        final_clip = os.path.join(input_folder, f"final_{original_segment_num}.mp4")

        # --- FFmpeg Command (Identical to original script) ---
        ffmpeg_cmd = [
            ffmpeg_executable, "-y", # Overwrite output without asking
            "-i", silent_clip,     # Input 0: Silent video clip
            "-i", sound_file,      # Input 1: Sound effect
            "-filter_complex",
            # Trim sound to 1s, generate 3s silence, concatenate to match 4s video
            # Ensure parameters match clip duration
            f"[1:a]atrim=duration=1,asetpts=PTS-STARTPTS[sound]; "
            f"anullsrc=r=44100:cl=stereo,atrim=duration={clip_duration - 1.0},asetpts=PTS-STARTPTS[silence]; "
            "[sound][silence]concat=n=2:v=0:a=1[a]",
            "-map", "0:v:0",       # Map video from input 0
            "-map", "[a]",         # Map the filtered audio stream
            "-c:v", "copy",        # Copy video codec (fast)
            "-c:a", "aac",         # Encode audio to AAC
            "-b:a", "128k",        # Set audio bitrate
            "-shortest",           # Ensure output duration matches shortest input (the video)
            final_clip
        ]
        print(f" Adding sound to clip {original_segment_num} -> {os.path.basename(final_clip)}...")
        try:
            # Use startupinfo on Windows
            startupinfo = None
            if os.name == 'nt':
                 startupinfo = subprocess.STARTUPINFO()
                 startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', startupinfo=startupinfo)
            # Check if output file exists and has size
            if os.path.exists(final_clip) and os.path.getsize(final_clip) > 0:
                 print(f"  Success.")
                 final_clip_paths.append(final_clip)
            else:
                 print(f"  ERROR: FFmpeg ran but output file {final_clip} is missing or empty.")
                 print(f"  FFmpeg stdout:\n{result.stdout}")
                 print(f"  FFmpeg stderr:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            print(f"  ERROR running FFmpeg for clip {original_segment_num}:")
            # Print command filtering complex parts for readability if needed
            print(f"  Command: {' '.join(e.cmd)}") # Basic command print
            print(f"  Return Code: {e.returncode}")
            print(f"  Stderr: {e.stderr}") # Crucial for debugging FFmpeg errors
            if os.path.exists(final_clip): # Clean up failed output
                try: os.remove(final_clip)
                except OSError as rm_err: print(f"  Warning: Could not remove failed output {final_clip}: {rm_err}")
        except Exception as e:
             print(f"  An unexpected error occurred during FFmpeg execution for clip {original_segment_num}: {e}")


# =====================================================
# Part 3: Combine Final Clips (Video Only) Using OpenCV
# (No changes needed in this section's logic, uses final_clip_paths)
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
        # Assumes filename format final_N.mp4
        final_clip_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        print(" Combining video from clips:", [os.path.basename(p) for p in final_clip_paths])

        # Get parameters from the first successfully created final clip
        first_clip_path = final_clip_paths[0]
        cap_check = cv2.VideoCapture(first_clip_path)
        if not cap_check.isOpened():
            raise IOError(f"Could not open first final clip {first_clip_path} to get parameters.")

        comb_width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        comb_height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        comb_fps = cap_check.get(cv2.CAP_PROP_FPS)

        # Validate parameters read from the first file
        if not (comb_width == output_width and comb_height == output_height):
             print(f"Warning: Dimensions read from {first_clip_path} ({comb_width}x{comb_height}) don't match target ({output_width}x{output_height}). Using target dimensions.")
             comb_width, comb_height = output_width, output_height
        if not (0 < comb_fps < 200): # Check for reasonable FPS range
             print(f"Warning: Read invalid FPS ({comb_fps}) from {first_clip_path}. Falling back to configured fps ({fps}).")
             comb_fps = fps
        else:
             # Optional: Check if read FPS matches target FPS
             if abs(comb_fps - fps) > 0.1:
                 print(f"Warning: FPS read from {first_clip_path} ({comb_fps:.2f}) differs from target ({fps}). Using read FPS for combiner.")
                 # Decide whether to use comb_fps or fps here. Using comb_fps might be safer if Part 2 changed it.
                 # Sticking with comb_fps read from the file for now.
        cap_check.release()

        print(f" Combining videos with parameters: {comb_width}x{comb_height} @ {comb_fps:.2f} fps")
        fourcc_comb = cv2.VideoWriter_fourcc(*"mp4v")
        out_combined = cv2.VideoWriter(combined_video_silent, fourcc_comb, comb_fps, (comb_width, comb_height))
        if not out_combined.isOpened():
             raise IOError(f"Could not open VideoWriter for {combined_video_silent}")
        combiner_initialized = True

        total_combined_frames = 0
        for clip_idx, clip_path in enumerate(final_clip_paths):
            clip_frame_count = 0
            segment_num = clip_idx + 1 # Or parse from filename again if needed
            print(f"  Appending segment {segment_num}: {os.path.basename(clip_path)}...")
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                print(f"  ERROR: Cannot open {clip_path} for combination. Skipping.")
                continue

            # Optional: Verify dimensions of each clip being added
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w != comb_width or h != comb_height:
                print(f"  ERROR: Dimension mismatch in {os.path.basename(clip_path)} ({w}x{h}). Expected ({comb_width}x{comb_height}). Skipping.")
                cap.release()
                continue

            frame_read_error = False
            while True:
                ret, frame = cap.read()
                if not ret: break
                try:
                    out_combined.write(frame)
                    clip_frame_count += 1
                except Exception as e:
                    print(f"\n  ERROR writing frame {clip_frame_count+1} from {os.path.basename(clip_path)}: {e}")
                    frame_read_error = True
                    break # Stop processing this clip

            cap.release()
            if frame_read_error:
                print(f"  Stopped appending {os.path.basename(clip_path)} due to frame write error.")
                # Decide if the whole combination should fail or just continue with partial data
                combiner_initialized = False # Mark combination as potentially failed/incomplete
                break # Stop combining further clips
            else:
                total_combined_frames += clip_frame_count
                print(f"    -> Appended {clip_frame_count} frames.")

        if combiner_initialized: # Only print success if no errors stopped the process
             print(f" Combined silent video saved: {combined_video_silent} ({total_combined_frames} frames)")

    except Exception as e:
        print(f"Error during video combination: {e}")
        combiner_initialized = False # Mark as failed
        combined_video_silent = None # Prevent further use
    finally:
         if out_combined is not None and out_combined.isOpened():
              out_combined.release() # Ensure writer is released even on error


# =====================================================
# Part 4: Extract Audio, Concatenate, Merge
# (No changes needed in this section's logic, uses final_clip_paths)
# =====================================================
print("\n--- Part 4: Processing Audio and Final Merge ---")

if not final_clip_paths:
    print("No final clips available. Skipping audio extraction and merge.")
elif not combiner_initialized or combined_video_silent is None or not os.path.exists(combined_video_silent):
     print("Combined silent video was not created successfully or combination failed. Skipping audio extraction and merge.")
elif not ffmpeg_ok:
     print("FFmpeg not found or not working. Skipping audio extraction and merge.")
else:
    audio_files = []
    # Use a temporary subfolder for intermediate audio files (optional but cleaner)
    audio_temp_dir = os.path.join(input_folder, "temp_audio")
    os.makedirs(audio_temp_dir, exist_ok=True)

    audio_list_file = os.path.join(audio_temp_dir, "audio_list.txt")
    combined_audio = os.path.join(audio_temp_dir, "combined_audio.aac")
    final_output = os.path.join(input_folder, "final_output.mp4") # Final output in main input folder
    extraction_successful = True

    print(" Extracting audio streams...")
    # Use the sorted list from Part 3
    for idx, clip_path in enumerate(final_clip_paths):
        # Extract original segment number for consistent naming
        try:
            original_segment_num = int(os.path.basename(clip_path).split('_')[1].split('.')[0])
        except:
            original_segment_num = idx + 1 # Fallback

        audio_file = os.path.join(audio_temp_dir, f"audio_{original_segment_num}.aac")
        extract_cmd = [ffmpeg_executable, "-y", "-i", clip_path, "-vn", "-acodec", "copy", audio_file]
        print(f"  Extracting audio from segment {original_segment_num} -> {os.path.basename(audio_file)}...")
        try:
            # Use startupinfo on Windows
            startupinfo = None
            if os.name == 'nt':
                 startupinfo = subprocess.STARTUPINFO()
                 startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(extract_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', startupinfo=startupinfo)
            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                 audio_files.append(audio_file)
                 print(f"    -> Success.")
            else:
                 print(f"    ERROR: Audio extraction from {clip_path} failed or produced empty file.")
                 print(f"    FFmpeg stdout:\n{result.stdout}")
                 print(f"    FFmpeg stderr:\n{result.stderr}")
                 extraction_successful = False; break
        except subprocess.CalledProcessError as e:
            print(f"    ERROR extracting audio from {clip_path}:")
            print(f"    Command: {' '.join(e.cmd)}")
            print(f"    Return Code: {e.returncode}")
            print(f"    Stderr: {e.stderr}")
            extraction_successful = False; break # Stop extraction if one fails
        except Exception as e:
             print(f"  An unexpected error occurred during audio extraction for {clip_path}: {e}")
             extraction_successful = False; break


    # Proceed only if all audio extractions were successful and counts match
    if extraction_successful and len(audio_files) == len(final_clip_paths):
        merge_step_ok = True # Flag for successful audio concat and final merge
        try:
            # Create audio list file (using relative paths within temp dir might be safer for ffmpeg concat)
            print(f" Creating audio list file: {audio_list_file}")
            with open(audio_list_file, "w", encoding='utf-8') as f:
                # Ensure audio files are sorted correctly based on the segment number for concatenation
                audio_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                for af in audio_files:
                    # Use just the filename for the list, as ffmpeg will run from the temp dir context or use -safe 0
                    f.write(f"file '{os.path.basename(af)}'\n")


            # Concatenate audio files using the list
            # Use -safe 0 if using absolute paths or paths outside the CWD for the list file
            # Running from the temp dir might avoid needing -safe 0
            concat_audio_cmd = [ffmpeg_executable, "-y", "-f", "concat", "-safe", "0", "-i", audio_list_file, "-c", "copy", combined_audio]
            print(" Concatenating audio files...")
            # Use startupinfo on Windows
            startupinfo = None
            if os.name == 'nt':
                 startupinfo = subprocess.STARTUPINFO()
                 startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(concat_audio_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', startupinfo=startupinfo, cwd=audio_temp_dir) # Run from temp dir?
            if not (os.path.exists(combined_audio) and os.path.getsize(combined_audio) > 0):
                 print(f"  ERROR: Audio concatenation failed or produced empty file: {combined_audio}")
                 print(f"  FFmpeg stdout:\n{result.stdout}")
                 print(f"  FFmpeg stderr:\n{result.stderr}")
                 merge_step_ok = False
            else:
                 print(f"  Combined audio saved: {combined_audio}")

            # Merge final video and concatenated audio
            if merge_step_ok:
                merge_cmd = [
                    ffmpeg_executable, "-y",
                    "-i", combined_video_silent, # Input 0: Video from Part 3
                    "-i", combined_audio,        # Input 1: Combined audio
                    "-c:v", "copy",              # Copy video stream
                    "-c:a", "copy",              # Copy audio stream
                    "-map", "0:v:0",             # Map video from input 0
                    "-map", "1:a:0",             # Map audio from input 1
                    "-shortest",                 # Ensure duration matches shortest (usually video)
                    final_output
                ]
                print(" Merging final video and audio...")
                result = subprocess.run(merge_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', startupinfo=startupinfo) # startupinfo for Windows
                if not (os.path.exists(final_output) and os.path.getsize(final_output) > 0):
                     print(f"  ERROR: Final merge failed or produced empty file: {final_output}")
                     print(f"  FFmpeg stdout:\n{result.stdout}")
                     print(f"  FFmpeg stderr:\n{result.stderr}")
                     merge_step_ok = False
                else:
                     expected_duration_seconds = len(final_clip_paths) * clip_duration
                     print(f"\n Final video (expected duration: ~{expected_duration_seconds:.1f}s) saved as: {final_output}")


        except subprocess.CalledProcessError as e:
            print(f"\nERROR running FFmpeg during audio concatenation or final merge:")
            print(f"  Command: {' '.join(e.cmd)}")
            print(f"  Return Code: {e.returncode}")
            print(f"  Stderr: {e.stderr}")
            merge_step_ok = False
        except Exception as e:
             print(f"\nAn error occurred during audio processing or final merge: {e}")
             merge_step_ok = False

        # --- Cleanup ---
        # Clean up intermediate files ONLY if the final merge was successful
        if merge_step_ok and os.path.exists(final_output):
            print("\n Cleaning up intermediate files...")
            # Include generated images, silent clips, final clips, intermediate audio folder
            cleanup_files = generated_image_paths + silent_clip_paths + final_clip_paths
            # No need to add individual audio files if we remove the whole dir
            # cleanup_files.extend(audio_files)
            # if os.path.exists(audio_list_file): cleanup_files.append(audio_list_file)
            # if os.path.exists(combined_audio): cleanup_files.append(combined_audio) # Keep this? No, remove temp dir
            if os.path.exists(combined_video_silent): cleanup_files.append(combined_video_silent)

            removed_count = 0
            failed_removals = []
            for f in cleanup_files:
                 if os.path.exists(f):
                     try:
                         os.remove(f)
                         removed_count += 1
                     except OSError as e:
                         print(f"  Warning: Could not remove {f}: {e}")
                         failed_removals.append(f)

            # Remove the temporary audio directory and its contents
            if os.path.exists(audio_temp_dir):
                try:
                    # Remove individual files first (more robust than shutil.rmtree on some systems/permissions)
                    for item in os.listdir(audio_temp_dir):
                        item_path = os.path.join(audio_temp_dir, item)
                        try:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                                removed_count += 1
                        except OSError as e:
                             print(f"  Warning: Could not remove temp audio file {item_path}: {e}")
                             failed_removals.append(item_path)
                    os.rmdir(audio_temp_dir) # Remove empty dir
                    print(f"  Removed temporary audio directory: {audio_temp_dir}")
                except OSError as e:
                    print(f"  Warning: Could not remove temporary audio directory {audio_temp_dir}: {e}")
                    failed_removals.append(audio_temp_dir)

            print(f" Cleanup complete (attempted removal of {len(cleanup_files) + len(os.listdir(audio_temp_dir))} items).")
            if failed_removals:
                 print(f"  Could not remove the following items: {failed_removals}")
        else:
             print("\nSkipping cleanup as final merge failed or did not produce an output file.")
             print("Intermediate files (generated images, silent clips, final clips, combined video/audio) may remain in:")
             print(f" - {input_folder}")
             if os.path.exists(audio_temp_dir): print(f" - {audio_temp_dir}")
        # --- End Cleanup ---

    elif not extraction_successful:
        print("Skipping audio concatenation and final merge due to audio extraction errors.")
    else: # Mismatch in count (shouldn't happen with current logic, but check anyway)
        print(f"Skipping audio concatenation and final merge. Expected {len(final_clip_paths)} audio files, but extracted {len(audio_files)}.")
        # Clean up any audio files that *were* extracted
        if os.path.exists(audio_temp_dir):
             print(" Cleaning up partially extracted audio files...")
             # (Add removal logic similar to cleanup block if desired)


print("\n--- Script finished ---")
