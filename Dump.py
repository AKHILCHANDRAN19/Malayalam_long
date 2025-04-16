# -*- coding: utf-8 -*- # Recommended for path handling if non-ASCII chars are possible
import cv2
import numpy as np
import os
import subprocess
import math # Needed for ceiling function and dynamic duration calculation
import re   # For sentence splitting
import time # For potential delays
import asyncio # For edge-tts

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

try:
    import edge_tts
except ImportError:
    print("ERROR: edge-tts library not found. Please install it: pip install edge-tts")
    exit()

try:
    from pydub import AudioSegment
except ImportError:
    print("ERROR: pydub library not found. Please install it: pip install pydub")
    print("NOTE: pydub also requires ffmpeg or libav to be installed for audio processing.")
    exit()
# --- End New Dependencies ---


# =====================================================
# Configuration
# =====================================================

input_folder = "/storage/emulated/0/INPUT"  # adjust this folder path
# Ensure input_folder uses the correct separator for the OS if needed
# input_folder = os.path.normpath(input_folder)

# --- TTS Configuration ---
TTS_VOICE = "ml-IN-MidhunNeural" # Default voice 1 as requested
# Voices: "ml-IN-MidhunNeural" (Male), "ml-IN-SobhanaNeural" (Female)
TTS_TEMP_DIR = os.path.join(input_folder, "temp_tts") # Subfolder for TTS files

# --- Video Generation Config ---
# num_segments WILL BE DETERMINED BY THE NUMBER OF SENTENCES * 2
num_background_videos = 4 # Number of background videos (1.mp4, 2.mp4, etc.) available
# clip_duration IS NOW DYNAMICALLY SET PER CLIP BASED ON TTS DURATION
fps = 30                   # frames per second (fixed)
# total_frames WILL BE DYNAMIC PER CLIP
output_width = 1280        # target width for 16:9 (e.g. YouTube)
output_height = 720        # target height for 16:9
margin = int(0.10 * output_width) # Margin for overlay image

# --- Animation Config ---
# Animation occurs at the START of each clip
animation_duration_seconds = 0.4 # Duration of the slide animation
# Check against a reasonable minimum clip duration, not a fixed one anymore
# We'll check per-clip later if animation is feasible
animation_duration_frames = int(animation_duration_seconds * fps)

# --- Resource Paths ---
# sound_file = os.path.join(input_folder, "Whoos 3.mp3") # No longer used for final audio merge
fire_overlay_video_path = os.path.join(input_folder, "Fire.mp4") # Path to fire overlay video

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
        print(f"Please ensure your {num_background_videos} background MP4s (1.mp4, 2.mp4, ...), Fire.mp4 are in this folder.") # Removed sound file ref
    except OSError as e:
        print(f"ERROR: Could not create input folder {input_folder}: {e}")
        exit() # Exit if we can't create the essential folder

# Create TTS temp folder
if not os.path.exists(TTS_TEMP_DIR):
    try:
        os.makedirs(TTS_TEMP_DIR)
        print(f"Created TTS temp folder: {TTS_TEMP_DIR}")
    except OSError as e:
        print(f"ERROR: Could not create TTS temp folder {TTS_TEMP_DIR}: {e}")
        exit()

# --- Check for essential files ---
essential_files_exist = True
# if not os.path.exists(sound_file): # Sound file no longer strictly essential for TTS path
#     print(f"WARNING: Sound file not found: {sound_file}. Not needed for TTS audio track.")
#     # essential_files_exist = False # Don't make it fatal if using TTS
if not os.path.exists(fire_overlay_video_path):
    print(f"ERROR: Fire overlay video not found: {fire_overlay_video_path}")
    essential_files_exist = False

# Check for background videos
for i in range(1, num_background_videos + 1):
    bg_vid_path = os.path.join(input_folder, f"{i}.mp4")
    if not os.path.exists(bg_vid_path):
        print(f"ERROR: Background video not found: {bg_vid_path}")
        essential_files_exist = False

# Check FFmpeg (needed for pydub duration check AND video/audio processing)
ffmpeg_executable = "ffmpeg"
ffmpeg_ok = False
try:
     startupinfo = None
     if os.name == 'nt':
         startupinfo = subprocess.STARTUPINFO()
         startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
     result = subprocess.run([ffmpeg_executable, "-version"], check=True, capture_output=True, text=True, startupinfo=startupinfo)
     print("FFmpeg found:", result.stdout.split('\n', 1)[0])
     ffmpeg_ok = True
except Exception as e:
     print(f"ERROR: FFmpeg check failed: {e}. FFmpeg (or libav) is required for pydub and video processing. Ensure it's installed and in PATH.")
     essential_files_exist = False # FFmpeg is essential now

if not essential_files_exist:
     print("Essential base files (fire video, background videos) or FFmpeg missing. Please check paths and files. Exiting.")
     exit()
# --- End Check ---

# --- Initialize AI Clients ---
try:
    g4f_client = Client()
    print("g4f client initialized.")
except Exception as e:
    print(f"ERROR: Failed to initialize g4f client: {e}")
    exit()

try:
    pollinations_image_model = pollinations.Image(
        model=pollinations_model,
        seed=pollinations_seed,
        width=pollinations_width,
        height=pollinations_height,
        enhance=pollinations_enhance,
        nologo=pollinations_nologo,
        private=pollinations_private,
        safe=pollinations_safe,
        referrer="Codepoli_tts.py" # Updated referrer based on typical filename
    )
    print(f"Pollinations client initialized for model: {pollinations_model}")
except Exception as e:
    print(f"ERROR: Failed to initialize Pollinations client: {e}")
    if pollinations_private:
        print("NOTE: Pollinations 'private=True' was used. Ensure you are authenticated.")
    exit()


# =====================================================
# Part 0: Text Input, Sentence Splitting, TTS Generation, Image Generation
# =====================================================
print("\n--- Part 0: Processing Text, Generating TTS & Images ---")

# --- Get User Input ---
user_text = input("Enter the text (paragraphs separated by newlines, sentences ending with '.'):\n")
if not user_text:
    print("ERROR: No text provided. Exiting.")
    exit()

# --- Split into Sentences ---
# Use regex to split by '.' possibly followed by whitespace, keeping punctuation conceptually with the sentence if needed later.
sentences = re.split(r'(?<=\.)\s*', user_text) # Split after dot and optional space
sentences = [s.strip() for s in sentences if s and s.strip()] # Remove empty strings and strip whitespace

if not sentences:
    print("ERROR: No valid sentences found (separated by '.') in the input text. Exiting.")
    exit()

print(f"\nFound {len(sentences)} sentences.")

# --- TTS Generation ---
async def generate_tts_async(text: str, output_file: str, voice: str):
    """Async helper for edge-tts generation."""
    try:
        communicate = edge_tts.Communicate(text, voice=voice)
        with open(output_file, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
        return True # Indicate success
    except Exception as e:
        print(f"  ERROR during TTS generation for '{text[:30]}...': {e}")
        return False # Indicate failure

async def run_tts_for_sentences(sentence_list, voice, output_dir):
    """Generates TTS for all sentences and gets duration."""
    tts_results = [] # List to store (sentence_index, tts_filepath, tts_duration)
    tasks = []
    output_files = []

    print(f"\nGenerating TTS audio using voice: {voice}...")
    for i, sentence in enumerate(sentence_list):
        output_file = os.path.join(output_dir, f"tts_{i+1}.mp3")
        output_files.append(output_file)
        # Create task for each sentence
        tasks.append(generate_tts_async(sentence, output_file, voice))

    # Run all TTS generation tasks concurrently
    results = await asyncio.gather(*tasks)

    print("\nCalculating TTS durations...")
    successful_tts = []
    for i, success in enumerate(results):
        output_file = output_files[i]
        if success and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                audio = AudioSegment.from_file(output_file)
                duration = audio.duration_seconds
                print(f"  Sentence {i+1}: Duration {duration:.2f}s")
                # Store index, path, duration for successful TTS
                successful_tts.append({"index": i, "path": output_file, "duration": duration})
            except Exception as e:
                print(f"  ERROR calculating duration for {output_file}: {e}. Skipping sentence {i+1}.")
                # Optionally remove the failed file
                # try: os.remove(output_file) except OSError: pass
        else:
            print(f"  Skipping sentence {i+1} due to TTS generation failure.")

    return successful_tts # Return list of dicts for successful ones

# Run the TTS generation and duration calculation
tts_data = []
try:
    # Run the async function synchronously
    tts_data = asyncio.run(run_tts_for_sentences(sentences, TTS_VOICE, TTS_TEMP_DIR))
except Exception as e:
    print(f"FATAL ERROR during TTS processing: {e}")
    exit()

if not tts_data:
    print("\nERROR: No TTS audio could be generated successfully. Cannot proceed. Exiting.")
    exit()

print(f"\nSuccessfully generated TTS for {len(tts_data)} sentences.")

# --- Re-filter sentences and prepare for image generation ---
# Keep only sentences for which TTS was successful
valid_sentences_info = [] # List of tuples: (original_sentence_text, tts_path, tts_duration)
valid_sentence_indices = {item['index'] for item in tts_data} # Set for quick lookup

for i, sentence in enumerate(sentences):
    if i in valid_sentence_indices:
        # Find the corresponding tts_data entry
        tts_entry = next((item for item in tts_data if item['index'] == i), None)
        if tts_entry:
            valid_sentences_info.append((sentence, tts_entry['path'], tts_entry['duration']))

if not valid_sentences_info:
    print("ERROR: No sentences remaining after TTS validation. Exiting.")
    exit()

print(f"Proceeding with {len(valid_sentences_info)} sentences that have valid TTS audio.")

# --- Generate Prompts and TWO Images per Valid Sentence ---
generated_image_paths = [] # List to hold paths of ALL successfully generated images (2 per sentence)
generated_image_details = [] # List of dicts: {sentence_idx, img_pair_idx (0 or 1), path}
failed_image_gen_sentences = set() # Keep track of sentence indices where image gen failed

# Get script context for g4f (same as before)
try:
    with open(__file__, 'r', encoding='utf-8') as f_script:
        script_content_for_g4f = f_script.read()
except Exception as e:
    print(f"Warning: Could not read script file for g4f context: {e}. Using fallback.")
    script_content_for_g4f = "Video generation script context: Creates dynamic duration clips combining background video, animated overlay image, fire effect, and TTS audio." # Updated fallback

# Loop through the VALID sentences
for sentence_idx, (sentence, tts_path, tts_duration) in enumerate(valid_sentences_info):
    print(f"\nProcessing Sentence {sentence_idx + 1}/{len(valid_sentences_info)}: '{sentence[:60]}...' (TTS Duration: {tts_duration:.2f}s)")

    # Generate TWO prompts and TWO images for this sentence
    for img_pair_idx in range(2): # 0 for first image, 1 for second
        print(f"  Generating Image {img_pair_idx + 1}/2 for sentence {sentence_idx + 1}...")
        image_gen_prompt = None
        # Define path early for potential cleanup in exceptions
        image_file_path = os.path.join(input_folder, f"generated_img_{sentence_idx + 1}_{img_pair_idx + 1}.png")

        # 1. Generate Image Prompt using g4f
        try:
            print("    Requesting image prompt from g4f...")
            # Add context about which part of the sentence this image is for
            g4f_prompt_content = f"""Give me an image generation prompt for the following sentence:
"{sentence}"

This is Image {img_pair_idx + 1} of 2 for this sentence. The final video will split the sentence's TTS audio duration ({tts_duration:.2f}s) across two clips, each showing one image. Tailor the prompt for a compelling visual for this part ({'first half' if img_pair_idx == 0 else 'second half'}) of the sentence's concept.

--- SCRIPT CONTEXT ---
{script_content_for_g4f}
--- END SCRIPT CONTEXT ---

Output only the image generation prompt itself.
"""
            response = g4f_client.chat.completions.create(
                model=g4f_model,
                messages=[{"role": "user", "content": g4f_prompt_content}],
                web_search=False
                # Consider adding timeout=60
            )
            image_gen_prompt = response.choices[0].message.content.strip().strip('"`*')
            if image_gen_prompt.lower().startswith(("image prompt:", "prompt:")):
                 image_gen_prompt = image_gen_prompt.split(":", 1)[-1].strip()
            print(f"    g4f generated prompt: {image_gen_prompt}")

        except Exception as e:
            print(f"    ERROR generating prompt with g4f for sentence {sentence_idx + 1}, image {img_pair_idx + 1}: {e}")
            failed_image_gen_sentences.add(sentence_idx)
            break # Stop trying images for this sentence if prompt fails

        # 2. Generate Image using Pollinations
        if image_gen_prompt:
            try:
                print(f"    Requesting image from Pollinations (Model: {pollinations_model})...")
                image = pollinations_image_model(prompt=image_gen_prompt)
                image.save(file=image_file_path)

                if os.path.exists(image_file_path) and os.path.getsize(image_file_path) > 0:
                    print(f"    Image saved successfully: {image_file_path}")
                    generated_image_paths.append(image_file_path) # Add path to the main list
                    generated_image_details.append({
                        "sentence_idx": sentence_idx,
                        "img_pair_idx": img_pair_idx,
                        "path": image_file_path
                    })
                else:
                     print(f"    ERROR: Pollinations generation/saving failed for {image_file_path}.")
                     failed_image_gen_sentences.add(sentence_idx)
                     # *** SYNTAX FIX HERE ***
                     if os.path.exists(image_file_path):
                         try:
                             os.remove(image_file_path)
                         except OSError:
                             pass # Ignore error if removal fails
                     break # Stop trying images for this sentence

            except Exception as e:
                print(f"    ERROR generating/saving image with Pollinations for sentence {sentence_idx + 1}, image {img_pair_idx + 1}: {e}")
                failed_image_gen_sentences.add(sentence_idx)
                # *** SYNTAX FIX HERE ***
                if os.path.exists(image_file_path):
                    try:
                        os.remove(image_file_path)
                    except OSError:
                        pass # Ignore error if removal fails
                break # Stop trying images for this sentence
        else:
            print(f"    Skipping image generation for sentence {sentence_idx + 1}, image {img_pair_idx + 1} because prompt generation failed.")
            failed_image_gen_sentences.add(sentence_idx)
            break # Stop trying images for this sentence

    # If this sentence failed image gen, need to clean up partially generated stuff for it
    if sentence_idx in failed_image_gen_sentences:
         print(f"  Image generation failed for sentence {sentence_idx+1}. This sentence will be excluded from the video.")
         # Remove any successfully generated image/details for this sentence index
         generated_image_paths = [p for p in generated_image_paths if not p.startswith(os.path.join(input_folder, f"generated_img_{sentence_idx + 1}_"))]
         generated_image_details = [d for d in generated_image_details if d["sentence_idx"] != sentence_idx]
         # Mark the corresponding TTS data entry for removal later
         # Find the entry in valid_sentences_info by index and mark it (e.g., set to None)
         # This assumes valid_sentences_info order matches sentence_idx
         if sentence_idx < len(valid_sentences_info):
             valid_sentences_info[sentence_idx] = None # Mark for removal


# --- Final Filtering after Image Generation ---
# Remove entries marked None in valid_sentences_info
original_valid_count = len(valid_sentences_info)
valid_sentences_info = [info for info in valid_sentences_info if info is not None]

if len(valid_sentences_info) < original_valid_count:
    print(f"\nRemoved {original_valid_count - len(valid_sentences_info)} sentences due to image generation failures.")
    # Adjust generated_image_details indices if needed? No, they map to the *original* sentence_idx which is now filtered out.

# Post-Generation Check
if not generated_image_paths or not valid_sentences_info:
    print("\nERROR: No images or valid sentences remaining after generation steps. Cannot proceed. Exiting.")
    # Clean up any TTS files generated if exiting here
    if os.path.exists(TTS_TEMP_DIR):
        print("Cleaning up temporary TTS files...")
        # Using shutil.rmtree is simpler if permissions allow
        import shutil
        try:
            shutil.rmtree(TTS_TEMP_DIR)
        except OSError as e:
            print(f"  Warning: Could not remove TTS temp directory {TTS_TEMP_DIR}: {e}")
    exit()

# Determine the number of segments based on successfully generated images
# Should be exactly 2 * len(valid_sentences_info)
num_segments = len(generated_image_paths)

# Adjust generated_image_details to only include images for the *remaining* valid sentences
valid_final_sentence_indices = {idx for idx, _ in enumerate(valid_sentences_info)} # Get indices of sentences we are keeping
generated_image_details = [d for d in generated_image_details if d["sentence_idx"] in valid_final_sentence_indices]
# We also need to remap the sentence_idx in generated_image_details to the new index in the filtered valid_sentences_info
# Create a mapping from old index to new index
old_to_new_idx_map = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(zip(valid_final_sentence_indices, valid_sentences_info))}
for detail in generated_image_details:
    detail["sentence_idx"] = old_to_new_idx_map[detail["sentence_idx"]]

# Now re-check counts
num_segments = len(generated_image_paths) # Should still be the same physical files
if num_segments != 2 * len(valid_sentences_info):
    print(f"ERROR: Mismatch after filtering. Expected segments ({2 * len(valid_sentences_info)}) vs generated images ({num_segments}). Exiting.")
    # Add cleanup here too
    if os.path.exists(TTS_TEMP_DIR):
        import shutil
        try: shutil.rmtree(TTS_TEMP_DIR)
        except OSError as e: print(f"  Warning: Could not remove TTS temp directory {TTS_TEMP_DIR}: {e}")
    exit()

print(f"\nSuccessfully generated {num_segments} images for {len(valid_sentences_info)} sentences. Proceeding to create {num_segments} video segments.")


# =====================================================
# Part 1: Create Silent Clips with DYNAMIC Duration, Overlay, Animation & Fire
# =====================================================
print("\n--- Part 1: Generating Silent Clips ---")
silent_clip_paths = []

# *** MODIFIED LOOP ***
# Loop through the TOTAL number of segments (images)
for i in range(num_segments):
    # Determine which original sentence this segment corresponds to using remapped index
    # Find the detail entry for this segment index 'i' (0 to num_segments-1)
    # This relies on generated_image_details being correctly sorted or searchable
    # Let's assume generated_image_details is now correctly mapped and ordered implicitly
    # sentence_idx is the index within the *filtered* valid_sentences_info
    sentence_idx = math.floor(i / 2)
    # clip_pair_idx is 0 or 1 for the image within that sentence pair
    clip_pair_idx = i % 2

    # Get sentence info (text, tts_path, tts_duration) from the filtered list
    try:
        sentence_text, sentence_tts_path, sentence_tts_duration = valid_sentences_info[sentence_idx]
    except IndexError:
         print(f"ERROR: Index out of bounds accessing valid_sentences_info for segment {i+1} (sentence index {sentence_idx}). Skipping.")
         continue

    # Get the corresponding generated image path from generated_image_details
    # We need to find the detail matching the *remapped* sentence_idx and clip_pair_idx
    img_detail = next((d for d in generated_image_details if d["sentence_idx"] == sentence_idx and d["img_pair_idx"] == clip_pair_idx), None)

    if img_detail is None:
        print(f"ERROR: Could not find image details for segment {i+1} (Remapped Sentence Idx {sentence_idx}, Clip {clip_pair_idx+1}). Skipping.")
        continue
    overlay_img_path = img_detail["path"]

    # *** Calculate DYNAMIC Clip Duration and Frames ***
    current_clip_duration = sentence_tts_duration / 2.0
    min_duration_threshold = animation_duration_seconds + 0.1
    if current_clip_duration < min_duration_threshold:
        # print(f"Warning: Calculated clip duration ({current_clip_duration:.2f}s) for segment {i+1} is very short. Clamping to {min_duration_threshold:.2f}s.")
        current_clip_duration = min_duration_threshold
    if current_clip_duration <= 0:
        # print(f"Warning: Calculated clip duration for segment {i+1} is zero or negative. Setting to 0.1s minimum.")
        current_clip_duration = 0.1 # Absolute minimum

    total_frames = max(1, int(current_clip_duration * fps))

    # *** Calculate Background Video Index (Looping based on overall segment index i) ***
    bg_video_index = (i % num_background_videos) + 1
    bg_video_path = os.path.join(input_folder, f"{bg_video_index}.mp4")

    # Define the output silent clip path using the overall segment number (1-based)
    segment_num = i + 1
    silent_clip = os.path.join(input_folder, f"clip_{segment_num}_silent.mp4")

    print(f"\nProcessing Segment {segment_num}/{num_segments} (Sentence {sentence_idx + 1}, Clip {clip_pair_idx + 1}, Duration: {current_clip_duration:.2f}s, Frames: {total_frames})")
    print(f" -> Using Img: {os.path.basename(overlay_img_path)}, BG: {bg_video_index}.mp4")

    # --- Input File Checks ---
    if not os.path.exists(bg_video_path):
        print(f"ERROR: Background video not found: {bg_video_path}. Skipping segment {segment_num}.")
        continue
    if not os.path.exists(overlay_img_path):
         print(f"ERROR: Overlay image not found: {overlay_img_path} (should not happen here). Skipping segment {segment_num}.")
         continue

    # --- Open Video Captures ---
    cap = cv2.VideoCapture(bg_video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open background video: {bg_video_path}. Skipping segment {segment_num}.")
        continue
    bg_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fire_cap = cv2.VideoCapture(fire_overlay_video_path)
    if not fire_cap.isOpened():
        print(f"ERROR: Cannot open fire overlay video: {fire_overlay_video_path}. Skipping segment {segment_num}.")
        cap.release()
        continue
    fire_frame_count = int(fire_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fire_frame_count <= 0:
         print(f"ERROR: Fire overlay video {fire_overlay_video_path} has no frames. Skipping segment {segment_num}.")
         cap.release()
         fire_cap.release()
         continue

    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(silent_clip, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        print(f"ERROR: Failed to open VideoWriter for {silent_clip}. Skipping segment {segment_num}.")
        cap.release()
        fire_cap.release()
        continue

    # --- Load and Prepare Overlay Image ---
    overlay_orig = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    if overlay_orig is None:
        print(f"ERROR: Cannot load overlay image: {overlay_img_path}. Skipping segment {segment_num}.")
        cap.release(); fire_cap.release(); out.release(); continue # Clean up resources

    # Ensure alpha channel and resize (same logic as before)
    if len(overlay_orig.shape) == 2: overlay_orig = cv2.cvtColor(overlay_orig, cv2.COLOR_GRAY2BGRA)
    elif overlay_orig.shape[2] == 3: overlay_orig = cv2.cvtColor(overlay_orig, cv2.COLOR_BGR2BGRA)
    max_resize_width = output_width - 2 * margin
    max_resize_height = output_height - 2 * margin
    orig_h, orig_w = overlay_orig.shape[:2]
    if orig_w == 0 or orig_h == 0:
        print(f"ERROR: Invalid overlay image dimensions for {overlay_img_path}. Skipping segment {segment_num}.")
        cap.release(); fire_cap.release(); out.release(); continue
    scale_factor = min(max_resize_width / orig_w, max_resize_height / orig_h)
    new_w = max(1, int(orig_w * scale_factor))
    new_h = max(1, int(orig_h * scale_factor))
    try:
        overlay_resized = cv2.resize(overlay_orig, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4 if scale_factor < 1.0 else cv2.INTER_LINEAR)
    except Exception as e:
        print(f"ERROR: Failed to resize overlay image {overlay_img_path}: {e}. Skipping segment {segment_num}.")
        cap.release(); fire_cap.release(); out.release(); continue
    # print(f"  Overlay Image resized to {new_w}x{new_h}")

    # --- Animation Setup (Cycles based on overall segment index 'i') ---
    target_x = margin + (max_resize_width - new_w) // 2
    target_y = margin + (max_resize_height - new_h) // 2
    animation_type = i % 4 # Cycle animation based on overall segment index
    start_x, start_y = target_x, target_y

    use_animation = animation_duration_frames > 0 and total_frames > animation_duration_frames
    if not use_animation and animation_duration_frames > 0:
         print(f"  Note: Clip duration ({current_clip_duration:.2f}s) too short for animation ({animation_duration_seconds:.2f}s). Animation disabled.")

    if use_animation:
        anim_map = {0: "Top", 1: "Left", 2: "Bottom", 3: "Right"}
        print(f"  Animating image slide from {anim_map[animation_type]} ({animation_duration_seconds}s)")
        if animation_type == 0: start_y = -new_h       # Start above screen
        elif animation_type == 1: start_x = -new_w       # Start left of screen
        elif animation_type == 2: start_y = output_height # Start below screen
        elif animation_type == 3: start_x = output_width  # Start right of screen

    # --- Frame Processing Loop (Runs for dynamic total_frames) ---
    processed_frames = 0
    last_good_bg_frame = None
    fire_frame_index = -1

    while processed_frames < total_frames:
        # --- Read Background Frame ---
        ret_bg, bg_frame = cap.read()
        current_bg_frame_for_compositing = None
        if not ret_bg:
            # If background ends, loop it or use last frame
            if bg_frame_count > 0: # Check if we ever read a frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back to the start
                ret_bg, bg_frame = cap.read() # Try reading the first frame again
                if not ret_bg: # If looping still fails, use last good frame or black
                   if last_good_bg_frame is None: last_good_bg_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                   current_bg_frame_for_compositing = last_good_bg_frame.copy()
                   if processed_frames % fps == 0: print(f"\nWarning: BG loop failed. Using last frame/black.", end='')
                # else: Process the first frame read after looping (handled below)
            else: # Background video was likely empty or unreadable from start
                if last_good_bg_frame is None: last_good_bg_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                current_bg_frame_for_compositing = last_good_bg_frame.copy()
                if processed_frames == 0: print(f"\nERROR: BG video {bg_video_path} unreadable. Using black.")

        # Process the successfully read frame (either initial read or after looping)
        if ret_bg and current_bg_frame_for_compositing is None: # Check if not already handled by error cases
             try:
                 bg_frame_resized = cv2.resize(bg_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                 last_good_bg_frame = bg_frame_resized # Store the successful resize
                 current_bg_frame_for_compositing = bg_frame_resized # Use it
             except Exception as e:
                 if last_good_bg_frame is None: # First frame resize failed
                      print(f"\nERROR: Failed resize first BG frame for segment {segment_num}: {e}. Using black fallback.")
                      last_good_bg_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                 # Use the last known good frame (or the newly created black frame)
                 current_bg_frame_for_compositing = last_good_bg_frame.copy()
                 if processed_frames % fps == 0: print(f"\nWarning: Failed resize BG frame {processed_frames}: {e}. Using last good frame.", end='')

        # --- Read Fire Frame (Looping) ---
        fire_frame_index = (fire_frame_index + 1) % fire_frame_count
        if fire_frame_index == 0: fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret_fire, fire_frame = fire_cap.read()
        fire_frame_resized = None # Define scope
        if not ret_fire:
             if processed_frames % fps == 0 : print(f"\nWarning: Failed read fire frame {fire_frame_index}. Attempting reset/reread.", end='')
             fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             ret_fire, fire_frame = fire_cap.read()
             if not ret_fire:
                 print("\nERROR: Cannot read fire video even after reset. Using black frame.")
                 fire_frame_resized = np.zeros((output_height, output_width, 3), dtype=np.uint8)
             else: fire_frame_index = 0
        # Resize fire frame if successfully read (or is black fallback)
        if fire_frame_resized is None and ret_fire: # Only resize if not already black
             try:
                 fire_frame_resized = cv2.resize(fire_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
             except Exception as e:
                 if processed_frames % 60 == 0: print(f"\nWarning: Could not resize fire frame {fire_frame_index}: {e}. Using black frame.", end='')
                 fire_frame_resized = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        elif fire_frame_resized is None: # Handles case where initial read failed and fallback wasn't assigned
             fire_frame_resized = np.zeros((output_height, output_width, 3), dtype=np.uint8)


        # --- Compositing ---
        composite = current_bg_frame_for_compositing.copy()

        # --- Calculate Image Overlay Position ---
        current_x, current_y = target_x, target_y
        if use_animation and processed_frames < animation_duration_frames:
            progress = min(1.0, (processed_frames + 1) / animation_duration_frames)
            current_x = int(start_x + progress * (target_x - start_x))
            current_y = int(start_y + progress * (target_y - start_y))

        # --- Place Image Overlay (Safe Placement with Alpha Blending) ---
        x1, y1, x2, y2 = current_x, current_y, current_x + new_w, current_y + new_h
        frame_x1, frame_y1 = max(x1, 0), max(y1, 0)
        frame_x2, frame_y2 = min(x2, output_width), min(y2, output_height)
        if frame_x1 < frame_x2 and frame_y1 < frame_y2:
            overlay_x1, overlay_y1 = frame_x1 - x1, frame_y1 - y1
            overlay_w, overlay_h = frame_x2 - frame_x1, frame_y2 - frame_y1
            overlay_x2, overlay_y2 = overlay_x1 + overlay_w, overlay_y1 + overlay_h
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
                        alpha_mask = cv2.cvtColor(alpha.astype(np.float32), cv2.COLOR_GRAY2BGR)
                        blended_roi = (overlay_part[:, :, :3] * alpha_mask) + (roi * (1.0 - alpha_mask))
                        composite[frame_y1:frame_y2, frame_x1:frame_x2] = blended_roi.astype(np.uint8)
                except Exception as e:
                     if processed_frames % 60 == 0: print(f"\nWarning: Error during overlay blend frame {processed_frames}: {e}")


        # --- Add Fire Overlay (Additive Blending) ---
        try:
             if composite.shape[2] == 4: composite = cv2.cvtColor(composite, cv2.COLOR_BGRA2BGR)
             if fire_frame_resized is not None and fire_frame_resized.shape[2] == 4: fire_frame_resized = cv2.cvtColor(fire_frame_resized, cv2.COLOR_BGRA2BGR)
             if fire_frame_resized is not None and composite.shape == fire_frame_resized.shape:
                 composite = cv2.add(composite, fire_frame_resized)
        except Exception as e:
             if processed_frames % 60 == 0: print(f"\nWarning: Error during fire overlay frame {processed_frames}: {e}", end='')

        # --- Write Frame ---
        try:
            out.write(composite)
        except Exception as e:
             print(f"\nFATAL ERROR: Failed to write frame {processed_frames} for segment {segment_num}: {e}")
             processed_frames = total_frames + 1 # Force loop exit
             break

        processed_frames += 1
        print(f"  Segment {segment_num}: Processing Frame {processed_frames}/{total_frames}...", end='\r')


    # --- End Frame Processing Loop ---
    print(f"  Segment {segment_num}: Processed {min(processed_frames, total_frames)} frames. {' ' * 20}")
    cap.release()
    fire_cap.release()
    out.release()

    if processed_frames > total_frames : # Loop aborted
        print(f" Failed silent clip generation for segment {segment_num} due to write error.")
        # *** SYNTAX FIX HERE ***
        if os.path.exists(silent_clip):
            try:
                os.remove(silent_clip)
            except OSError:
                pass # Ignore error
    elif processed_frames == total_frames:
        print(f" Silent clip saved: {silent_clip}")
        silent_clip_paths.append(silent_clip)
    else: # Should not happen unless error before loop
        print(f" Incomplete silent clip for segment {segment_num}. Not adding.")
        # *** SYNTAX FIX HERE ***
        if os.path.exists(silent_clip):
            try:
                os.remove(silent_clip)
            except OSError:
                pass # Ignore error


# =====================================================
# Part 2: Add Sound Effect Using FFmpeg - SKIPPED
# =====================================================
print("\n--- Part 2: Adding Specific Sound Effect (Skipped for TTS Workflow) ---")


# =====================================================
# Part 3: Combine Final Clips (Video Only) Using OpenCV
# =====================================================
print("\n--- Part 3: Combining Video Tracks ---")
combined_video_silent = os.path.join(input_folder, "combined_video_silent.mp4")
combiner_initialized = False
out_combined = None

if not silent_clip_paths:
    print("No silent clips were successfully created in Part 1. Skipping video combination.")
else:
    try:
        # Sort silent clips by number in filename
        silent_clip_paths.sort(key=lambda x: int(re.search(r'_(\d+)_silent\.mp4', os.path.basename(x)).group(1)))
        print(" Combining video from clips:", [os.path.basename(p) for p in silent_clip_paths])

        # Get parameters from the first clip
        first_clip_path = silent_clip_paths[0]
        cap_check = cv2.VideoCapture(first_clip_path)
        if not cap_check.isOpened(): raise IOError(f"Could not open first clip {first_clip_path}")

        comb_width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        comb_height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        comb_fps = cap_check.get(cv2.CAP_PROP_FPS)
        cap_check.release()

        # Validate parameters
        if not (output_width * 0.99 < comb_width < output_width * 1.01): # Stricter check?
             print(f"Warning: Width {comb_width} vs target {output_width}. Using target.")
             comb_width = output_width
        if not (output_height * 0.99 < comb_height < output_height * 1.01):
             print(f"Warning: Height {comb_height} vs target {output_height}. Using target.")
             comb_height = output_height
        # Use configured FPS for consistency
        if abs(comb_fps - fps) > 1: # Allow minor float differences
             print(f"Warning: FPS mismatch {comb_fps:.2f} vs target {fps}. Using target FPS {fps}.")
        comb_fps = fps # Use the target FPS defined in config

        print(f" Combining with parameters: {comb_width}x{comb_height} @ {comb_fps:.2f} fps")

        fourcc_comb = cv2.VideoWriter_fourcc(*"mp4v")
        out_combined = cv2.VideoWriter(combined_video_silent, fourcc_comb, comb_fps, (comb_width, comb_height))
        if not out_combined.isOpened(): raise IOError(f"Could not open VideoWriter for {combined_video_silent}")
        combiner_initialized = True

        total_combined_frames = 0
        for clip_idx, clip_path in enumerate(silent_clip_paths):
            segment_num = clip_idx + 1
            print(f"  Appending segment {segment_num}: {os.path.basename(clip_path)}...")
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened(): print(f"  ERROR: Cannot open {clip_path}. Skipping."); continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w != comb_width or h != comb_height:
                print(f"  ERROR: Dimension mismatch in {os.path.basename(clip_path)}. Skipping."); cap.release(); continue

            clip_frame_count = 0
            frame_read_error = False
            while True:
                ret, frame = cap.read()
                if not ret: break
                try: out_combined.write(frame); clip_frame_count += 1
                except Exception as e: print(f"\n  ERROR writing frame {clip_frame_count+1}: {e}"); frame_read_error = True; break
            cap.release()
            if frame_read_error: combiner_initialized = False; break
            else: total_combined_frames += clip_frame_count; print(f"    -> Appended {clip_frame_count} frames.")

        if combiner_initialized: print(f" Combined silent video saved: {combined_video_silent} ({total_combined_frames} frames)")

    except Exception as e:
        print(f"Error during video combination: {e}")
        combiner_initialized = False; combined_video_silent = None
    finally:
         if out_combined is not None and out_combined.isOpened(): out_combined.release()


# =====================================================
# Part 4: Concatenate TTS Audio and Merge with Combined Video
# =====================================================
print("\n--- Part 4: Concatenating TTS Audio and Final Merge ---")

if not combiner_initialized or combined_video_silent is None or not os.path.exists(combined_video_silent):
     print("Combined silent video was not created successfully. Skipping audio concatenation and merge.")
elif not tts_data: # This now refers to the original list from TTS generation
     print("No valid TTS data available. Skipping audio concatenation and merge.")
elif not ffmpeg_ok:
     print("FFmpeg not found or not working. Skipping audio concatenation and merge.")
else:
    # Get the list of successful TTS file paths IN THE CORRECT ORDER
    # Use the paths from the *filtered* valid_sentences_info list
    valid_sentences_info.sort(key=lambda item: item[0]) # Sort based on original sentence text maybe? Or rely on index?
    # Let's rely on the indices being correct from the filtering stage
    ordered_tts_files = [item[1] for item in valid_sentences_info] # Get the path (index 1)

    if len(ordered_tts_files) != len(valid_sentences_info):
         print("ERROR: Mismatch between valid sentences and TTS file list length. Cannot proceed.")
         merge_step_ok = False
    else:
        audio_list_file = os.path.join(TTS_TEMP_DIR, "tts_audio_list.txt")
        combined_audio = os.path.join(TTS_TEMP_DIR, "combined_tts_audio.aac")
        final_output = os.path.join(input_folder, "final_output.mp4")
        merge_step_ok = True # Assume success initially

        try:
            # Create audio list file
            print(f" Creating audio list file: {audio_list_file}")
            with open(audio_list_file, "w", encoding='utf-8') as f:
                for tts_file_path in ordered_tts_files:
                    safe_path = tts_file_path.replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

            # Concatenate TTS audio files
            concat_audio_cmd = [
                ffmpeg_executable, "-y", "-f", "concat", "-safe", "0",
                "-i", audio_list_file, "-c:a", "aac", "-b:a", "128k",
                combined_audio
            ]
            print(" Concatenating TTS audio files...")
            startupinfo = None
            if os.name == 'nt': startupinfo = subprocess.STARTUPINFO(); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.run(concat_audio_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', startupinfo=startupinfo)

            if not (os.path.exists(combined_audio) and os.path.getsize(combined_audio) > 0):
                 print(f"  ERROR: TTS audio concatenation failed: {combined_audio}")
                 print(f"  FFmpeg stderr:\n{result.stderr}") # Print stderr for errors
                 merge_step_ok = False
            else:
                 print(f"  Combined TTS audio saved: {combined_audio}")

            # Merge final video and concatenated TTS audio
            if merge_step_ok:
                merge_cmd = [
                    ffmpeg_executable, "-y",
                    "-i", combined_video_silent, # Input 0: Video
                    "-i", combined_audio,        # Input 1: Audio
                    "-c:v", "copy", "-c:a", "copy", # Copy streams
                    "-map", "0:v:0", "-map", "1:a:0", # Map streams
                    "-shortest",
                    final_output
                ]
                print(" Merging final video and audio...")
                result = subprocess.run(merge_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', startupinfo=startupinfo)

                if not (os.path.exists(final_output) and os.path.getsize(final_output) > 0):
                     print(f"  ERROR: Final merge failed: {final_output}")
                     print(f"  FFmpeg stderr:\n{result.stderr}") # Print stderr for errors
                     merge_step_ok = False
                else:
                     total_tts_duration = sum(item[2] for item in valid_sentences_info) # Duration is index 2
                     print(f"\n Final video (expected duration: ~{total_tts_duration:.1f}s) saved as: {final_output}")

        except subprocess.CalledProcessError as e:
            print(f"\nERROR running FFmpeg:")
            print(f"  Command: {' '.join(e.cmd)}")
            print(f"  Return Code: {e.returncode}")
            print(f"  Stderr: {e.stderr}") # Crucial for FFmpeg errors
            merge_step_ok = False
        except Exception as e:
             print(f"\nAn error occurred during audio processing or final merge: {e}")
             merge_step_ok = False

        # --- Cleanup ---
        # Use shutil for directory removal if possible, fall back to os for files
        import shutil
        if merge_step_ok and os.path.exists(final_output):
            print("\n Cleaning up intermediate files...")
            cleanup_files = generated_image_paths + silent_clip_paths
            if os.path.exists(combined_video_silent): cleanup_files.append(combined_video_silent)

            removed_count = 0
            failed_removals = []
            for f in cleanup_files:
                 if os.path.exists(f):
                     try: os.remove(f); removed_count += 1
                     except OSError as e: print(f"  Warning: Could not remove {f}: {e}"); failed_removals.append(f)

            # Remove the temporary TTS directory
            if os.path.exists(TTS_TEMP_DIR):
                print(f" Removing temporary TTS directory: {TTS_TEMP_DIR}")
                try:
                    shutil.rmtree(TTS_TEMP_DIR)
                except OSError as e:
                    print(f"  Warning: Could not remove temporary TTS directory {TTS_TEMP_DIR}: {e}")
                    failed_removals.append(TTS_TEMP_DIR)

            print(f" Cleanup complete (attempted removal of ~{len(cleanup_files) + len(ordered_tts_files) + 2} items).")
            if failed_removals: print(f"  Could not remove the following items: {failed_removals}")
        elif merge_step_ok and not os.path.exists(final_output):
             print("\nFinal merge command ran but output file is missing. Skipping cleanup.")
             # Keep intermediate files for debugging
        else: # merge_step_ok is False
             print("\nSkipping cleanup as final merge failed or did not run.")
             print("Intermediate files may remain in:")
             print(f" - {input_folder}")
             if os.path.exists(TTS_TEMP_DIR): print(f" - {TTS_TEMP_DIR}")
        # --- End Cleanup ---


print("\n--- Script finished ---")
