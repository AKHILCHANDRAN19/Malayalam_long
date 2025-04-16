# -*- coding: utf-8 -*- # Recommended for path handling if non-ASCII chars are possible
import cv2
import numpy as np
import os
import subprocess
import math # Needed for ceiling function and dynamic duration calculation
import re   # For sentence splitting
import time # For potential delays
import asyncio # For edge-tts
import shutil # For directory removal

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
animation_duration_frames = int(animation_duration_seconds * fps)

# --- Resource Paths ---
sound_file = os.path.join(input_folder, "Whoos 3.mp3") # WHOOSH sound at the start of each clip
fire_overlay_video_path = os.path.join(input_folder, "Fire.mp4") # Path to fire overlay video
FINAL_OUTPUT_FILENAME = "final_output_with_whoosh.mp4" # Name for the final video

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
# Helper Functions
# =====================================================
def run_ffmpeg_command(cmd_list, description="FFmpeg command"):
    """Runs an FFmpeg command using subprocess, handling startup info and errors."""
    print(f"  Running {description}: {' '.join(cmd_list)}")
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result = subprocess.run(cmd_list, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', startupinfo=startupinfo)
        # print(f"  FFmpeg stdout:\n{result.stdout}") # Optional: Log stdout
        if result.stderr:
             print(f"  FFmpeg stderr:\n{result.stderr}") # Log stderr even on success (might contain warnings)
        return True, result
    except subprocess.CalledProcessError as e:
        print(f"  ERROR running {description}:")
        print(f"    Command: {' '.join(e.cmd)}")
        print(f"    Return Code: {e.returncode}")
        print(f"    Stderr: {e.stderr}") # Crucial for FFmpeg errors
        return False, e
    except Exception as e:
        print(f"  UNEXPECTED ERROR running {description}: {e}")
        return False, e

# =====================================================
# Setup & Input Checks
# =====================================================
print("\n--- Setup & Input Checks ---")

# Create input folder if it doesn't exist
if not os.path.exists(input_folder):
    try:
        os.makedirs(input_folder)
        print(f"Created input folder: {input_folder}")
        print(f"Please ensure your {num_background_videos} background MP4s (1.mp4, 2.mp4, ...), Fire.mp4, and Whoos 3.mp3 are in this folder.")
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
if not os.path.exists(sound_file): # Whoosh sound is now essential
    print(f"ERROR: Sound effect file not found: {sound_file}")
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
     print(f"ERROR: FFmpeg check failed: {e}. FFmpeg (or libav) is required for pydub and video/audio processing. Ensure it's installed and in PATH.")
     essential_files_exist = False # FFmpeg is essential now

if not essential_files_exist:
     print("Essential base files (fire video, background videos, Whoos 3.mp3) or FFmpeg missing. Please check paths and files. Exiting.")
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
        referrer="Codepoli_tts_v2.py" # Updated referrer
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
        # Use 1-based index for filename for clarity if needed, but store 0-based index internally
        output_file = os.path.join(output_dir, f"tts_{i+1}.mp3") # Consistent naming
        output_files.append(output_file)
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
                print(f"  Sentence {i+1} (tts_{i+1}.mp3): Duration {duration:.3f}s") # More precision
                # Store index, path, duration for successful TTS
                successful_tts.append({"index": i, "path": output_file, "duration": duration})
            except Exception as e:
                print(f"  ERROR calculating duration for {output_file}: {e}. Skipping sentence {i+1}.")
                if os.path.exists(output_file):
                    try: os.remove(output_file)
                    except OSError: pass
        else:
            print(f"  Skipping sentence {i+1} due to TTS generation failure.")

    return successful_tts # Return list of dicts for successful ones

# Run the TTS generation and duration calculation
tts_data = []
try:
    tts_data = asyncio.run(run_tts_for_sentences(sentences, TTS_VOICE, TTS_TEMP_DIR))
except Exception as e:
    print(f"FATAL ERROR during TTS processing: {e}")
    exit()

if not tts_data:
    print("\nERROR: No TTS audio could be generated successfully. Cannot proceed. Exiting.")
    exit()

print(f"\nSuccessfully generated TTS for {len(tts_data)} sentences.")

# --- Re-filter sentences and prepare for image generation ---
valid_sentences_info = [] # List of tuples: (original_sentence_text, tts_path, tts_duration)
valid_sentence_indices = {item['index'] for item in tts_data} # Set for quick lookup

for i, sentence in enumerate(sentences):
    if i in valid_sentence_indices:
        tts_entry = next((item for item in tts_data if item['index'] == i), None)
        if tts_entry:
            # Store ORIGINAL index along with info
            valid_sentences_info.append({"original_index": i, "text": sentence, "tts_path": tts_entry['path'], "tts_duration": tts_entry['duration']})

if not valid_sentences_info:
    print("ERROR: No sentences remaining after TTS validation. Exiting.")
    exit()

print(f"Proceeding with {len(valid_sentences_info)} sentences that have valid TTS audio.")

# --- Generate Prompts and TWO Images per Valid Sentence ---
generated_image_paths = [] # List to hold paths of ALL successfully generated images
generated_image_details = [] # List of dicts: {original_sentence_idx, img_pair_idx (0 or 1), path}
failed_image_gen_sentences_indices = set() # Keep track of *original* sentence indices where image gen failed

# Get script context for g4f
try:
    with open(__file__, 'r', encoding='utf-8') as f_script:
        script_content_for_g4f = f_script.read()
except Exception as e:
    print(f"Warning: Could not read script file for g4f context: {e}. Using fallback.")
    script_content_for_g4f = "Video generation script context: Creates dynamic duration clips combining background video, animated overlay image, fire effect, Whoosh sound, and split TTS audio." # Updated fallback

# Loop through the VALID sentences (using enumerate to get a sequential index 0, 1, 2...)
for valid_idx, sentence_info in enumerate(valid_sentences_info):
    original_sentence_idx = sentence_info["original_index"]
    sentence = sentence_info["text"]
    tts_duration = sentence_info["tts_duration"]

    print(f"\nProcessing Sentence (Original Index {original_sentence_idx + 1}, Valid Index {valid_idx + 1}/{len(valid_sentences_info)}): '{sentence[:60]}...' (TTS Duration: {tts_duration:.3f}s)")

    # Generate TWO prompts and TWO images for this sentence
    for img_pair_idx in range(2): # 0 for first image, 1 for second
        print(f"  Generating Image {img_pair_idx + 1}/2 for sentence {original_sentence_idx + 1}...")
        image_gen_prompt = None
        # Use original sentence index in filename for easier tracking if debugging
        image_file_path = os.path.join(input_folder, f"generated_img_orig_{original_sentence_idx + 1}_{img_pair_idx + 1}.png")

        # 1. Generate Image Prompt using g4f
        try:
            print("    Requesting image prompt from g4f...")
            g4f_prompt_content = f"""Give me an image generation prompt for the following sentence:
"{sentence}"

This is Image {img_pair_idx + 1} of 2 for this sentence. The final video will split the sentence's TTS audio duration ({tts_duration:.3f}s) across two clips, each showing one image. Tailor the prompt for a compelling visual for this part ({'first half' if img_pair_idx == 0 else 'second half'}) of the sentence's concept.

--- SCRIPT CONTEXT ---
{script_content_for_g4f}
--- END SCRIPT CONTEXT ---

Output only the image generation prompt itself.
"""
            response = g4f_client.chat.completions.create(
                model=g4f_model,
                messages=[{"role": "user", "content": g4f_prompt_content}],
                web_search=False,
                timeout=90 # Increased timeout
            )
            image_gen_prompt = response.choices[0].message.content.strip().strip('"`*')
            if not image_gen_prompt: raise ValueError("g4f returned empty prompt")
            if image_gen_prompt.lower().startswith(("image prompt:", "prompt:")):
                 image_gen_prompt = image_gen_prompt.split(":", 1)[-1].strip()
            print(f"    g4f generated prompt: {image_gen_prompt}")

        except Exception as e:
            print(f"    ERROR generating prompt with g4f for sentence {original_sentence_idx + 1}, image {img_pair_idx + 1}: {e}")
            failed_image_gen_sentences_indices.add(original_sentence_idx)
            break # Stop trying images for this sentence if prompt fails

        # 2. Generate Image using Pollinations
        if image_gen_prompt:
            try:
                print(f"    Requesting image from Pollinations (Model: {pollinations_model})...")
                image = pollinations_image_model(prompt=image_gen_prompt)
                image.save(file=image_file_path)

                if os.path.exists(image_file_path) and os.path.getsize(image_file_path) > 0:
                    print(f"    Image saved successfully: {image_file_path}")
                    generated_image_paths.append(image_file_path)
                    generated_image_details.append({
                        "original_sentence_idx": original_sentence_idx,
                        "img_pair_idx": img_pair_idx, # 0 or 1
                        "path": image_file_path
                    })
                else:
                     print(f"    ERROR: Pollinations generation/saving failed for {image_file_path}.")
                     failed_image_gen_sentences_indices.add(original_sentence_idx)
                     if os.path.exists(image_file_path):
                         try: os.remove(image_file_path)
                         except OSError: pass
                     break # Stop trying images for this sentence

            except Exception as e:
                print(f"    ERROR generating/saving image with Pollinations for sentence {original_sentence_idx + 1}, image {img_pair_idx + 1}: {e}")
                failed_image_gen_sentences_indices.add(original_sentence_idx)
                if os.path.exists(image_file_path):
                    try: os.remove(image_file_path)
                    except OSError: pass
                break # Stop trying images for this sentence
        else:
            # This case should technically not be reached if prompt generation failed due to the 'break' above
            print(f"    Skipping image generation for sentence {original_sentence_idx + 1}, image {img_pair_idx + 1} because prompt generation failed.")
            failed_image_gen_sentences_indices.add(original_sentence_idx)
            break # Stop trying images for this sentence

    # If this sentence failed image gen at any point, need to clean up
    if original_sentence_idx in failed_image_gen_sentences_indices:
         print(f"  Image generation failed for sentence {original_sentence_idx+1}. This sentence will be excluded from the video.")
         # Remove any successfully generated image/details for this sentence index
         generated_image_paths = [p for p in generated_image_paths if not p.startswith(os.path.join(input_folder, f"generated_img_orig_{original_sentence_idx + 1}_"))]
         generated_image_details = [d for d in generated_image_details if d["original_sentence_idx"] != original_sentence_idx]
         # Mark the corresponding entry in valid_sentences_info for removal later by setting to None
         # Find by original_index and mark
         for idx, info in enumerate(valid_sentences_info):
             if info is not None and info["original_index"] == original_sentence_idx:
                 valid_sentences_info[idx] = None
                 break


# --- Final Filtering after Image Generation ---
original_valid_count = len([info for info in valid_sentences_info if info is not None])
valid_sentences_info = [info for info in valid_sentences_info if info is not None] # Filter out None entries

if len(valid_sentences_info) < original_valid_count:
    print(f"\nRemoved {original_valid_count - len(valid_sentences_info)} sentences due to image generation failures.")

# Post-Generation Check
if not generated_image_details or not valid_sentences_info:
    print("\nERROR: No images or valid sentences remaining after generation steps. Cannot proceed. Exiting.")
    if os.path.exists(TTS_TEMP_DIR):
        print("Cleaning up temporary TTS files...")
        try: shutil.rmtree(TTS_TEMP_DIR)
        except OSError as e: print(f"  Warning: Could not remove TTS temp directory {TTS_TEMP_DIR}: {e}")
    exit()

# Assign a new sequential index to the remaining valid sentences and images
# This `final_clip_index` (0 to N-1) will be used for generating clips
final_clip_mapping = [] # List of dicts: {final_clip_index, original_sentence_idx, img_pair_idx, image_path, sentence_info}
final_clip_index = 0
for valid_idx, sentence_info in enumerate(valid_sentences_info):
    original_sentence_idx = sentence_info["original_index"]
    # Find the two images for this sentence
    images_for_sentence = [d for d in generated_image_details if d["original_sentence_idx"] == original_sentence_idx]
    images_for_sentence.sort(key=lambda x: x["img_pair_idx"]) # Ensure order 0, then 1

    if len(images_for_sentence) == 2:
        for img_detail in images_for_sentence:
            final_clip_mapping.append({
                "final_clip_index": final_clip_index,
                "original_sentence_idx": original_sentence_idx, # Keep for reference
                "img_pair_idx": img_detail["img_pair_idx"], # 0 or 1
                "image_path": img_detail["path"],
                "sentence_info": sentence_info # Contains text, tts_path, tts_duration
            })
            final_clip_index += 1
    else:
        print(f"WARNING: Expected 2 images for sentence {original_sentence_idx+1} but found {len(images_for_sentence)}. Skipping this sentence.")
        # Need to potentially remove the TTS file too if we skip here? Or let cleanup handle it.

num_segments = len(final_clip_mapping)

if num_segments == 0:
    print("\nERROR: No valid sentence/image pairs remaining after final mapping. Exiting.")
    # Add cleanup here too
    if os.path.exists(TTS_TEMP_DIR):
        try: shutil.rmtree(TTS_TEMP_DIR)
        except OSError as e: print(f"  Warning: Could not remove TTS temp directory {TTS_TEMP_DIR}: {e}")
    exit()

print(f"\nSuccessfully prepared {num_segments} image/sentence segments for video generation.")


# =====================================================
# Part 1: Create Clips with DYNAMIC Duration, Overlay, Animation, Fire, WHOOSH & TTS
# =====================================================
print("\n--- Part 1: Generating Video Clips with Audio ---")
final_clip_paths = []
temp_files_to_clean = [] # Keep track of silent clips and split audio parts

# Loop through the mapped segments
for clip_info in final_clip_mapping:
    final_clip_idx = clip_info["final_clip_index"] # 0, 1, 2...
    segment_num = final_clip_idx + 1 # 1, 2, 3...
    img_pair_idx = clip_info["img_pair_idx"] # 0 or 1
    overlay_img_path = clip_info["image_path"]
    sentence_info = clip_info["sentence_info"]
    sentence_text = sentence_info["text"]
    sentence_tts_path = sentence_info["tts_path"]
    sentence_tts_duration = sentence_info["tts_duration"]
    original_sentence_idx = clip_info["original_sentence_idx"] # For reference

    # *** Calculate DYNAMIC Clip Duration and Frames ***
    # Ensure duration is positive before dividing
    if sentence_tts_duration <= 0:
        print(f"WARNING: Sentence {original_sentence_idx+1} has non-positive TTS duration ({sentence_tts_duration:.3f}s). Setting clip duration to minimum.")
        current_clip_duration = animation_duration_seconds + 0.1 # Use minimum threshold
    else:
        current_clip_duration = sentence_tts_duration / 2.0

    # Apply minimum duration threshold, considering animation
    min_duration_threshold = animation_duration_seconds + 0.1 # Need *some* time after animation
    if current_clip_duration < min_duration_threshold:
        # print(f"Info: Calculated clip duration ({current_clip_duration:.3f}s) for segment {segment_num} is short. Clamping to {min_duration_threshold:.3f}s.")
        current_clip_duration = min_duration_threshold

    total_frames = max(1, int(current_clip_duration * fps)) # Ensure at least 1 frame

    # *** Calculate Background Video Index (Looping based on final_clip_idx) ***
    bg_video_index = (final_clip_idx % num_background_videos) + 1
    bg_video_path = os.path.join(input_folder, f"{bg_video_index}.mp4")

    # Define intermediate and final clip paths for this segment
    silent_clip_path = os.path.join(TTS_TEMP_DIR, f"clip_{segment_num}_silent.mp4") # Put intermediate in temp
    final_clip_path = os.path.join(input_folder, f"clip_{segment_num}_final.mp4") # Final clips in main folder for concat

    print(f"\nProcessing Segment {segment_num}/{num_segments} (Orig Sent Idx {original_sentence_idx + 1}, Clip Pair {img_pair_idx + 1})")
    print(f" -> Duration: {current_clip_duration:.3f}s, Frames: {total_frames}")
    print(f" -> Using Img: {os.path.basename(overlay_img_path)}, BG: {bg_video_index}.mp4")
    print(f" -> TTS: {os.path.basename(sentence_tts_path)} (Part {img_pair_idx + 1})")

    # --- Input File Checks ---
    if not os.path.exists(bg_video_path):
        print(f"ERROR: Background video not found: {bg_video_path}. Skipping segment {segment_num}.")
        continue
    if not os.path.exists(overlay_img_path):
         print(f"ERROR: Overlay image not found: {overlay_img_path}. Skipping segment {segment_num}.")
         continue
    if not os.path.exists(sentence_tts_path):
         print(f"ERROR: Sentence TTS audio not found: {sentence_tts_path}. Skipping segment {segment_num}.")
         continue

    # --- 1. Generate Silent Video Clip ---
    print(f"  Step 1: Generating silent video: {os.path.basename(silent_clip_path)}")
    silent_clip_ok = False
    try:
        # Open Video Captures
        cap = cv2.VideoCapture(bg_video_path)
        if not cap.isOpened(): raise IOError(f"Cannot open background video: {bg_video_path}")
        bg_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fire_cap = cv2.VideoCapture(fire_overlay_video_path)
        if not fire_cap.isOpened(): raise IOError(f"Cannot open fire overlay video: {fire_overlay_video_path}")
        fire_frame_count = int(fire_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fire_frame_count <= 0: raise IOError(f"Fire overlay video {fire_overlay_video_path} has no frames.")

        # Setup Video Writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # H.264 might need 'avc1' or 'h264' depending on codec availability
        out = cv2.VideoWriter(silent_clip_path, fourcc, fps, (output_width, output_height))
        if not out.isOpened(): raise IOError(f"Failed to open VideoWriter for {silent_clip_path}")

        # Load and Prepare Overlay Image
        overlay_orig = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
        if overlay_orig is None: raise IOError(f"Cannot load overlay image: {overlay_img_path}")

        if len(overlay_orig.shape) == 2: overlay_orig = cv2.cvtColor(overlay_orig, cv2.COLOR_GRAY2BGRA)
        elif overlay_orig.shape[2] == 3: overlay_orig = cv2.cvtColor(overlay_orig, cv2.COLOR_BGR2BGRA)
        max_resize_width = output_width - 2 * margin
        max_resize_height = output_height - 2 * margin
        orig_h, orig_w = overlay_orig.shape[:2]
        if orig_w == 0 or orig_h == 0: raise ValueError(f"Invalid overlay image dimensions for {overlay_img_path}")
        scale_factor = min(max_resize_width / orig_w, max_resize_height / orig_h)
        new_w = max(1, int(orig_w * scale_factor))
        new_h = max(1, int(orig_h * scale_factor))
        overlay_resized = cv2.resize(overlay_orig, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4 if scale_factor < 1.0 else cv2.INTER_LINEAR)

        # Animation Setup (Cycles based on final_clip_idx)
        target_x = margin + (max_resize_width - new_w) // 2
        target_y = margin + (max_resize_height - new_h) // 2
        animation_type = final_clip_idx % 4 # Cycle animation based on overall segment index
        start_x, start_y = target_x, target_y
        use_animation = animation_duration_frames > 0 and total_frames > animation_duration_frames

        if use_animation:
            anim_map = {0: "Top", 1: "Left", 2: "Bottom", 3: "Right"}
            # print(f"    Animating image slide from {anim_map[animation_type]} ({animation_duration_seconds}s)")
            if animation_type == 0: start_y = -new_h
            elif animation_type == 1: start_x = -new_w
            elif animation_type == 2: start_y = output_height
            elif animation_type == 3: start_x = output_width
        # elif animation_duration_frames > 0: print(f"    Note: Clip duration too short for animation.")

        # Frame Processing Loop
        processed_frames = 0
        last_good_bg_frame = None
        fire_frame_index = -1
        write_error_occurred = False

        while processed_frames < total_frames:
            # Read Background Frame (with looping and fallback)
            ret_bg, bg_frame = cap.read()
            current_bg_frame_for_compositing = None
            if not ret_bg:
                if bg_frame_count > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret_bg, bg_frame = cap.read()
                    if not ret_bg:
                       if last_good_bg_frame is None: last_good_bg_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                       current_bg_frame_for_compositing = last_good_bg_frame.copy()
                else:
                    if last_good_bg_frame is None: last_good_bg_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    current_bg_frame_for_compositing = last_good_bg_frame.copy()
                    if processed_frames == 0: print(f"    ERROR: BG video {bg_video_path} unreadable. Using black.")

            if ret_bg and current_bg_frame_for_compositing is None:
                 try:
                     bg_frame_resized = cv2.resize(bg_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                     last_good_bg_frame = bg_frame_resized
                     current_bg_frame_for_compositing = bg_frame_resized
                 except Exception as e:
                     if last_good_bg_frame is None:
                          print(f"    ERROR: Failed resize first BG frame: {e}. Using black fallback.")
                          last_good_bg_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                     current_bg_frame_for_compositing = last_good_bg_frame.copy()

            # Read Fire Frame (Looping)
            fire_frame_index = (fire_frame_index + 1) % fire_frame_count
            if fire_frame_index == 0: fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_fire, fire_frame = fire_cap.read()
            fire_frame_resized = None
            if not ret_fire:
                fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret_fire, fire_frame = fire_cap.read()
                if not ret_fire: fire_frame_resized = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                else: fire_frame_index = 0
            if fire_frame_resized is None and ret_fire:
                 try: fire_frame_resized = cv2.resize(fire_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                 except Exception: fire_frame_resized = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            elif fire_frame_resized is None: fire_frame_resized = np.zeros((output_height, output_width, 3), dtype=np.uint8)


            # Compositing
            composite = current_bg_frame_for_compositing.copy()

            # Calculate Image Overlay Position
            current_x, current_y = target_x, target_y
            if use_animation and processed_frames < animation_duration_frames:
                progress = min(1.0, (processed_frames + 1) / animation_duration_frames)
                current_x = int(start_x + progress * (target_x - start_x))
                current_y = int(start_y + progress * (target_y - start_y))

            # Place Image Overlay (Safe Placement with Alpha Blending)
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
                    except Exception as e: pass # Ignore minor blending errors

            # Add Fire Overlay (Additive Blending)
            try:
                 if composite.shape[2] == 4: composite = cv2.cvtColor(composite, cv2.COLOR_BGRA2BGR)
                 if fire_frame_resized is not None and fire_frame_resized.shape[2] == 4: fire_frame_resized = cv2.cvtColor(fire_frame_resized, cv2.COLOR_BGRA2BGR)
                 if fire_frame_resized is not None and composite.shape == fire_frame_resized.shape:
                     composite = cv2.add(composite, fire_frame_resized)
            except Exception as e: pass # Ignore minor fire overlay errors

            # Write Frame
            try:
                out.write(composite)
            except Exception as e:
                 print(f"\n    FATAL ERROR: Failed to write frame {processed_frames} for segment {segment_num}: {e}")
                 write_error_occurred = True
                 break # Exit loop

            processed_frames += 1
            print(f"  Segment {segment_num}: Generating frame {processed_frames}/{total_frames}...", end='\r')

        # End Frame Processing Loop
        print(f"  Segment {segment_num}: Generated {min(processed_frames, total_frames)} frames. {' ' * 20}")
        cap.release()
        fire_cap.release()
        out.release()

        if write_error_occurred:
             raise Exception("Frame write error occurred during silent clip generation.")
        elif processed_frames < total_frames:
             raise Exception(f"Silent clip generation incomplete ({processed_frames}/{total_frames} frames).")
        else:
            silent_clip_ok = True
            temp_files_to_clean.append(silent_clip_path) # Mark for cleanup later

    except Exception as e:
        print(f"  ERROR during silent clip generation for segment {segment_num}: {e}")
        # Ensure resources are released even if error happened mid-way
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'fire_cap' in locals() and fire_cap.isOpened(): fire_cap.release()
        if 'out' in locals() and out.isOpened(): out.release()
        if os.path.exists(silent_clip_path): # Clean up partial silent clip
             try: os.remove(silent_clip_path)
             except OSError: pass
        continue # Skip to next segment

    # --- 2. Split TTS Audio ---
    if silent_clip_ok:
        print(f"  Step 2: Splitting TTS audio: {os.path.basename(sentence_tts_path)}")
        tts_part1_path = os.path.join(TTS_TEMP_DIR, f"tts_{original_sentence_idx + 1}_part1.mp3")
        tts_part2_path = os.path.join(TTS_TEMP_DIR, f"tts_{original_sentence_idx + 1}_part2.mp3")
        split_time = sentence_tts_duration / 2.0
        split_success = False

        # Only split if not already done for the other pair of this sentence
        # Check if the *other* part exists if this is the second clip (img_pair_idx == 1)
        # Or if *neither* part exists if this is the first clip (img_pair_idx == 0)
        needs_split = False
        if img_pair_idx == 0 and not (os.path.exists(tts_part1_path) or os.path.exists(tts_part2_path)):
            needs_split = True
        elif img_pair_idx == 1 and not os.path.exists(tts_part1_path): # If part 1 isn't there, something went wrong, try split again
            needs_split = True
        # No need to split if img_pair_idx is 1 and part1 *does* exist (assume part2 also exists or will be created)

        if needs_split and sentence_tts_duration > 0.01: # Avoid splitting tiny files
            try:
                # Command for first half
                cmd1 = [
                    ffmpeg_executable, "-y", "-i", sentence_tts_path,
                    "-ss", "0", "-to", f"{split_time:.4f}", # Use more precision
                    "-c", "copy", tts_part1_path
                ]
                success1, _ = run_ffmpeg_command(cmd1, "TTS Split Part 1")

                # Command for second half
                cmd2 = [
                    ffmpeg_executable, "-y", "-i", sentence_tts_path,
                    "-ss", f"{split_time:.4f}", # Start precisely at split time
                    # No -to needed, goes to end by default
                    "-c", "copy", tts_part2_path
                ]
                success2, _ = run_ffmpeg_command(cmd2, "TTS Split Part 2")

                if success1 and success2 and os.path.exists(tts_part1_path) and os.path.exists(tts_part2_path):
                    split_success = True
                    temp_files_to_clean.extend([tts_part1_path, tts_part2_path]) # Mark for cleanup
                    print("    TTS split successful.")
                else:
                     print("    ERROR: TTS split command failed or output files missing.")
                     if os.path.exists(tts_part1_path): os.remove(tts_part1_path)
                     if os.path.exists(tts_part2_path): os.remove(tts_part2_path)

            except Exception as e:
                print(f"    ERROR during TTS split: {e}")
                if os.path.exists(tts_part1_path): os.remove(tts_part1_path)
                if os.path.exists(tts_part2_path): os.remove(tts_part2_path)
        elif not needs_split:
             print("    TTS likely already split for this sentence.")
             # Verify the required part actually exists
             required_part = tts_part1_path if img_pair_idx == 0 else tts_part2_path
             if os.path.exists(required_part):
                 split_success = True
             else:
                 print(f"    ERROR: Required TTS part {os.path.basename(required_part)} not found!")
                 split_success = False # Force failure if expected part missing
        elif sentence_tts_duration <= 0.01:
             print("    WARNING: TTS duration too short to split accurately. Skipping split.")
             # Treat as failure? Or attempt to use the whole short TTS? Let's treat as failure for now.
             split_success = False


        # Determine which TTS part to use for this segment
        current_tts_part_path = tts_part1_path if img_pair_idx == 0 else tts_part2_path

        if not split_success:
             print(f"  Skipping audio merge for segment {segment_num} due to TTS split failure.")
             if os.path.exists(silent_clip_path): # Clean up unused silent clip
                  try: os.remove(silent_clip_path)
                  except OSError: pass
             continue # Skip to next segment

    # --- 3. Merge Video, Whoosh, and TTS Part ---
    if silent_clip_ok and split_success:
        print(f"  Step 3: Merging video, Whoosh, and TTS part -> {os.path.basename(final_clip_path)}")

        # Ensure the specific TTS part we need exists
        if not os.path.exists(current_tts_part_path):
             print(f"    ERROR: Cannot find required TTS part: {current_tts_part_path}. Skipping merge.")
             if os.path.exists(silent_clip_path): # Clean up unused silent clip
                  try: os.remove(silent_clip_path)
                  except OSError: pass
             continue # Skip to next segment

        merge_ok = False
        try:
            # FFmpeg command to mix Whoosh and TTS part, map video, limit duration
            merge_cmd = [
                ffmpeg_executable, "-y",
                "-i", silent_clip_path,        # Input 0: Silent Video
                "-i", sound_file,              # Input 1: Whoosh sound
                "-i", current_tts_part_path,   # Input 2: Correct TTS part
                "-filter_complex",
                # Delay both audio streams slightly? Maybe not needed if Whoosh is short
                # "[1:a]adelay=0s:all=1[a_whoosh];[2:a]adelay=50s:all=1[a_tts];[a_whoosh][a_tts]amix=inputs=2:duration=first:dropout_transition=0[a_mix]", # Mix them
                # Simpler mix, let them overlay from the start
                "[1:a][2:a]amix=inputs=2:duration=longest:dropout_transition=1[a_mix]", # Mix Whoosh and TTS
                "-map", "0:v:0",               # Map video from input 0
                "-map", "[a_mix]",             # Map the mixed audio
                "-c:v", "copy",                # Copy video codec
                "-c:a", "aac", "-b:a", "128k", # Encode audio to AAC
                "-shortest",                   # Ensure output duration matches shortest input (the video)
                final_clip_path
            ]
            success, _ = run_ffmpeg_command(merge_cmd, "Video/Audio Merge")

            if success and os.path.exists(final_clip_path) and os.path.getsize(final_clip_path) > 0:
                print(f"    Segment {segment_num} with audio saved: {os.path.basename(final_clip_path)}")
                final_clip_paths.append(final_clip_path)
                merge_ok = True
            else:
                 print(f"    ERROR: Final merge failed for segment {segment_num}.")
                 if os.path.exists(final_clip_path): os.remove(final_clip_path) # Clean up failed merge

        except Exception as e:
             print(f"    ERROR during final merge: {e}")
             if os.path.exists(final_clip_path): os.remove(final_clip_path)

        if not merge_ok:
            # Clean up silent clip if merge failed
            if os.path.exists(silent_clip_path):
                 try: os.remove(silent_clip_path)
                 except OSError: pass
            # Note: Split TTS parts are handled by the main cleanup list


# =====================================================
# Part 2: (Skipped) - No longer needed
# =====================================================
print("\n--- Part 2: Specific Sound Effect Addition (Integrated into Part 1) ---")


# =====================================================
# Part 3: Combine Final Clips (with audio) Using FFmpeg Concat
# =====================================================
print("\n--- Part 3: Combining Final Video Clips ---")
final_output_path = os.path.join(input_folder, FINAL_OUTPUT_FILENAME)
concat_list_file = os.path.join(TTS_TEMP_DIR, "ffmpeg_concat_list.txt")
concat_ok = False

if not final_clip_paths:
    print("No final video clips were successfully created in Part 1. Skipping final combination.")
else:
    # Sort final clips numerically based on filename "clip_X_final.mp4"
    try:
        final_clip_paths.sort(key=lambda x: int(re.search(r'clip_(\d+)_final\.mp4', os.path.basename(x)).group(1)))
    except Exception as e:
        print(f"Warning: Could not sort final clip paths based on number: {e}. Using existing order.")

    print(" Preparing FFmpeg concat list...")
    try:
        with open(concat_list_file, "w", encoding='utf-8') as f:
            for clip_path in final_clip_paths:
                # Need to escape special characters for ffmpeg concat list, single quotes are common
                safe_path = clip_path.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
        temp_files_to_clean.append(concat_list_file) # Add list file to cleanup
        print(f" Concat list created: {concat_list_file}")

        # Run FFmpeg concat
        concat_cmd = [
            ffmpeg_executable, "-y",
            "-f", "concat",
            "-safe", "0", # Allows relative/absolute paths if needed, use with caution if paths are untrusted
            "-i", concat_list_file,
            "-c", "copy", # Copy existing streams (video and audio) without re-encoding
            final_output_path
        ]
        success, _ = run_ffmpeg_command(concat_cmd, "Final Video Concatenation")

        if success and os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
             print(f"\n Successfully combined {len(final_clip_paths)} clips.")
             print(f" Final video saved as: {final_output_path}")
             concat_ok = True
        else:
             print("\n ERROR: Final video concatenation failed.")
             if os.path.exists(final_output_path): os.remove(final_output_path) # Clean failed output

    except Exception as e:
        print(f" An error occurred during final video combination: {e}")


# =====================================================
# Part 4: (Skipped) - Audio processing now in Part 1 & 3
# =====================================================
print("\n--- Part 4: Audio Concatenation & Merge (Integrated into Part 1 & 3) ---")


# =====================================================
# Cleanup
# =====================================================
print("\n--- Cleanup ---")

if concat_ok and os.path.exists(final_output_path):
    print(" Final video created successfully. Cleaning up intermediate files...")
    # Files to remove: generated images, final clips used for concat, other temp files (silent clips, split TTS, concat list)
    cleanup_files = generated_image_paths + final_clip_paths + temp_files_to_clean

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

    # Remove the temporary TTS directory (contains original TTS and potentially failed intermediates)
    if os.path.exists(TTS_TEMP_DIR):
        print(f" Removing temporary TTS directory: {TTS_TEMP_DIR}")
        try:
            shutil.rmtree(TTS_TEMP_DIR)
        except OSError as e:
            print(f"  Warning: Could not remove temporary TTS directory {TTS_TEMP_DIR}: {e}")
            failed_removals.append(TTS_TEMP_DIR)

    print(f" Cleanup complete. Attempted removal of {len(cleanup_files)} files/dirs.")
    if failed_removals: print(f"  Could not remove the following items: {failed_removals}")

else:
    print(" Final video generation failed or did not complete. Skipping cleanup.")
    print(" Intermediate files may remain in:")
    print(f" - {input_folder}")
    print(f" - {TTS_TEMP_DIR}")


print("\n--- Script finished ---")
