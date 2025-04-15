import cv2
import numpy as np
import os
import subprocess

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
sound_file = os.path.join(input_folder, "Whoos 3.mp3")

# =====================================================
# Part 1: Create 4-Second Silent Clips with Overlay
# =====================================================
# For each segment:
#  - Read a 4-second portion from the background video.
#  - Resize the background to 1280x720.
#  - Load the corresponding overlay image and scale it so that it fits inside 
#    the region defined by (output_width - 2*margin, output_height - 2*margin),
#    preserving its aspect ratio.
#  - Center the overlay within that region (so that the gap is equal on all sides).
#  - Write the composite frames to a silent video file ("clip_i_silent.mp4").
silent_clip_paths = []

for i in range(1, num_segments + 1):
    bg_video_path = os.path.join(input_folder, f"{i}.mp4")
    overlay_img_path = os.path.join(input_folder, f"{i}.png")
    silent_clip = os.path.join(input_folder, f"clip_{i}_silent.mp4")
    
    cap = cv2.VideoCapture(bg_video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open background video: {bg_video_path}")
        continue

    # Create VideoWriter for silent clip (4 seconds exactly)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(silent_clip, fourcc, fps, (output_width, output_height))
    
    # Load overlay image (in color)
    overlay = cv2.imread(overlay_img_path, cv2.IMREAD_COLOR)
    if overlay is None:
        print(f"ERROR: Cannot load overlay image: {overlay_img_path}")
        cap.release()
        continue

    # Determine maximum available size for the overlay:
    max_width = output_width - 2 * margin
    max_height = output_height - 2 * margin

    # Get original overlay dimensions
    orig_h, orig_w, _ = overlay.shape
    # Compute scaling factor to fit inside (max_width, max_height) while preserving aspect ratio.
    scale_factor = min(max_width / orig_w, max_height / orig_h)
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    overlay_resized = cv2.resize(overlay, (new_w, new_h))

    # Compute placement so that the overlay is centered in the available region.
    # The available region is centered within the full frame with a margin.
    x_offset = margin + (max_width - new_w) // 2
    y_offset = margin + (max_height - new_h) // 2

    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the background frame to 1280x720.
        frame = cv2.resize(frame, (output_width, output_height))
        # Composite: copy background and overlay the image
        composite = frame.copy()
        # Place the overlay: ensure indices do not exceed frame boundaries.
        composite[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = overlay_resized
        out.write(composite)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Silent clip for segment {i} saved as: {silent_clip}")
    silent_clip_paths.append(silent_clip)

# =====================================================
# Part 2: Add a Single Sound Effect at the Beginning Using FFmpeg
# =====================================================
# For each silent clip, we add the sound effect at the beginning.
# The sound effect is trimmed to 1 second, and then 3 seconds of silence are appended
# so that the audio track is exactly 4 seconds long.
# The FFmpeg filter_complex below does this by:
#   - Trimming the input sound to 1 second.
#   - Generating 3 seconds of silence (using anullsrc).
#   - Concatenating the two segments.
final_clip_paths = []

for idx, silent_clip in enumerate(silent_clip_paths, start=1):
    final_clip = os.path.join(input_folder, f"final_{idx}.mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", silent_clip,
        "-i", sound_file,
        "-filter_complex",
        # Explanation of filter:
        # [1]atrim=duration=1,asetpts=PTS-STARTPTS -> takes the first 1s of the sound effect.
        # anullsrc=r=44100:cl=stereo,atrim=duration=3 -> generates 3s of silence.
        # Then concatenate the two audio segments: n=2 means 2 segments.
        "[1]atrim=duration=1,asetpts=PTS-STARTPTS[sound]; anullsrc=r=44100:cl=stereo,atrim=duration=3[silence]; [sound][silence]concat=n=2:v=0:a=1[a]",
        "-map", "0:v:0",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        final_clip
    ]
    print(f"Adding sound effect to clip {idx} (sound plays for 1s at start)...")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Final clip with audio saved as: {final_clip}")
    final_clip_paths.append(final_clip)

# =====================================================
# Part 3: Combine Final Clips (Video Only) Using OpenCV
# =====================================================
# Here we combine the video streams (ignoring audio) from all final clips into one silent video.
combined_video_silent = os.path.join(input_folder, "combined_video_silent.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_combined = cv2.VideoWriter(combined_video_silent, fourcc, fps, (output_width, output_height))

for clip in final_clip_paths:
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        print(f"ERROR: Cannot open final clip: {clip}")
        continue
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_combined.write(frame)
    cap.release()
out_combined.release()
print(f"Combined silent video saved as: {combined_video_silent}")

# =====================================================
# Part 4: Extract Audio from Final Clips, Concatenate, and Merge with Video
# =====================================================
# Extract the audio stream from each final clip.
audio_files = []
for idx, clip in enumerate(final_clip_paths, start=1):
    audio_file = os.path.join(input_folder, f"audio_{idx}.aac")
    extract_cmd = [
        "ffmpeg", "-y",
        "-i", clip,
        "-vn",
        "-acodec", "copy",
        audio_file
    ]
    subprocess.run(extract_cmd, check=True)
    audio_files.append(audio_file)

# Create a text file list for audio concatenation.
audio_list_file = os.path.join(input_folder, "audio_list.txt")
with open(audio_list_file, "w") as f:
    for af in audio_files:
        f.write(f"file '{af}'\n")

combined_audio = os.path.join(input_folder, "combined_audio.aac")
concat_audio_cmd = [
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", audio_list_file,
    "-c", "copy",
    combined_audio
]
subprocess.run(concat_audio_cmd, check=True)
print(f"Combined audio saved as: {combined_audio}")

# Finally, merge the combined silent video with the concatenated audio.
final_output = os.path.join(input_folder, "final_output.mp4")
merge_cmd = [
    "ffmpeg", "-y",
    "-i", combined_video_silent,
    "-i", combined_audio,
    "-c:v", "copy",
    "-c:a", "aac",
    "-shortest",
    final_output
]
subprocess.run(merge_cmd, check=True)
print(f"\nFinal video (16 seconds total) saved as: {final_output}")
