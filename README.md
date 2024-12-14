Super Marker Tracker

Super Marker Tracker is a Python-based implementation for supervised marker tracking in video data, inspired by Facemap. It processes videos to accurately detect and track markers recorded from a subject.

Features

Marker Detection: Identify specific markers in video frames.

Supervised Tracking: Leverage annotations to improve tracking precision.

Facemap Integration: Utilize core concepts from the Facemap framework.

Customizable Pipeline: Flexible setup for various video datasets.

Installation

Clone the repository:

git clone https://github.com/mohamad-hoseini/super-marker-tracker.git
cd super-marker-tracker

Install dependencies:

pip install -r requirements.txt

Usage

Prepare Input Videos

Place your videos in the input_videos/ folder.

Run Marker Tracking

python track_markers.py --video_path input_videos/sample_video.mp4

Output

Tracked markers will be saved in the output_data/ folder.

Folder Structure

.
|-- input_videos/       # Folder for input video files
|-- output_data/        # Output folder for tracking results
|-- pose_estimation/    # Code for marker pose estimation
|-- source/             # Core source files
|-- track_markers.py    # Main script for marker tracking
|-- README.md           # Project documentation

