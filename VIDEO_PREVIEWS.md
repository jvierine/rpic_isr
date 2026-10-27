# Separate density video previews

The low- and high-density previews each contain 30 seconds of fullscreen
plasma motion followed by a 15-second hold on the actual viewer GUI, with
the full voltage record and both spectra visible. They use the existing
unmagnetized low/high datasets, not the new perpendicular-field pilot.

The rendering uses the same Rust/Wasm PlasmaView and compact replay chunks
as the live viewer. Motion starts at replay frame 16 and advances at five
replay frames per output second, matching quarter-speed playback. Both
species share the same clock. Display selection is capped at 1000 dots.
The closing GUI screenshots are taken at replay frame 166, matching the
end of the motion segment. They retain the viewer's Rustplotlib plots.

Each encoded preview is 1920 by 1080, 60 fps, 45 seconds, 2700 frames.
Decoded timestamp checks pass, and all 1800 motion frames are unique in
each file. Motion and closing frames were visually inspected. Numerical
frame checks do not replace the user's requested smoothness approval.

Output MP4s and timestamp validation HDF5 are under results/youtube.
Neither preview is approved for YouTube publication yet. Descriptions
include physical interpretation, model limitations, viewer and GitHub links.

To reproduce, start scripts/video_preview_server.py in the base conda
environment, open the local video-fullscreen.html page, and render. The
local endpoint saves H.264 directly without a save dialog. Mux at 60 fps
with scripts/package_fullscreen_videos.sh. Existing outputs are protected
against accidental overwrite.
