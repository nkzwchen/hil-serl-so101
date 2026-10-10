#!/usr/bin/env python3
"""Select a fixed cam_main crop; subscribes to images only, never controls arms."""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / 'hil-serl' if (HERE / 'hil-serl').is_dir() else HERE.parent
sys.path.insert(0, str(REPO / 'examples/experiments/so101'))
from camera_observation import DEFAULT_ROI_PATH, MAIN_ROI_PATH, decode_rgb, validate_roi, crop_roi


def main():
    import tkinter as tk
    from PIL import Image, ImageTk
    import rclpy
    from sensor_msgs.msg import Image as ROSImage
    from rclpy.qos import qos_profile_sensor_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', choices=['cam_main', 'cam_main_roi'], default='cam_main_roi',
                        help='Which output view to crop (both use the original cam_main frame)')
    parser.add_argument('--out', help='Override the selected view configuration path')
    args = parser.parse_args()
    args.out = args.out or str(MAIN_ROI_PATH if args.view == "cam_main" else DEFAULT_ROI_PATH)
    rclpy.init()
    node = rclpy.create_node('select_cam_main_roi')
    root = tk.Tk()
    root.title(f'{args.view} ROI — drag a rectangle, preview, Save')
    state = {'rgb': None, 'photo': None, 'start': None, 'box': None,
             'frozen': False, 'arrival': 0.0, 'preview': None}
    canvas = tk.Canvas(root, width=640, height=480)
    canvas.pack()
    status = tk.StringVar(value='Waiting for /follower/cam_main/image_raw')
    tk.Label(root, textvariable=status).pack()
    preview = tk.Label(root)
    preview.pack()

    def receive(msg):
        if state['frozen']:
            return
        try:
            state['rgb'] = decode_rgb(msg)
            state['arrival'] = time.monotonic()
        except ValueError as exc:
            status.set(str(exc))

    node.create_subscription(ROSImage, '/follower/cam_main/image_raw', receive,
                             qos_profile_sensor_data)

    def spec():
        rgb = state['rgb']
        x0, y0 = state['start']
        x1, y1 = state['end']
        h, w = rgb.shape[:2]
        return validate_roi({'version': 1, 'camera': 'cam_main',
                             'source_topic': '/follower/cam_main/image_raw',
                             'source_size': [w, h],
                             'xyxy': [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                             'preprocess': 'letterbox_rgb_v1'})

    def press(event):
        if state['rgb'] is None or (not state['frozen'] and
                                    time.monotonic() - state['arrival'] > 0.5):
            status.set('No fresh image; check the camera')
            return
        state['frozen'] = True
        h, w = state['rgb'].shape[:2]
        state['start'] = (max(0, min(w, event.x)), max(0, min(h, event.y)))
        state.pop('end', None)
        if state['box']:
            canvas.delete(state['box'])
        state['box'] = canvas.create_rectangle(*state['start'], *state['start'],
                                                outline='lime', width=2)

    def drag(event):
        if state['start'] is None:
            return
        h, w = state['rgb'].shape[:2]
        state['end'] = (max(0, min(w, event.x)), max(0, min(h, event.y)))
        canvas.coords(state['box'], *state['start'], *state['end'])
        try:
            selected = spec()
            state['preview'] = ImageTk.PhotoImage(Image.fromarray(crop_roi(state['rgb'], selected, 224)))
            preview.configure(image=state['preview'])
            status.set(f"ROI {selected['xyxy']} | source {w}x{h} | preview 224x224")
        except ValueError:
            status.set('Select a non-empty rectangle')

    def save():
        try:
            selected = spec()
        except (KeyError, TypeError, ValueError) as exc:
            status.set(f'Select a valid rectangle first: {exc}')
            return
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix('.tmp')
        temporary.write_text(json.dumps(selected, indent=2) + '\n')
        temporary.replace(out)
        print(f'Saved {out}; copy this exact file to the learner before training.')
        root.destroy()

    def resume():
        state.update(frozen=False, start=None)
        state.pop('end', None)
        if state['box']:
            canvas.delete(state['box'])
        state['box'] = None

    canvas.bind('<ButtonPress-1>', press)
    canvas.bind('<B1-Motion>', drag)
    canvas.bind('<ButtonRelease-1>', drag)
    tk.Button(root, text='Live / reselect', command=resume).pack(side='left')
    tk.Button(root, text='Save ROI', command=save).pack(side='right')

    def tick():
        rclpy.spin_once(node, timeout_sec=0)
        if state['rgb'] is not None and not state['frozen']:
            h, w = state['rgb'].shape[:2]
            canvas.configure(width=w, height=h)
            state['photo'] = ImageTk.PhotoImage(Image.fromarray(state['rgb']))
            canvas.delete('frame')
            canvas.create_image(0, 0, image=state['photo'], anchor='nw', tags='frame')
            canvas.tag_lower('frame')
        root.after(30, tick)
    root.after(30, tick)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
