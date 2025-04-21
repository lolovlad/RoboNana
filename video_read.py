'''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import cv2

cap = None
current_position = 0
playing = False
fps = 30
video_path = None


def open_file():
    global cap, current_position, playing, fps, video_path
    file_path = filedialog.askopenfilename(filetypes=[('Video files', ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv'))])
    if file_path:
        print(f'Выбранный файл: {file_path}')
        video_path = file_path
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            cap = None
            return
        current_position = 0
        fps_value = int(cap.get(cv2.CAP_PROP_FPS))
        if fps_value > 0:
            fps = fps_value
        else:
            fps = 30

        playing = True
        update_frame()
        #play_button['state'] = 'normal'
        #rewind_button['state'] = 'normal'
        #forward_button['state'] = 'normal'


def update_frame():
    global cap, current_position, playing, video_path
    if cap is not None and playing:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (720, 480))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            label.config(image=image)
            label.image = image

            current_position += 1
            label.after(int(1000 / fps), update_frame)
        else:
            stop_video()


def stop_video():
    global cap, playing
    playing = False
    if cap is not None:
        cap.release()
        cap = None
    #play_button['state'] = 'normal'
    #rewind_button['state'] = 'normal'
    #forward_button['state'] = 'normal'


def start_video():
    global cap, playing, video_path, current_position

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            cap = None
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
        playing = True
        update_frame()
        play_button['state'] = 'normal'
        rewind_button['state'] = 'normal'
        forward_button['state'] = 'normal'


def rewind_video():
    global current_position, cap
    if cap is not None:
        current_position = max(0, current_position - fps * 10)
        update_frame()
def forward_video():
    global current_position, cap
    if cap is not None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_position = min(total_frames - 1, current_position + fps * 10)
        update_frame()

def toggle_playback():
    global cap, playing, video_path, current_position
    if playing:
        stop_video()
        play_button.config(text='Воспроизвести')
    else:
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video file.")
                cap = None
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
            playing = True
            update_frame()
            play_button.config(text='Остановить')

root = tk.Tk()
root.title('Камера')

label = tk.Label(root)
label.pack()

open_button = ttk.Button(root, text='Открыть файл', command=open_file)
open_button.pack()

frame_buttons = ttk.Frame(root)
frame_buttons.pack()


play_button = ttk.Button(frame_buttons, text='Остановить', command=toggle_playback)
play_button.pack(side='left', padx=5)

rewind_button = ttk.Button(frame_buttons, text='Перемотать << 10 сек', command=rewind_video, state='disabled')
rewind_button.pack(side='left', padx=5)

forward_button = ttk.Button(frame_buttons, text='Перемотать >> 10 сек', command=forward_video, state='disabled')
forward_button.pack(side='left', padx=5)


root.mainloop()
'''