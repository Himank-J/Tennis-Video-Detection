import cv2

def read_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret,frame = cap.read()
        if not(ret):
            break
        frames.append(frame)
    
    cap.release()
    return frames

def save_video(output_video_frames,output_video_path):
    '''
    This line initializes the FourCC (Four Character Code) which is a 4-byte code 
    used to specify the video codec. In this case, 'MJPG' stands for Motion-JPEG, 
    a codec for video compression.
    '''
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    '''
    This line creates a VideoWriter object which will be used to write frames to a 
    video file.
        - output_video_path: the path where the video will be saved.
        - fourcc: the codec used for the video.
        - 24: the frames per second (fps) for the video.
        - (output_video_frames[0].shape[1], output_video_frames[0].shape[0]): 
        the width and height of the video frames. 
        This is taken from the shape of the first frame in the output_video_frames list.
    '''
    out = cv2.VideoWriter(output_video_path,fourcc,24,(output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def annotate_frame_number(frames):

    for i,frame in enumerate(frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    return frames