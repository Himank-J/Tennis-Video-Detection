from utils import (
    read_video,
    save_video,
    annotate_frame_number,
    measure_distance,
    convert_pixel_distance_to_meters,
    draw_player_stats
)
from trackers import (
    PlayerTracker,
    BallTracker
)

import constants
import pandas as pd
from copy import deepcopy
from mini_court import MiniCourt
from court_line_detectors import CourtLineDetector

import warnings
warnings.filterwarnings(action='ignore')

def main():

    # Read input video
    input_video_path = 'input_data/input_video.mp4'
    video_frames = read_video(input_video_path)
    
    # Detect players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_saved_results=True,
        result_path="trackers_saved/player_detections.pkl"
    )

    # Detect ball
    ball_tracker = BallTracker(model_path='models/best.pt')
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_saved_results=True,
        result_path="trackers_saved/ball_detections.pkl"
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect keypoints on court
    court_line_detector = CourtLineDetector("models/keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose and filter players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints,player_detections)

    # Draw player bounding box and id for player and ball
    output_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames,ball_detections)

    # Draw keypoints on court
    output_frames = court_line_detector.draw_keypoints_on_video(output_frames,court_keypoints)

    # Add mini court
    minicourt = MiniCourt(output_frames[0])
    output_frames = minicourt.draw_mini_court(output_frames)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections  = minicourt.convert_bounding_boxes_to_mini_court_coordinates(player_detections,ball_detections,court_keypoints)
    output_frames = minicourt.draw_points_on_mini_court(output_frames,player_mini_court_detections)
    output_frames = minicourt.draw_points_on_mini_court(output_frames,ball_mini_court_detections,color=(0,255,255))
    
    
    # Detect ball shots
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,
        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],ball_mini_court_detections[end_frame][1])
        
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,constants.DOUBLE_LINE_WIDTH,minicourt.get_width_of_mini_court()) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],player_mini_court_detections[end_frame][opponent_player_id])
        
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,constants.DOUBLE_LINE_WIDTH,minicourt.get_width_of_mini_court()) 
        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']

    # Draw player stats
    output_frames = draw_player_stats(output_frames,player_stats_data_df)
    
    # Annotate frame number
    output_frames = annotate_frame_number(output_frames)

    # Save video
    save_video(output_frames,"output_data/output_video.avi")

if __name__ == '__main__':
    main()