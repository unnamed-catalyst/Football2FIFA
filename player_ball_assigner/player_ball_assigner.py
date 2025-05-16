import sys

sys.path.append("../")
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 1000

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position
            )
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position
            )
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player

    def assign_ball_to_players_with_teams(self, players, ball_bbox, frame_num):
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance = 99999
        team1_min_distance = 99999
        team2_min_distance = 99999
        closest_player = -1
        closest_team_1 = -1
        closest_team_2 = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]
            team = player["team"]

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position
            )
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position
            )
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                # Closest overall
                if distance < min_distance:
                    min_distance = distance
                    closest_player = player_id
                # Closest for team 1
                if team == 1 and distance < team1_min_distance:
                    team1_min_distance = distance
                    closest_team_1 = player_id
                # Closest for team 2
                if team == 2 and distance < team2_min_distance:
                    team2_min_distance = distance
                    closest_team_2 = player_id

            # Closest for team 1
            # if (
            #     team == 1
            #     and distance < self.max_player_ball_distance
            #     and distance < team1_min_distance
            # ):
            #     team1_min_distance = distance
            #     closest_team_1 = player_id

            # # Closest for team 2
            # if (
            #     team == 2
            #     and distance < self.max_player_ball_distance
            #     and distance < team2_min_distance
            # ):
            #     team2_min_distance = distance
            #     closest_team_2 = player_id

        return closest_player, closest_team_1, closest_team_2
