cmake_minimum_required(VERSION 3.5)

idf_component_register(
    SRCS
        "app_main.cpp"
        "camera.cpp"
    INCLUDE_DIRS "."
    EMBED_FILES "models/player_1_actor.espdl"
)