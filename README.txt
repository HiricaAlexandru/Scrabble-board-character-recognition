The project documentation is in "Tema 1_cava_Hirica_Ioan_Alexandru.pdf"

1. the libraries required to run the project including the full version of each library


numpy==1.23.5
opencv_python==4.6.0.66

2. how to run each task and where to look for the output file.

The application will create the output files in the folder specified in the variable "output_folder_name" inside the function "game_play".
The default value is in a folder named output_algo.

In order to run:
    - The variable "input_folder" inside the function "game_play" should have the value of the path to the images to be recognized.
    - The variable "path_templates" from the function "load_templates" should hold the path to the templates of the letter. The default value for it is
    "letter_templates".
