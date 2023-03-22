import cv2 as cv
import numpy as np
import time
import os

dim_delimiter = 2
dim_square_x = 65
dim_square_y = 65
offset = -2

def is_red_square(line, column):
    red_square_list = [(1, 'A'), (8,'A'), (15,'A'), (1,'H'), (15,'H'), (1,'O'), (8,'O'), (15,'O')]
    return (line, column) in red_square_list

def is_pink_square(line, column):
    pink_square_list1 = [(2,'B'), (3,'C'),(4,'D'), (5,'E'), (8,'H'), (11,'K'), (12, 'L'), (13,'M'), (14,'N')]
    pink_square_list2 = [(14,'B'), (13,'C'), (12,'D'),(11,'E'), (5,'K'), (4, 'L'), (3,'M'), (2,'N')]
    pink_square_list = pink_square_list1 + pink_square_list2
    return (line, column) in pink_square_list

def is_dark_blue_square(line, column):
    blue_square_list1 = [(1,'D'), (1,'L'), (3,'G'),(3,'I'),(4,'A'), (4, 'H'), (4,'O'),(7,'C'),(7,'G'),(7,'I'),(7,'M')]
    blue_square_list2 = [(8,'D'), (8,'L'), (9,'C'),(9,'G'),(9,'I'),(9,'M'),(12,'A'), (12,'H'),(12,'O'),(13,'G'),(13,'I'),(15,'D'),(15,'L')]
    blue_square_list = blue_square_list1 + blue_square_list2
    return (line, column) in blue_square_list

def is_blue_square(line, column):
    dark_blue_square_list = [(2,'F'), (2,'J'), (6,'B'),(6,'F'),(6,'J'),(6,'N'),(10,'B'),(10,'F'),(10,'J'),(10,'N'),(14,'F'),(14,'J')]
    return (line, column) in dark_blue_square_list

def get_letter_score(letter):
    values = {'A':1, 'B':9,'C':1,'D':2,'E':1,'F':8,'G':9,'H':10,'I':1,'J':10,'L':1,'M':4,'N':1,'O':1,'P':2,'R':1,'S':1,'T':1,'U':1,'V':8,'X':10,'Z':10,'?':0}
    return values[letter]

def get_letter_premium_score(line, column):
    if is_blue_square(line,column) == True:
        return 3
    if is_dark_blue_square(line, column) == True:
        return 2
    return 1

def show_image(img, name):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 

def find_occ_positions(bin_image, imgray):
    lines = range(1, 16)
    columns = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
    positions = []
    for line in lines:
        for column in columns:
            square = get_square(line,column,bin_image)
            sum = square.sum() / 255

            if sum > 1000:
                set_square_0(line,column,imgray,square)
                positions.append((line,column))
    return positions


def get_square(line, column, image):
    poz_line_begin = (line-1) * dim_square_x + (line) * dim_delimiter + offset
    poz_column_begin = (ord(column) - ord("A")) * dim_square_y + (ord(column) - ord("A")) * dim_delimiter - offset

    if poz_column_begin < 0:
        poz_column_begin = 0
    if poz_line_begin < 0:
        poz_line_begin = 0

    square = image.copy()[poz_line_begin:poz_line_begin + dim_square_x, poz_column_begin:poz_column_begin + dim_square_y]
    return square

def set_square_0(line, column, image, square):
    zeros = np.zeros((square.shape[0], square.shape[1]), np.uint8)
    poz_line_begin = (line-1) * dim_square_x + (line) * dim_delimiter + offset
    poz_column_begin = (ord(column) - ord("A")) * dim_square_y + (ord(column) - ord("A")) * dim_delimiter - offset

    if poz_column_begin < 0:
        poz_column_begin = 0
    if poz_line_begin < 0:
        poz_line_begin = 0

    image[poz_line_begin:poz_line_begin + square.shape[0], poz_column_begin:poz_column_begin + square.shape[1]] = zeros

def find_corners(contours):
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    if len(contours) > 3:
        for i in range(len(contours)):
            point = contours[i].squeeze()

            if top_left is None or point[0] + point[1] < top_left[0] + top_left[1]:
                top_left = point

            if bottom_right is None or point[0] + point[1] > bottom_right[0] + bottom_right[1]:
                bottom_right = point

        diff = np.diff(contours).squeeze()
        
        top_right = contours[np.argmin(diff)].squeeze()
        bottom_left = contours[np.argmax(diff)].squeeze()
    
    return top_left, top_right, bottom_left, bottom_right

def get_perspective_transform(contours, img, debug = 0):


        top_left, top_right, bottom_left, bottom_right = find_corners(contours)

        if debug == 1:
            img_copy = img.copy()
            cv.circle(img_copy,tuple(top_left),20,(0,0,255),-1)
            cv.circle(img_copy,tuple(top_right),20,(0,0,255),-1)
            cv.circle(img_copy,tuple(bottom_left),20,(0,0,255),-1)
            cv.circle(img_copy,tuple(bottom_right),20,(0,0,255),-1)
            show_image(img_copy, "Contours")


        width = 1000
        height = 1000

        table_coordinates = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
        destination_of_table_coordinates = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

        M = cv.getPerspectiveTransform(table_coordinates, destination_of_table_coordinates)

        result = cv.warpPerspective(img, M, (width, height))
        return result

def read_image(name):
    return cv.imread(name)

def sharpen_image(img):
    image_m_blur = cv.medianBlur(img,3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
    return image_sharpened

def dilate_image(img, iterations):
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv.dilate(img, kernel, iterations=2)
    return img_dilation

def threshold_image(img):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret,thresh1 = cv.threshold(imgray,80,255,cv.THRESH_BINARY)
    return thresh1

def find_contours_table(img):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    arii_contur = []
    for cnt in contours:
        arii_contur.append(cv.contourArea(cnt))

    media = sum(arii_contur) / len(arii_contur)

    contur_maxim = max(arii_contur)
    contur_maxmax = None

    for cnt in contours:
        if cv.contourArea(cnt) == contur_maxim:
            contur_maxmax = cnt

    return contur_maxmax

def augument_image(image):
    image_thresh = image[1]
    (h, w) = image_thresh.shape[:2]
    (cx, cy) = (w//2, h//2)

    images = []

    #rotations
    for angle in np.arange(-5, 5.5,0.5):
        rotation_matrix = cv.getRotationMatrix2D((cx,cy), angle, 1.0)
        rotated_image = cv.warpAffine(image_thresh, rotation_matrix, (w,h))
        images.append((0, rotated_image))

    return images

def load_templates():
    templates = dict()

    path_templates = "letter_templates\\"
    dir_list = os.listdir(path_templates)
    
    for file in dir_list:
        template_image = read_image(path_templates+file)
        template_image = cv.cvtColor(template_image, cv.COLOR_BGR2GRAY)
        letter = file[0]
        if letter == '1':
            letter = '?'
        thresh_template_image = cv.threshold(template_image,70,255,cv.THRESH_BINARY_INV)
        all_images_for_template = [thresh_template_image]
        all_images_for_template.extend(augument_image(thresh_template_image)) #add rotated thresholds
        if not(letter in templates.keys()):
            templates[letter] = all_images_for_template
        else:
            templates[letter].append(thresh_template_image)
    return templates

def get_individual_square_value(square_image, templates):
    max_value = -500
    letter = None
    for template_name in templates.keys():
        for i in range(len(templates[template_name])):
            template_image = templates[template_name][i][1]
            corr = cv.matchTemplate(square_image, template_image,  cv.TM_CCOEFF_NORMED)
            corr=np.max(corr)
            if corr > max_value:
                max_value = corr
                letter = template_name

    if max_value > 0.6:
        return letter
    else:
        return None

def get_squares_values(image_thresholded, templates, positions):
    positions_with_values = []
    for position_row, position_column in positions:
        square_image = get_square(position_row, position_column, image_thresholded)
        letter_for_position = get_individual_square_value(square_image, templates)
        if not(letter_for_position is None):
            positions_with_values.append((position_row, position_column, letter_for_position))

    return positions_with_values

def init_dictionary_positions_for_game():
    lines = range(1, 16)
    columns = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
    positions_row = dict()
    for line in lines:
        for column in columns:
            if column == 'A':
                positions_row[line] = dict()
            positions_row[line][column] = False

    return positions_row

def make_output_list(letters_positions_current_game, game_occ_found_positions):
    list_for_output = []
    for position in letters_positions_current_game:
        if game_occ_found_positions[position[0]][position[1]] == False:
            game_occ_found_positions[position[0]][position[1]] = True
            list_for_output.append((position[0],position[1], position[2]))
    return list_for_output

def write_to_file(list_positions_for_output, image_name, output_folder_name, score):
    file_name = image_name.split('.')[0] + ".txt"
    file = open(output_folder_name + "/" + file_name,'w')

    for i in range(len(list_positions_for_output)):
        position = f"{list_positions_for_output[i][0]}{list_positions_for_output[i][1]}"
        letter = list_positions_for_output[i][2]
        file.write(f"{position} {letter}\n")

    file.write(f"{score}")
    file.close()


def get_letter_from_position(line, column, letter_list):
    for pos in letter_list:
        if pos[0] == line and pos[1] == column:
            return pos[2]
    return None

def make_new_positions_visited(letters_list):
    new_postions = []
    for line, column, _ in letters_list:
        new_postions.append([line,column,False,False]) #First false means vertically, second means horizontally
    return new_postions

def go_to_beginning_vertically(begin_line, begin_column, letter_list):
    current_line = begin_line

    while current_line != 0:
        letter = get_letter_from_position(current_line, begin_column, letter_list)

        if letter == None:
            return current_line+1, begin_column
        
        current_line -= 1
    
    return 1, begin_column

def go_to_beginning_horizontally(begin_line, begin_column, letter_list):
    current_column = ord(begin_column)
    steps = 0
    while chr(current_column) != '@':
        letter = get_letter_from_position(begin_line, chr(current_column), letter_list)

        if letter == None:
            return begin_line, chr(current_column+1)

        steps+=1
        current_column -= 1
    
    return begin_line, 'A'

def mark_as_visited_new_letter(line, column, vertical, visited_positions):
    #new_postions.append([line,column,False,False])
    for i in range(len(visited_positions)):
        x = visited_positions[i][0]
        y = visited_positions[i][1]
        if x == line and column == y:
            if vertical == True:
                visited_positions[i][2] = True
            else:
                visited_positions[i][3] = True

def calculate_score_horizontal(begin_line, begin_column, letter_list, visited_position):
    score = 0
    number_of_premium_word = 0
    nr_pink = 0
    nr_red = 0
    current_column = ord(begin_column)
    while  chr(current_column) != '@':
        letter = get_letter_from_position(begin_line, chr(current_column), letter_list)
        if letter == None:
            break
        mark_as_visited_new_letter(begin_line, chr(current_column), False, visited_position)
        score_of_letter_default = get_letter_score(letter)

        was_before = get_letter_from_position(begin_line, chr(current_column), visited_position)

        if was_before != None:
            score += get_letter_premium_score(begin_line, chr(current_column)) * score_of_letter_default
        else:
            score += get_letter_score(letter)

        if is_pink_square(begin_line, chr(current_column)) == True and was_before != None:
            number_of_premium_word += 1
            nr_pink += 1
        if is_red_square(begin_line, chr(current_column)) == True and was_before != None:
            number_of_premium_word += 1
            nr_red += 1
        current_column += 1
    
    if nr_pink > 0:
        score = score * nr_pink * 2
    if nr_red > 0:
        score = score * nr_red * 3
    if number_of_premium_word >= 2:
        score = score * 2
    return score

def calculate_score_vertical(begin_line, begin_column, letter_list, visited_position):
    score = 0
    number_of_premium_word = 0
    nr_pink = 0
    nr_red = 0
    current_line = begin_line
    while current_line != 16:
        letter = get_letter_from_position(current_line, begin_column, letter_list)
        if letter == None:
            break
        mark_as_visited_new_letter(current_line, begin_column, True, visited_position)
        score_of_letter_default = get_letter_score(letter)
        was_before = get_letter_from_position(current_line, begin_column, visited_position)
        
        if was_before != None:
            score += get_letter_premium_score(current_line, begin_column) * score_of_letter_default
        else:
            score += get_letter_score(letter)
        
        if is_pink_square(current_line, begin_column) == True and was_before != None:
            number_of_premium_word += 1
            nr_pink += 1
        if is_red_square(current_line, begin_column) == True and was_before != None:
            number_of_premium_word += 1
            nr_red += 1
        current_line += 1
    
    if nr_pink > 0:
        score = score * nr_pink * 2
    if nr_red > 0:
        score = score * nr_red * 3
    if number_of_premium_word >= 2:
        score = score * 2
    return score

def calculate_score(all_letters, new_letters):
    visited_position = make_new_positions_visited(new_letters)
    score = 0
    for x,y,vertical,horizontal in visited_position:
        if vertical == False:
            line, column = go_to_beginning_vertically(x,y,all_letters)
            if get_letter_from_position(line+1, column,all_letters) != None:
                score += calculate_score_vertical(line,column,all_letters,visited_position)
        if horizontal == False:
            line, column = go_to_beginning_horizontally(x,y,all_letters)
            if get_letter_from_position(line, chr(ord(column)+1),all_letters) != None:
                score += calculate_score_horizontal(line,column,all_letters,visited_position)
        if len(new_letters) >= 7:
            score+=50
    return score

def game_play():
    templates = load_templates()

    output_folder_name = "output_algo"
    input_folder = "teste"


    if os.path.exists(output_folder_name) == False:
        os.mkdir(output_folder_name)

    photos_list = os.listdir(input_folder)
    current_game = None
    game_occ_positions = None
    #photos_list = ["5_17.jpg", "5_18.jpg"]
    for photo_name in photos_list:

        if current_game is None:
            current_game = photo_name[0]
            game_occ_positions = init_dictionary_positions_for_game()
        else:
            if photo_name[0] != current_game:
                current_game = photo_name[0]
                game_occ_positions = init_dictionary_positions_for_game()

        image = read_image(input_folder + "/" + photo_name)
        img = sharpen_image(image)
        img = dilate_image(img, 2)
        img = threshold_image(img)
        contours = find_contours_table(img)
        result = get_perspective_transform(contours=contours, img=image,debug=0)

        ####gasire_pozitii_piese
        imgray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        imgray_sharpen = sharpen_image(imgray)
        ret,thresh1 = cv.threshold(imgray_sharpen,70,255,cv.THRESH_BINARY)
        imgray_2 = np.copy(imgray)
        #cv.imwrite("pozdeanalizat.jpg", thresh1)
        letter_positions = find_occ_positions(thresh1, imgray_2)
        
        ####gasire_denumire_piese
        ret,thresh_for_templates = cv.threshold(imgray,70,255,cv.THRESH_BINARY_INV)
        pozitii_litere = get_squares_values(thresh_for_templates, templates, letter_positions)
        list_positions_for_output = make_output_list(pozitii_litere, game_occ_positions)
        score = calculate_score(pozitii_litere, list_positions_for_output)
        write_to_file(list_positions_for_output, photo_name, output_folder_name, score)


game_play()

