import src.tools 
import cv2
import numpy as np
import sys
import random

stack3 = None

RUGD_CLASSES = [
    {"id": 0, "R": 0, "G": 0, "B": 0},         # void_c
    {"id": 1, "R": 108, "G": 64, "B": 20},     # dirt
    {"id": 2, "R": 255, "G": 229, "B": 204},   # sand
    {"id": 3, "R": 0, "G": 102, "B": 0},       # grass
    {"id": 4, "R": 0, "G": 255, "B": 0},       # tree
    {"id": 5, "R": 0, "G": 153, "B": 153},     # pole
    {"id": 6, "R": 0, "G": 128, "B": 255},     # water
    {"id": 7, "R": 0, "G": 0, "B": 255},       # sky
    {"id": 8, "R": 255, "G": 255, "B": 0},     # vehicle
    {"id": 9, "R": 255, "G": 0, "B": 127},     # container
    {"id": 10, "R": 64, "G": 64, "B": 64},     # asphalt
    {"id": 11, "R": 255, "G": 128, "B": 0},    # gravel
    {"id": 12, "R": 255, "G": 0, "B": 0},      # building
    {"id": 13, "R": 153, "G": 76, "B": 0},     # mulch
    {"id": 14, "R": 102, "G": 102, "B": 0},    # rock_bed
    {"id": 15, "R": 102, "G": 0, "B": 0},      # log_c
    {"id": 16, "R": 0, "G": 255, "B": 128},    # bicycle
    {"id": 17, "R": 204, "G": 153, "B": 255},  # person
    {"id": 18, "R": 102, "G": 0, "B": 204},    # fence
    {"id": 19, "R": 255, "G": 153, "B": 204},  # bush
    {"id": 20, "R": 0, "G": 102, "B": 102},    # sign
    {"id": 21, "R": 153, "G": 204, "B": 255},  # rock
    {"id": 22, "R": 102, "G": 255, "B": 255},  # bridge
    {"id": 23, "R": 101, "G": 101, "B": 11},   # concrete
    {"id": 24, "R": 114, "G": 85, "B": 47},    # picnic_table
    {"id": 25, "R": 255, "G": 255, "B": 255},   # time 
    {"id": 26, "R": 200, "G": 200, "B": 200}   # accuracy
]

class_names = {
    0: "void_c", 1: "dirt", 2: "sand", 3: "grass", 4: "tree", 
    5: "pole", 6: "water", 7: "sky", 8: "vehicle", 9: "container", 
    10: "asphalt", 11: "gravel", 12: "building", 13: "mulch", 14: "rock_bed", 
    15: "log_c", 16: "bicycle", 17: "person", 18: "fence", 19: "bush", 
    20: "sign", 21: "rock", 22: "bridge", 23: "concrete", 24: "pic_tab",
    25: "Time:", 26: "Accuracy:"
}

def mouse_callback(event, x, y, flags, param):
    if stack3 is not None:
        h, w, _ = stack3.shape
        if 0 <= y < h and 0 <= x < w:
            b, g, r = stack3[y, x]
            if event == cv2.EVENT_MOUSEMOVE:
                for cls in RUGD_CLASSES:
                    if cls["R"] == r and cls["G"] == g and cls["B"] == b:
                        print(f"Mouse at ({x}, {y}) - Color: (R: {r}, G: {g}, B: {b}) - Class ID: {cls['id']}")
                        break

def main():

    if len(sys.argv) < 2:
        print("Usage: python vis.py <id>")
        id = random.randint(0, len(src.tools.RUGDDataset.__len__()) - 1)
        sys.exit(1)
    else:
        id = int(sys.argv[1])
    output, img, annot, time  = src.tools.make_set(id)
    cv2_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2_annot = cv2.cvtColor(annot, cv2.COLOR_RGB2BGR)

    acc = src.tools.evaluate_images(cv2_annot, cv2_output)

    break_col = np.zeros((cv2_img.shape[0], 10, 3), dtype=np.uint8)

    stack1 = np.hstack((cv2_img,break_col, cv2_annot, break_col, cv2_output))

    img_width =  stack1.shape[1]

    row_height = 30
    items_per_row = 20
    col_width = img_width // items_per_row

    legend_rows = []
    current_row = np.zeros((row_height, img_width, 3), dtype=np.uint8) + 245 # počiatočný prázdny riadok

    for i, c in enumerate(RUGD_CLASSES):
        idx_in_row = i % items_per_row
        start_x = idx_in_row * col_width
        bgr_color = (c["B"], c["G"], c["R"])
        name = f"{c['id']}: {class_names.get(c['id'], 'unknown')}"
        
        if idx_in_row == 0 and i > 0:
            legend_rows.append(current_row)
            current_row = np.zeros((row_height, img_width, 3), dtype=np.uint8) + 245
            
        box_x = start_x + 10
        box_y_top = 5
        box_y_bottom = row_height - 5

        if c["id"] == 25:
            name = name + f" {time*1000:.0f} ms"
            box_y_top = 10
            box_y_bottom = row_height - 10
            cv2.putText(current_row, name, (box_x + 1, row_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
        elif c["id"] == 26:
            name = name + f" {acc:.2f}%"
            box_y_top = 10
            box_y_bottom = row_height - 10
            cv2.putText(current_row, name, (box_x + 1, row_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
        else:  
            cv2.rectangle(current_row, (box_x, box_y_top), (box_x + 20, box_y_bottom), bgr_color, -1)
            cv2.rectangle(current_row, (box_x, box_y_top), (box_x + 20, box_y_bottom), (100, 100, 100), 1) # okraj
        
            cv2.putText(current_row, name, (box_x + 28, row_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)

    # Pridanie posledného riadku do zoznamu
    legend_rows.append(current_row)

    # 3. Spojenie malých riadkov legendy do jedného celkového panelu pomocou np.vstack
    legend_panel = np.vstack(legend_rows)

    stack2 = np.vstack((legend_panel, stack1))

    window_name = "Image, Annotation, Prediction"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
            cv2.imshow(window_name, stack2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
if __name__ == "__main__":
    main()