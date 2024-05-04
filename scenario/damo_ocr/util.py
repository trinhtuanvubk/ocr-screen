import sys
import numpy as np
import cv2

def general_text_reading_visualization(predictions, image, color = (49, 125, 237), thickness = 6):

    # draw quadrangles
    output_image = image.copy()
    for item in predictions:
        quadrangle = item['position']
        pts = np.array([[quadrangle[0], quadrangle[1]], [quadrangle[2], quadrangle[3]],
                        [quadrangle[4], quadrangle[5]], [quadrangle[6], quadrangle[7]]],
                        np.int32)
    
        pts = pts.reshape((-1, 1, 2))

        # draw poly
        isClosed = True
        output_image = cv2.polylines(output_image, [pts],  isClosed, color, thickness)

    return output_image