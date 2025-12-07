def iou(boxA, boxB):
    # box: (x,y,w,h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom

def centroid_from_bbox(bbox):
    x, y, w, h = bbox
    return (int(x + w/2), int(y + h/2))