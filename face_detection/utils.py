



def crop_face(frame, box, padding=0.2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * padding)
    pad_h = int(bh * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    face = frame[y1:y2, x1:x2]
    return face



