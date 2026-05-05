import argparse
import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from torchvision import transforms
from PIL import Image

# 1. الدالة دي اللي بتستقبل الأوامر من الكوماند لاين
def get_args():
    parser = argparse.ArgumentParser(description="تشغيل متتبع السيارات على فيديو معين.")
    parser.add_argument("--video", type=str, required=True, help="مسار الفيديو (مثال: video.mp4)")
    parser.add_argument("--annot", type=str, required=True, help="مسار ملف الإحداثيات (annotation.txt)")
    parser.add_argument("--output", type=str, default="submission.csv", help="مسار حفظ النتيجة")
    parser.add_argument("--model", type=str, default="weights/siamese_tracker_v2.onnx", help="مسار الموديل")
    return parser.parse_args()

# 2. دوال التجهيز (زي ما هي في كودك)
_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def prep(img_rgb, size):
    patch = cv2.resize(img_rgb, (size, size))
    return _tfm(Image.fromarray(patch)).unsqueeze(0).numpy().astype(np.float32)

def make_cosine_window(hm_h, hm_w):
    win = np.outer(np.hanning(hm_h), np.hanning(hm_w))
    return (win / win.max()).astype(np.float32)

def parse_first_bbox(ann_path):
    try:
        with open(ann_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or any(c.isalpha() for c in line):
                    continue
                sep  = ',' if ',' in line else None
                vals = line.split(sep)
                if len(vals) >= 4:
                    x,y,w,h = [max(0, int(float(v))) for v in vals[:4]]
                    if w > 0 and h > 0:
                        return x, y, w, h
    except Exception as e:
        print(f"Error reading annotation: {e}")
    return None

def predict_frame(session, tmpl_rgb, srch_rgb, fw, fh, gt_w, gt_h, cos_win, last_cx, last_cy):
    z = prep(tmpl_rgb, 127)
    x = prep(srch_rgb, 255)
    
    hm = session.run(None, {'template': z, 'search': x})[0][0, 0]
    
    # Cosine window penalty
    win = cv2.resize(cos_win, (hm.shape[1], hm.shape[0]))
    hm_penalised = hm * (1 - 0.25) + win * 0.25
    
    H, W = hm_penalised.shape
    idx  = np.argmax(hm_penalised)
    row, col = divmod(int(idx), W)
    
    cx = int(col / W * fw)
    cy = int(row / H * fh)
    conf = float(hm[row, col])
    
    if conf < 0.30 and last_cx is not None:
        cx, cy = last_cx, last_cy
        
    x_out = max(0, cx - gt_w // 2)
    y_out = max(0, cy - gt_h // 2)
    
    return x_out, y_out, gt_w, gt_h, cx, cy, conf

# 3. الكود الأساسي للتشغيل
def main():
    args = get_args()
    
    print(f"Loading ONNX model from {args.model}...")
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    cos_win = make_cosine_window(17, 17)
    
    print(f"Processing Video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")
        
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, init_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame.")
    init_rgb = cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB)
    
    bbox = parse_first_bbox(args.annot)
    if not bbox:
        print("Warning: Could not parse bbox, using default fallback.")
        iw, ih = max(20, int(fw * 0.05)), max(20, int(fh * 0.05))
        ix, iy = fw//2 - iw//2, fh//2 - ih//2
    else:
        ix, iy, iw, ih = bbox
        
    iy_ = max(0, iy); ix_ = max(0, ix)
    tmpl_crop = init_rgb[iy_:iy_+ih, ix_:ix_+iw]
    if tmpl_crop.size == 0:
        tmpl_crop = init_rgb[:max(1,ih), :max(1,iw)]
        
    tmpl_ema = tmpl_crop.astype(np.float32)
    last_cx, last_cy = ix + iw//2, iy + ih//2
    
    records = []
    frame_idx = 0
    records.append({"Frame_Index": frame_idx, "X": ix, "Y": iy, "Width": iw, "Height": ih})
    
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        srch_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        x_out, y_out, w_out, h_out, last_cx, last_cy, conf = predict_frame(
            session, tmpl_ema.astype(np.uint8), srch_rgb, fw, fh, iw, ih, cos_win, last_cx, last_cy
        )
        
        records.append({"Frame_Index": frame_idx, "X": x_out, "Y": y_out, "Width": w_out, "Height": h_out})
        
        if conf > 0.5:
            new_crop = srch_rgb[max(0,last_cy-ih//2):last_cy+ih//2, max(0,last_cx-iw//2):last_cx+iw//2]
            if new_crop.size > 0:
                new_crop_r = cv2.resize(new_crop, (tmpl_ema.shape[1], tmpl_ema.shape[0]))
                tmpl_ema = ((1 - 0.02) * tmpl_ema + 0.02 * new_crop_r.astype(np.float32))
                
    cap.release()
    
    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    print(f"✅ Tracking complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()