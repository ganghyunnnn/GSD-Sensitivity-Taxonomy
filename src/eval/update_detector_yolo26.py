"""Re-run ONLY the detector with YOLO26-OBB (newer SOTA, +3.4 mAP on DOTA-OBB vs
YOLO11) on the same DOTA-val cells, updating the 'det' field in dota_val_counting.json
while keeping the three VLM columns. Pre-reg conf=0.25, imgsz=1536 unchanged."""
import io, json, sys
from pathlib import Path
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass
ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = Path("E:/FP-Tag/DOTAv1/images/val")
DATA = ROOT / "experiments/dota_val_counting.json"
DOTA = {0:"plane",1:"ship",2:"storage tank",3:"baseball diamond",4:"tennis court",
        5:"basketball court",6:"ground track field",7:"harbor",8:"bridge",9:"large vehicle",
        10:"small vehicle",11:"helicopter",12:"roundabout",13:"soccer ball field",14:"swimming pool"}
PRIMARY=["plane","ship","small vehicle","large vehicle"]

def main():
    from ultralytics import YOLO
    model=YOLO("yolo26l-obb.pt")
    n2i={v:k for k,v in DOTA.items()}
    d=json.load(open(DATA,encoding="utf-8"))
    rows=d["rows"]
    by_img={}
    for r in rows: by_img.setdefault(r["image"],[]).append(r)
    images=sorted(by_img)
    for k,stem in enumerate(images):
        ip=IMG_DIR/f"{stem}.jpg"
        res=model.predict(str(ip),imgsz=1536,conf=0.25,verbose=False)[0]
        ids=res.obb.cls.tolist() if res.obb is not None else []
        cnt={c:0 for c in PRIMARY}
        for i in ids:
            nm=DOTA.get(int(i))
            if nm in cnt: cnt[nm]+=1
        for r in by_img[stem]:
            r["det"]=cnt[r["cls"]]
        if k%20==0:
            json.dump(d,open(DATA,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
            print(f"  {k+1}/{len(images)} images",flush=True)
    d.setdefault("config",{})["detector"]="yolo26l-obb.pt"
    json.dump(d,open(DATA,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
    print(f"DONE detector->YOLO26 updated {len(rows)} cells")

if __name__=="__main__":
    main()
