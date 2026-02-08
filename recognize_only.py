#!/usr/bin/env python3
"""
Recognition Only - 30 second test
==================================
Uses existing database. Just recognize.
"""

import cv2
import numpy as np
import time
import sys

sys.path.insert(0, '/home/intruder/vision-assistant')

from src.v2.storage.schema import VisionDB

THRESHOLD = 0.25


def main():
    print("=" * 60)
    print("  RECOGNITION TEST (30 seconds)")
    print("=" * 60)

    # Try existing databases
    for db_path in ["klement_test.db", "vision_test.db", "vision.db", "faces.db"]:
        try:
            db = VisionDB(db_path=db_path, match_threshold=THRESHOLD)
            if db.person_count > 0:
                print(f"\n✅ Found database: {db_path}")
                print(f"   Persons: {db.person_count}")
                for p in db.list_persons():
                    emb_count = db.get_embedding_count(p.id)
                    print(f"   - {p.name} ({p.group_type}): {emb_count} embeddings")
                break
            db.close()
        except:
            continue
    else:
        print("❌ No database with registered faces found")
        return

    print("\nLoading face detector...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_thresh=0.4, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    print("\nPress SPACE to start 30-second recognition test")
    print("Move around, change lighting, test distance\n")

    # Wait for space
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cv2.putText(frame, "Press SPACE to start", (w//2-120, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Recognition Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == ord(' '): break

    # Recognition loop
    start = time.time()
    duration = 30

    results = {
        'recognized': [],
        'not_recognized': [],
        'no_face': 0
    }

    print("🔍 Running recognition...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        elapsed = time.time() - start
        remaining = duration - elapsed
        if remaining <= 0: break

        faces = app.get(frame)

        if faces:
            face = faces[0]
            bbox = face.bbox.astype(int)
            embedding = face.normed_embedding
            conf = float(face.det_score)
            face_size = bbox[3] - bbox[1]

            person, sim, _ = db.identify(embedding)

            if person:
                results['recognized'].append({
                    'name': person.name,
                    'similarity': sim,
                    'confidence': conf,
                    'face_size': face_size,
                    'time': elapsed
                })
                color = (0, 255, 0)
                status = f"{person.name} ({sim:.2f})"
            else:
                results['not_recognized'].append({
                    'similarity': sim,
                    'confidence': conf,
                    'face_size': face_size,
                    'time': elapsed
                })
                color = (0, 255, 255)
                status = f"Unknown ({sim:.2f})"

            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(display, status, (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            results['no_face'] += 1

        # Live stats
        total = len(results['recognized']) + len(results['not_recognized'])
        rate = len(results['recognized']) / total * 100 if total > 0 else 0

        cv2.putText(display, f"Time: {int(remaining)}s", (w-120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, f"Recognition: {rate:.0f}% ({len(results['recognized'])}/{total})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Recognition Test", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    total = len(results['recognized']) + len(results['not_recognized'])
    rate = len(results['recognized']) / total * 100 if total > 0 else 0

    print(f"\n📊 OVERALL:")
    print(f"   Total frames with face: {total}")
    print(f"   Recognized:             {len(results['recognized'])} ({rate:.1f}%)")
    print(f"   Not recognized:         {len(results['not_recognized'])} ({100-rate:.1f}%)")
    print(f"   No face detected:       {results['no_face']}")

    if results['recognized']:
        sims = [r['similarity'] for r in results['recognized']]
        confs = [r['confidence'] for r in results['recognized']]
        print(f"\n✅ RECOGNIZED FRAMES:")
        print(f"   Similarity: min={min(sims):.3f}, max={max(sims):.3f}, mean={np.mean(sims):.3f}")
        print(f"   Confidence: min={min(confs):.3f}, max={max(confs):.3f}, mean={np.mean(confs):.3f}")

    if results['not_recognized']:
        sims = [r['similarity'] for r in results['not_recognized']]
        confs = [r['confidence'] for r in results['not_recognized']]
        sizes = [r['face_size'] for r in results['not_recognized']]
        print(f"\n❌ NOT RECOGNIZED FRAMES:")
        print(f"   Similarity: min={min(sims):.3f}, max={max(sims):.3f}, mean={np.mean(sims):.3f}")
        print(f"   Confidence: min={min(confs):.3f}, max={max(confs):.3f}")
        print(f"   Face size:  min={min(sizes)}px, max={max(sizes)}px")

        # Why failed?
        below_threshold = len([r for r in results['not_recognized'] if r['similarity'] < THRESHOLD])
        print(f"\n   Failure analysis:")
        print(f"   - Below threshold ({THRESHOLD}): {below_threshold}")

    # Time analysis
    if results['recognized'] and results['not_recognized']:
        first_half_rec = len([r for r in results['recognized'] if r['time'] < 15])
        first_half_not = len([r for r in results['not_recognized'] if r['time'] < 15])
        second_half_rec = len([r for r in results['recognized'] if r['time'] >= 15])
        second_half_not = len([r for r in results['not_recognized'] if r['time'] >= 15])

        first_rate = first_half_rec / (first_half_rec + first_half_not) * 100 if (first_half_rec + first_half_not) > 0 else 0
        second_rate = second_half_rec / (second_half_rec + second_half_not) * 100 if (second_half_rec + second_half_not) > 0 else 0

        print(f"\n⏱️  TIME ANALYSIS:")
        print(f"   First 15s:  {first_rate:.1f}% recognized")
        print(f"   Last 15s:   {second_rate:.1f}% recognized")

        if second_rate < first_rate - 20:
            print(f"   ⚠️  Significant drop - lighting change impact")

    print("\n" + "=" * 60)
    db.close()
    print("Done!")


if __name__ == "__main__":
    main()
