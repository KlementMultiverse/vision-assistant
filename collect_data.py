#!/usr/bin/env python3
"""
Data Collection
================
Collects 30 seconds of face data for analysis.
Move around, change lighting - we log everything.

Press SPACE to start, 'q' to quit.
"""

import cv2
import numpy as np
import time
import sys
import json
from datetime import datetime

sys.path.insert(0, '/home/intruder/vision-assistant')


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "test_subject"

    print("=" * 60)
    print(f"  DATA COLLECTION: 30 seconds")
    print("=" * 60)

    print("\nLoading face detector...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_thresh=0.4, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    print("\nPress SPACE to start 30-second collection")
    print("During collection:")
    print("  - Move closer/farther (1m to 2m)")
    print("  - Turn head slightly")
    print("  - Switch lights on/off")
    print("Press 'q' to quit\n")

    # Wait for start
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cv2.putText(frame, "Press SPACE to start", (w//2-150, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord(' '):
            break

    # Collection
    start_time = time.time()
    duration = 30
    data = []
    embeddings = []

    print("🎬 COLLECTING...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        elapsed = time.time() - start_time
        remaining = duration - elapsed

        if remaining <= 0:
            break

        # Detect
        faces = app.get(frame)

        frame_data = {
            'time': round(elapsed, 2),
            'face_detected': False,
            'confidence': 0,
            'face_size': 0,
            'bbox': None
        }

        if faces:
            face = faces[0]
            bbox = face.bbox.astype(int).tolist()
            conf = float(face.det_score)
            face_size = bbox[3] - bbox[1]

            frame_data['face_detected'] = True
            frame_data['confidence'] = round(conf, 3)
            frame_data['face_size'] = face_size
            frame_data['bbox'] = bbox

            embeddings.append({
                'time': elapsed,
                'embedding': face.normed_embedding,
                'confidence': conf,
                'face_size': face_size
            })

            # Draw
            color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(display, f"conf:{conf:.2f} size:{face_size}px",
                       (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        data.append(frame_data)

        # Timer and count
        cv2.putText(display, f"Time: {int(remaining)}s | Faces: {len(embeddings)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Data Collection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("  DATA ANALYSIS")
    print("=" * 60)

    total = len(data)
    detected = len([d for d in data if d['face_detected']])

    print(f"\n📊 FRAME STATS:")
    print(f"   Total frames:    {total}")
    print(f"   Face detected:   {detected} ({detected/total*100:.1f}%)")
    print(f"   No face:         {total - detected}")

    if embeddings:
        confs = [e['confidence'] for e in embeddings]
        sizes = [e['face_size'] for e in embeddings]

        print(f"\n📊 DETECTION QUALITY:")
        print(f"   Confidence: min={min(confs):.3f}, max={max(confs):.3f}, mean={np.mean(confs):.3f}")
        print(f"   Face size:  min={min(sizes)}px, max={max(sizes)}px, mean={np.mean(sizes):.0f}px")

        # Quality breakdown
        high_conf = len([e for e in embeddings if e['confidence'] >= 0.6])
        med_conf = len([e for e in embeddings if 0.5 <= e['confidence'] < 0.6])
        low_conf = len([e for e in embeddings if e['confidence'] < 0.5])

        print(f"\n   Quality breakdown:")
        print(f"   - High (>=0.6):  {high_conf} ({high_conf/len(embeddings)*100:.1f}%)")
        print(f"   - Medium (0.5-0.6): {med_conf} ({med_conf/len(embeddings)*100:.1f}%)")
        print(f"   - Low (<0.5):    {low_conf} ({low_conf/len(embeddings)*100:.1f}%)")

        # Compute similarities between embeddings
        print(f"\n📊 EMBEDDING SIMILARITY (same person consistency):")

        emb_matrix = np.array([e['embedding'] for e in embeddings])
        similarities = np.dot(emb_matrix, emb_matrix.T)
        mask = ~np.eye(len(embeddings), dtype=bool)
        pairwise = similarities[mask]

        print(f"   All pairs:     min={np.min(pairwise):.3f}, max={np.max(pairwise):.3f}, mean={np.mean(pairwise):.3f}")

        # Quality pairs only
        quality_idx = [i for i, e in enumerate(embeddings) if e['confidence'] >= 0.5]
        if len(quality_idx) > 1:
            quality_emb = emb_matrix[quality_idx]
            q_sims = np.dot(quality_emb, quality_emb.T)
            q_mask = ~np.eye(len(quality_idx), dtype=bool)
            q_pairwise = q_sims[q_mask]
            print(f"   Quality only:  min={np.min(q_pairwise):.3f}, max={np.max(q_pairwise):.3f}, mean={np.mean(q_pairwise):.3f}")

        # Time-based analysis
        print(f"\n⏱️  TIME ANALYSIS:")
        first_half = [e for e in embeddings if e['time'] < 15]
        second_half = [e for e in embeddings if e['time'] >= 15]

        if first_half:
            print(f"   First 15s: {len(first_half)} faces, avg conf={np.mean([e['confidence'] for e in first_half]):.3f}")
        if second_half:
            print(f"   Last 15s:  {len(second_half)} faces, avg conf={np.mean([e['confidence'] for e in second_half]):.3f}")

        # Cross-half similarity (tests lighting change impact)
        if first_half and second_half:
            first_embs = np.array([e['embedding'] for e in first_half[:5]])  # First 5
            second_embs = np.array([e['embedding'] for e in second_half[:5]])  # First 5 of second half
            cross_sims = np.dot(first_embs, second_embs.T).flatten()
            print(f"   Cross-half similarity: min={np.min(cross_sims):.3f}, max={np.max(cross_sims):.3f}, mean={np.mean(cross_sims):.3f}")

            if np.mean(cross_sims) < 0.4:
                print(f"   ⚠️  Low cross-half similarity - lighting change detected!")

        # Recommendation
        print(f"\n💡 RECOMMENDATIONS:")
        if np.mean(confs) < 0.5:
            print(f"   - Detection confidence low - improve lighting or move closer")

        quality_mean = np.mean(q_pairwise) if len(quality_idx) > 1 else 0
        if quality_mean > 0:
            suggested_threshold = max(0.25, np.min(q_pairwise) - 0.05)
            print(f"   - Suggested threshold: {suggested_threshold:.3f}")
            print(f"   - Store {min(10, len([e for e in embeddings if e['confidence'] >= 0.6]))} best embeddings")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
