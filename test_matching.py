#!/usr/bin/env python
"""Test the improved matching algorithm"""

from utils.recognition_deepface import load_db_cache, cosine_distance
import numpy as np

# Load embeddings
db = load_db_cache('embeddings.pkl')

print('=== Testing Improved Matching Algorithm ===\n')

# Test 1: Hossein face
print('TEST 1: Hossein face image')
print('='*50)
test_emb = db['hossein'][0]

distances = {}
for label, embs in db.items():
    dists = [cosine_distance(test_emb, emb) for emb in embs]
    min_dist = min(dists)
    mean_dist = np.mean(dists)
    score = min_dist + (mean_dist - min_dist) * 0.2
    
    distances[label] = (min_dist, mean_dist, len(embs), score)
    print(f'{label}:')
    print(f'  Min distance:  {min_dist:.4f}')
    print(f'  Mean distance: {mean_dist:.4f}')
    print(f'  Num images:    {len(embs)}')
    print(f'  Score:         {score:.4f}')
    print()

best_label = min(distances.items(), key=lambda x: x[1][3])[0]
best_dist = distances[best_label][0]
print(f'Best match: {best_label} (distance: {best_dist:.4f})')
print(f'Threshold: 0.50')
print(f'Result: {"CORRECT" if best_label == "hossein" else "WRONG"}')

# Test 2: Ali face
print('\n\nTEST 2: Ali face image')
print('='*50)
test_emb2 = db['ali'][0]

distances2 = {}
for label, embs in db.items():
    dists = [cosine_distance(test_emb2, emb) for emb in embs]
    min_dist = min(dists)
    mean_dist = np.mean(dists)
    score = min_dist + (mean_dist - min_dist) * 0.2
    
    distances2[label] = (min_dist, mean_dist, len(embs), score)
    print(f'{label}:')
    print(f'  Min distance:  {min_dist:.4f}')
    print(f'  Mean distance: {mean_dist:.4f}')
    print(f'  Num images:    {len(embs)}')
    print(f'  Score:         {score:.4f}')
    print()

best_label2 = min(distances2.items(), key=lambda x: x[1][3])[0]
best_dist2 = distances2[best_label2][0]
print(f'Best match: {best_label2} (distance: {best_dist2:.4f})')
print(f'Threshold: 0.50')
print(f'Result: {"CORRECT" if best_label2 == "ali" else "WRONG"}')

# Test 3: Hossein image 5 (different angle)
print('\n\nTEST 3: Hossein image 5 (different from image 1)')
print('='*50)
test_emb3 = db['hossein'][4]  # Different image

distances3 = {}
for label, embs in db.items():
    dists = [cosine_distance(test_emb3, emb) for emb in embs]
    min_dist = min(dists)
    mean_dist = np.mean(dists)
    score = min_dist + (mean_dist - min_dist) * 0.2
    
    distances3[label] = (min_dist, mean_dist, len(embs), score)
    print(f'{label}: min={min_dist:.4f}, mean={mean_dist:.4f}, score={score:.4f}')

best_label3 = min(distances3.items(), key=lambda x: x[1][3])[0]
best_dist3 = distances3[best_label3][0]
print(f'\nBest match: {best_label3} (distance: {best_dist3:.4f})')
print(f'Result: {"CORRECT" if best_label3 == "hossein" else "WRONG"}')

print('\n\n=== SUMMARY ===')
print(f'Test 1 (Hossein): {best_label} - {"PASS" if best_label == "hossein" else "FAIL"}')
print(f'Test 2 (Ali):     {best_label2} - {"PASS" if best_label2 == "ali" else "FAIL"}')
print(f'Test 3 (Hossein): {best_label3} - {"PASS" if best_label3 == "hossein" else "FAIL"}')
