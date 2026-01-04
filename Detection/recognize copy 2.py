import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

def recognize_faces(query_img_path, app, client, args):
    """Recognize faces in query image"""
    
    # Read query image
    query_img = cv2.imread(query_img_path)
    if query_img is None:
        print(f"Error: Could not load image {query_img_path}")
        return None, None, None
    
    # Create a copy for drawing
    result_img = query_img.copy()
    
    # Detect faces in query image
    query_faces = app.get(query_img)
    
    if len(query_faces) == 0:
        print("No faces detected in query image")
        return result_img, [], []
    
    print(f"Detected {len(query_faces)} face(s) in query image")
    
    recognitions = []
    
    # Define colors for different persons
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    # Process each detected face
    for i, face in enumerate(query_faces):
        # Get embedding
        embedding = face.embedding.tolist()
        
        # Search in Qdrant
        search_result = client.search(
            collection_name=args.collection,
            query_vector=embedding,
            limit=args.top_k,
            score_threshold=args.threshold
        )
        
        if search_result:
            # Handle tuple response
            if isinstance(search_result, tuple):
                points = search_result[0]
            else:
                points = search_result
            
            if points:
                # Get the best match
                best_match = points[0]
                
                # Access payload properly
                best_person = "Unknown"
                best_score = 0.0
                
                if hasattr(best_match, 'payload') and best_match.payload:
                    best_person = best_match.payload.get("person", "Unknown")
                if hasattr(best_match, 'score'):
                    best_score = best_match.score
                
                recognition_info = {
                    "face_index": i,
                    "bbox": face.bbox,
                    "det_score": face.det_score,
                    "best_match": best_person,
                    "best_score": best_score,
                    "all_matches": [],
                    "embedding": embedding,
                    "color": colors[i % len(colors)]
                }
                
                # Get all matches
                for j, point in enumerate(points):
                    person = "Unknown"
                    score = 0.0
                    if hasattr(point, 'payload') and point.payload:
                        person = point.payload.get("person", "Unknown")
                    if hasattr(point, 'score'):
                        score = point.score
                    recognition_info['all_matches'].append((person, score))
                
                recognitions.append(recognition_info)
                
                # Draw on image
                bbox = face.bbox.astype(int)
                color = recognition_info['color']
                
                # Draw bounding box
                cv2.rectangle(result_img, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            color, 2)
                
                # Prepare label
                label = f"{best_person} ({best_score:.2f})"
                if best_person == "Unknown":
                    label = f"Unknown ({face.det_score:.2f})"
                
                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Background rectangle for text
                cv2.rectangle(result_img,
                            (bbox[0], bbox[1] - text_height - 10),
                            (bbox[0] + text_width, bbox[1]),
                            color, -1)
                
                # Draw text
                cv2.putText(result_img, label,
                          (bbox[0], bbox[1] - 5),
                          font, font_scale, (255, 255, 255), thickness)
                
                print(f"\nFace {i+1}:")
                print(f"  Bounding box: {face.bbox}")
                print(f"  Detection score: {face.det_score:.3f}")
                print(f"  Best match: {best_person} (score: {best_score:.3f})")
                
                if recognition_info['all_matches']:
                    print(f"  Top {min(3, len(recognition_info['all_matches']))} matches:")
                    for j, (person, match_score) in enumerate(recognition_info['all_matches'][:3]):
                        print(f"    {j+1}. {person}: {match_score:.3f}")
            else:
                # No points found
                recognition_info = {
                    "face_index": i,
                    "bbox": face.bbox,
                    "det_score": face.det_score,
                    "best_match": "Unknown",
                    "best_score": 0.0,
                    "all_matches": [],
                    "embedding": embedding,
                    "color": colors[i % len(colors)]
                }
                recognitions.append(recognition_info)
                print(f"\nFace {i+1}: No matches found")
        else:
            # No search result
            recognition_info = {
                "face_index": i,
                "bbox": face.bbox,
                "det_score": face.det_score,
                "best_match": "Unknown",
                "best_score": 0.0,
                "all_matches": [],
                "embedding": embedding,
                "color": colors[i % len(colors)]
            }
            recognitions.append(recognition_info)
            print(f"\nFace {i+1}: No matches found (below threshold)")
    
    return result_img, query_faces, recognitions

# -------------------------
# Display results with cv2
# -------------------------
def display_results_cv2(result_img, recognitions, args):
    """Display results using cv2"""
    
    # Save output image
    cv2.imwrite(args.output, result_img)
    print(f"Output saved to: {args.output}")
    
    # Show image with cv2
    window_name = "Face Recognition Results"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)
    
    # Create a help text overlay
    help_text = [
        "Face Recognition Results",
        "Press:",
        "  's' - Save image",
        "  'q' or ESC - Quit",
        "  'c' - Continue to console output"
    ]
    
    # Add help text to image copy
    display_img = result_img.copy()
    y0 = 30
    line_height = 30
    
    for j, line in enumerate(help_text):
        y = y0 + j * line_height
        if j == 0:
            cv2.putText(display_img, line, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(display_img, line, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imshow(window_name, display_img)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('s'):  # Save
            timestamp = cv2.getTickCount()
            save_path = f"recognition_result_{timestamp}.jpg"
            cv2.imwrite(save_path, result_img)
            print(f"Image saved as: {save_path}")
            
        elif key == ord('c'):  # Continue
            cv2.destroyAllWindows()
            return
            
        elif key == ord('q') or key == 27:  # Quit (q or ESC)
            cv2.destroyAllWindows()
            exit(0)
# -------------------------
# Argument parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("Face Recognition with Qdrant")
    parser.add_argument("--query-image", type=str, required=True, help="Path to query image")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top matches to return")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold (higher = more strict)")
    parser.add_argument("--collection", type=str, default="n3", help="Qdrant collection name")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image path")
    parser.add_argument("--show-plot", action="store_true", help="Show interactive plot")
    return parser.parse_args()

# -------------------------
# Initialize models and clients
# -------------------------
def initialize_models(args):
    """Initialize face analysis model and Qdrant client"""
    
    # Initialize FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider"] if args.gpu >= 0 else ["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=args.gpu, det_size=(640, 640))
    
    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")
    
    return app, client

# -------------------------
# Recognize faces in image
# -------------------------
def recognize_faces(query_img_path, app, client, args):
    """Recognize faces in query image"""
    
    # Read query image
    query_img = cv2.imread(query_img_path)
    if query_img is None:
        print(f"Error: Could not load image {query_img_path}")
        return None, None, None
    
    # Convert BGR to RGB for matplotlib
    query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    # Detect faces in query image
    query_faces = app.get(query_img)
    
    if len(query_faces) == 0:
        print("No faces detected in query image")
        return query_img_rgb, [], []
    
    print(f"Detected {len(query_faces)} face(s) in query image")
    
    recognitions = []
    
    # Process each detected face
    for i, face in enumerate(query_faces):
        # Get embedding
        embedding = face.embedding.tolist()
        
        # Search in Qdrant
        # search_result = client.search(
        #     collection_name=args.collection,
        #     query_vector=embedding,
        #     limit=args.top_k,
        #     score_threshold=args.threshold
        # )
        search_result = client.query_points(
                                        collection_name="n3", 
                                        query= embedding, 
                                        limit=1
                                    )
        
        if search_result.points:
            # Get the best match
            best_match = search_result.points[0]
            print('search_result', search_result)
            recognition_info = {
    "face_index": i,
    "bbox": face.bbox,  # [x1, y1, x2, y2]
    "det_score": face.det_score,
    "best_match": best_match.payload["person"] if best_match.payload else "Unknown",
    "best_score": best_match.score,
    "all_matches": [(hit.payload["person"] if hit.payload else "Unknown", hit.score) 
                    for hit in search_result.points],
    "embedding": embedding
}
            
            recognitions.append(recognition_info)

            print(f"\nFace {i+1}:")
            print(f"  Bounding box: {face.bbox}")
            print(f"  Detection score: {face.det_score:.3f}")
            print(f"  Best match: {recognition_info['best_match']} (score: {best_match.score:.3f})")
            print(f"  Top {min(3, len(search_result.points))} matches:")
            for j, (person, score) in enumerate(recognition_info['all_matches'][:3]):
                print(f"    {j+1}. {person}: {score:.3f}")


            
        else:
            recognition_info = {
                "face_index": i,
                "bbox": face.bbox,
                "det_score": face.det_score,
                "best_match": "Unknown",
                "best_score": 0.0,
                "all_matches": [],
                "embedding": embedding
            }
            recognitions.append(recognition_info)
            print(f"\nFace {i+1}: No matches found (below threshold)")
    
    return query_img_rgb, query_faces, recognitions

# -------------------------
# Plot results
# -------------------------
def plot_results(query_img_rgb, faces, recognitions, args):
    """Plot the query image with bounding boxes and labels"""
    
    # Create a copy for drawing
    img_with_boxes = query_img_rgb.copy()
    
    # Define colors for different persons (cycling through a colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Draw bounding boxes and labels
    for rec in recognitions:
        bbox = rec["bbox"].astype(int)
        person = rec["best_match"]
        score = rec["best_score"]
        det_score = rec["det_score"]
        
        # Choose color based on person name hash
        color_idx = hash(person) % len(colors)
        color = tuple(int(255 * c) for c in colors[color_idx][:3])
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, 
                     (bbox[0], bbox[1]), 
                     (bbox[2], bbox[3]), 
                     color, 2)
        
        # Prepare label text
        if person == "Unknown":
            label = f"Unknown ({det_score:.2f})"
        else:
            label = f"{person} ({score:.2f})"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(img_with_boxes,
                     (bbox[0], bbox[1] - text_height - 10),
                     (bbox[0] + text_width, bbox[1]),
                     color, -1)
        
        # Draw text
        cv2.putText(img_with_boxes, label,
                   (bbox[0], bbox[1] - 5),
                   font, font_scale, (255, 255, 255), thickness)
    
    # Save output image
    output_img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, output_img_bgr)
    print(f"\nOutput saved to: {args.output}")
    
    # Display image
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    ax[0].imshow(query_img_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Image with recognition results
    ax[1].imshow(img_with_boxes)
    ax[1].set_title("Face Recognition Results")
    ax[1].axis('off')
    
    plt.tight_layout()
    
    if args.show_plot:
        plt.show()
    
    # Save the plot as well
    plot_path = args.output.replace(".jpg", "_plot.jpg").replace(".png", "_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    return img_with_boxes

# -------------------------
# Display detailed results
# -------------------------
def display_detailed_results(recognitions):
    """Display detailed recognition results"""
    
    print("\n" + "="*50)
    print("DETAILED RECOGNITION RESULTS")
    print("="*50)
    
    for i, rec in enumerate(recognitions):
        print(f"\nFace {i+1}:")
        print(f"  Detection confidence: {rec['det_score']:.3f}")
        print(f"  Best match: {rec['best_match']}")
        print(f"  Similarity score: {rec['best_score']:.3f}")
        
        if rec['all_matches']:
            print(f"  Top matches:")
            for j, (person, score) in enumerate(rec['all_matches']):
                similarity_percent = score * 100
                print(f"    {j+1}. {person}: {similarity_percent:.1f}%")
        
        # Interpret similarity score
        score = rec['best_score']
        if score > 0.8:
            confidence = "HIGH"
        elif score > 0.6:
            confidence = "MEDIUM"
        elif score > 0.4:
            confidence = "LOW"
        else:
            confidence = "VERY LOW"
        
        print(f"  Confidence level: {confidence}")

# -------------------------
# Main function
# -------------------------


def main():
    args = parse_args()
    
    print("Initializing models...")
    app, client = initialize_models(args)
    
    print(f"Recognizing faces in: {args.query_image}")
    result_img, faces, recognitions = recognize_faces(args.query_image, app, client, args)
    
    if result_img is None:
        return
    
    if recognitions:
        # Display with cv2
        display_results_cv2(result_img, recognitions, args)
        
        # Display detailed results in console
        display_detailed_results(recognitions)
        
    else:
        print("\nNo faces recognized in the image.")
        # Still show the image if faces were detected but not recognized
        if faces:
            cv2.imshow("Original Image", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    
# -------------------------
if __name__ == "__main__":
    main()