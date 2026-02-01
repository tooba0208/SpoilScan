# SpoilScan
SpoilScan is an AI-powered fruit quality assessment system that uses image analysis to detect fungal spoilage, track degradation stages, and support sustainable postharvest decision-making.
Complete Orange Quality Detection System
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class OrangeQualityDetector:
    def __init__(self):
        # Color ranges for different conditions (in HSV)
        self.color_ranges = {
            'healthy': {
                'lower': np.array([10, 100, 100]),
                'upper': np.array([25, 255, 255])
            },
            'anthracnose': {  # Black spots/fungal infection
                'lower': np.array([0, 0, 0]),
                'upper': np.array([180, 255, 50])
            },
            'mold': {  # Blue/green mold
                'lower': np.array([70, 50, 50]),
                'upper': np.array([100, 255, 255])
            },
            'rot': {  # Brown rot
                'lower': np.array([5, 50, 50]),
                'upper': np.array([15, 255, 150])
            }
        }
        
        # Load or initialize SVM classifier
        self.classifier = None
        self.feature_names = None
        
    def preprocess_image(self, image_path):
        """Load and preprocess the image"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize for consistency
        img = cv2.resize(img, (400, 400))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        return img, blurred
    
    def extract_color_features(self, image):
        """Extract color-based features from the image"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        features = []
        
        # Analyze color distribution in different regions
        height, width = hsv.shape[:2]
        
        # Divide image into 4 quadrants
        quadrants = [
            hsv[0:height//2, 0:width//2],      # Top-left
            hsv[0:height//2, width//2:width],   # Top-right
            hsv[height//2:height, 0:width//2],  # Bottom-left
            hsv[height//2:height, width//2:width] # Bottom-right
        ]
        
        for quadrant in quadrants:
            # Calculate mean HSV values
            mean_h = np.mean(quadrant[:,:,0])
            mean_s = np.mean(quadrant[:,:,1])
            mean_v = np.mean(quadrant[:,:,2])
            
            # Calculate standard deviation
            std_h = np.std(quadrant[:,:,0])
            std_s = np.std(quadrant[:,:,1])
            std_v = np.std(quadrant[:,:,2])
            
            features.extend([mean_h, mean_s, mean_v, std_h, std_s, std_v])
        
        # Calculate overall color statistics
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        features.extend(hist_h[:20])  # First 20 bins of hue histogram
        features.extend(hist_s[:20])  # First 20 bins of saturation histogram
        features.extend(hist_v[:20])  # First 20 bins of value histogram
        
        return np.array(features)
    
    def detect_defects(self, image):
        """Detect defects using multiple techniques"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for orange color
        orange_mask = cv2.inRange(hsv, 
                                 self.color_ranges['healthy']['lower'],
                                 self.color_ranges['healthy']['upper'])
        
        # Create masks for defects
        defect_masks = {}
        for defect_type, color_range in self.color_ranges.items():
            if defect_type != 'healthy':
                mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                defect_masks[defect_type] = mask
        
        # Combine all defect masks
        all_defects_mask = np.zeros_like(orange_mask)
        for mask in defect_masks.values():
            all_defects_mask = cv2.bitwise_or(all_defects_mask, mask)
        
        # Apply morphological operations to clean up masks
        kernel = np.ones((5,5), np.uint8)
        all_defects_mask = cv2.morphologyEx(all_defects_mask, cv2.MORPH_CLOSE, kernel)
        all_defects_mask = cv2.morphologyEx(all_defects_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of defects
        contours, _ = cv2.findContours(all_defects_mask, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        return orange_mask, all_defects_mask, contours, defect_masks
    
    def analyze_defects(self, image, contours, defect_masks):
        """Analyze detected defects and classify infection type"""
        results = {
            'status': 'Healthy',
            'infection_type': 'None',
            'defect_percentage': 0,
            'defect_count': len(contours),
            'defect_details': {}
        }
        
        if len(contours) > 0:
            results['status'] = 'Spoiled'
            
            # Calculate defect percentage
            total_pixels = image.shape[0] * image.shape[1]
            defect_pixels = 0
            
            # Analyze each defect type
            for defect_type, mask in defect_masks.items():
                defect_area = np.sum(mask > 0)
                percentage = (defect_area / total_pixels) * 100
                
                results['defect_details'][defect_type] = {
                    'percentage': percentage,
                    'area': defect_area
                }
                
                defect_pixels += defect_area
            
            results['defect_percentage'] = (defect_pixels / total_pixels) * 100
            
            # Determine primary infection type
            max_percentage = 0
            primary_infection = 'None'
            
            for defect_type, details in results['defect_details'].items():
                if details['percentage'] > max_percentage:
                    max_percentage = details['percentage']
                    primary_infection = defect_type
            
            results['infection_type'] = primary_infection
            
            # Classify severity
            if results['defect_percentage'] < 5:
                results['severity'] = 'Minor'
            elif results['defect_percentage'] < 20:
                results['severity'] = 'Moderate'
            else:
                results['severity'] = 'Severe'
        
        return results
    
    def extract_texture_features(self, image):
        """Extract texture features using GLCM-like calculations"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features
        features = []
        
        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        features.append(np.mean(sobelx))
        features.append(np.mean(sobely))
        features.append(np.std(sobelx))
        features.append(np.std(sobely))
        
        # Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(np.mean(laplacian))
        features.append(np.std(laplacian))
        
        # Calculate entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features.append(entropy)
        
        return np.array(features)
    
    def train_classifier(self, image_folder, labels):
        """
        Train a classifier on labeled orange images
        image_folder: path to folder containing images
        labels: list of labels corresponding to images
        """
        features_list = []
        
        # Extract features from all images
        for img_file in os.listdir(image_folder):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(image_folder, img_file)
                img, blurred = self.preprocess_image(img_path)
                
                # Extract color features
                color_features = self.extract_color_features(blurred)
                
                # Extract texture features
                texture_features = self.extract_texture_features(blurred)
                
                # Combine features
                combined_features = np.concatenate([color_features, texture_features])
                features_list.append(combined_features)
        
        # Create feature matrix
        X = np.array(features_list)
        y = np.array(labels[:len(X)])  # Ensure labels match number of images
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train SVM classifier
        self.classifier = SVC(kernel='rbf', probability=True, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save classifier
        joblib.dump(self.classifier, 'orange_classifier.pkl')
        
        return self.classifier
    
    def predict_with_classifier(self, image_path):
        """Predict using trained classifier"""
        if self.classifier is None:
            # Try to load pre-trained classifier
            try:
                self.classifier = joblib.load('orange_classifier.pkl')
            except:
                print("No trained classifier found. Using rule-based detection.")
                return self.analyze_single_image(image_path)
        
        # Extract features
        img, blurred = self.preprocess_image(image_path)
        color_features = self.extract_color_features(blurred)
        texture_features = self.extract_texture_features(blurred)
        combined_features = np.concatenate([color_features, texture_features])
        
        # Predict
        prediction = self.classifier.predict([combined_features])[0]
        probabilities = self.classifier.predict_proba([combined_features])[0]
        
        return {
            'predicted_class': prediction,
            'probabilities': dict(zip(self.classifier.classes_, probabilities))
        }
    
    def analyze_single_image(self, image_path):
        """Complete analysis of a single orange image"""
        print(f"\n{'='*50}")
        print(f"Analyzing: {image_path}")
        print(f"{'='*50}")
        
        # Load and preprocess image
        original, processed = self.preprocess_image(image_path)
        
        # Detect defects
        orange_mask, defects_mask, contours, defect_masks = self.detect_defects(processed)
        
        # Analyze results
        results = self.analyze_defects(processed, contours, defect_masks)
        
        # Display results
        self.display_results(original, processed, orange_mask, defects_mask, contours, results)
        
        return results
    
    def display_results(self, original, processed, orange_mask, defects_mask, contours, results):
        """Display analysis results with visualization"""
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Processed image
        axes[0, 1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Processed Image')
        axes[0, 1].axis('off')
        
        # Orange mask
        axes[0, 2].imshow(orange_mask, cmap='gray')
        axes[0, 2].set_title('Orange Region Mask')
        axes[0, 2].axis('off')
        
        # Defects mask
        axes[1, 0].imshow(defects_mask, cmap='hot')
        axes[1, 0].set_title('Detected Defects')
        axes[1, 0].axis('off')
        
        # Image with contours
        img_with_contours = original.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 0, 255), 2)
        axes[1, 1].imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Defect Contours: {len(contours)} found')
        axes[1, 1].axis('off')
        
        # Results text
        axes[1, 2].axis('off')
        info_text = f"""
        ANALYSIS RESULTS:
        ------------------
        Status: {results['status']}
        Infection Type: {results['infection_type']}
        Severity: {results.get('severity', 'N/A')}
        Defect Percentage: {results['defect_percentage']:.2f}%
        Number of Defects: {results['defect_count']}
        
        DEFECT DETAILS:
        """
        for defect_type, details in results['defect_details'].items():
            info_text += f"\n{defect_type}: {details['percentage']:.2f}%"
        
        axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, 
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        # Print results to console
        print("\n" + "="*50)
        print("DETECTION RESULTS:")
        print("="*50)
        print(f"Status: {results['status']}")
        print(f"Primary Infection: {results['infection_type']}")
        print(f"Severity: {results.get('severity', 'N/A')}")
        print(f"Defect Coverage: {results['defect_percentage']:.2f}%")
        print(f"Number of Defect Areas: {results['defect_count']}")
        
        if results['defect_details']:
            print("\nDetailed Defect Analysis:")
            for defect_type, details in results['defect_details'].items():
                print(f"  {defect_type}: {details['percentage']:.2f}% coverage")

def create_sample_dataset():
    """Create a sample dataset directory structure"""
    import shutil
    
    dataset_dir = "orange_dataset"
    categories = ['healthy', 'anthracnose', 'mold', 'rot']
    
    # Remove existing dataset
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    # Create directories
    for category in categories:
        os.makedirs(os.path.join(dataset_dir, category), exist_ok=True)
    
    print(f"Created dataset directory structure at: {dataset_dir}")
    print("Please add your orange images to the respective folders:")
    for category in categories:
        print(f"  - {dataset_dir}/{category}/")
    
    return dataset_dir

def batch_process_images(image_folder):
    """Process all images in a folder"""
    detector = OrangeQualityDetector()
    
    results_summary = {
        'healthy': 0,
        'spoiled': 0,
        'infections': {}
    }
    
    for img_file in os.listdir(image_folder):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, img_file)
            
            try:
                results = detector.analyze_single_image(img_path)
                
                # Update summary
                if results['status'] == 'Healthy':
                    results_summary['healthy'] += 1
                else:
                    results_summary['spoiled'] += 1
                    infection = results['infection_type']
                    results_summary['infections'][infection] = \
                        results_summary['infections'].get(infection, 0) + 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images processed: {results_summary['healthy'] + results_summary['spoiled']}")
    print(f"Healthy oranges: {results_summary['healthy']}")
    print(f"Spoiled oranges: {results_summary['spoiled']}")
    
    if results_summary['infections']:
        print("\nInfection Distribution:")
        for infection, count in results_summary['infections'].items():
            print(f"  {infection}: {count} oranges")

# Main execution
if __name__ == "__main__":
    # Initialize detector
    detector = OrangeQualityDetector()
    
    # Create sample dataset structure (if needed)
    # dataset_dir = create_sample_dataset()
    
    # Example 1: Analyze a single image
    print("Example 1: Single Image Analysis")
    print("-" * 40)
    
    # Replace with your image path
    sample_image_path = "orange_sample.jpg"  # Change this to your image path
    
    if os.path.exists(sample_image_path):
        results = detector.analyze_single_image(sample_image_path)
    else:
        print(f"Sample image not found at: {sample_image_path}")
        print("Please provide a valid image path.")
        
        # Alternative: Use webcam for real-time detection
        print("\n" + "="*60)
        print("REAL-TIME DETECTION USING WEBCAM")
        print("="*60)
        print("Press 'q' to quit")
        
        # Real-time detection using webcam
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (400, 400))
            
            # Detect defects
            orange_mask, defects_mask, contours, defect_masks = detector.detect_defects(frame)
            
            # Analyze
            results = detector.analyze_defects(frame, contours, defect_masks)
            
            # Display results on frame
            cv2.putText(frame, f"Status: {results['status']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Infection: {results['infection_type']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Defects: {results['defect_percentage']:.1f}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw contours
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
            
            # Display
            cv2.imshow('Orange Quality Detection', frame)
            
            # Show defect mask
            cv2.imshow('Defect Detection', defects_mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Example 2: Batch processing
    print("\n\nExample 2: Batch Processing")
    print("-" * 40)
    
    # Replace with your folder path
    image_folder = "orange_images"  # Change this to your folder path
    
    if os.path.exists(image_folder):
        batch_process_images(image_folder)
    else:
        print(f"Image folder not found: {image_folder}")
    
    print("\nProgram completed successfully!")
Installation Requirements
Create a requirements.txt file:
txt
Copy
Download
opencv-python==4.8.1.78
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
joblib==1.3.2
Install with:
bash
Copy
Download
pip install -r requirements.txt
How to Use the Code:
1. Single Image Analysis
python
Copy
Download
detector = OrangeQualityDetector()
results = detector.analyze_single_image("path/to/your/orange.jpg")
2. Batch Processing
python
Copy
Download
batch_process_images("path/to/orange/folder")
3. Train Custom Classifier
python
Copy
Download
# Prepare your dataset with labeled images
image_folder = "dataset/images"
labels = ["healthy", "rot", "mold", ...]  # Corresponding labels

detector = OrangeQualityDetector()
detector.train_classifier(image_folder, labels)

