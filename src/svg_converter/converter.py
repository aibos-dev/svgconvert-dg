import numpy as np
import cv2
import svgwrite
from scipy.spatial.distance import euclidean
from pathlib import Path
from scipy.ndimage import distance_transform_edt

class StrokeAnalyzer:
    def __init__(self, image):
        self.image = image
        self.dist_transform = None
        self.compute_distance_transform()
        
    def compute_distance_transform(self):
        """Compute distance transform for stroke width estimation"""
        # Ensure binary image
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        else:
            binary = self.image
            
        # Compute distance transform
        self.dist_transform = distance_transform_edt(binary)
    
    def estimate_stroke_width(self, contour):
        """Estimate stroke width for a specific contour"""
        mask = np.zeros_like(self.image, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, 1)
        
        # Get points along the contour
        points = contour.reshape(-1, 2)
        
        # Sample distance transform values along the contour
        widths = []
        for point in points:
            x, y = point
            if 0 <= y < self.dist_transform.shape[0] and 0 <= x < self.dist_transform.shape[1]:
                width = self.dist_transform[int(y), int(x)]
                if width > 0:
                    widths.append(width)
        
        if not widths:
            return 2.0  # Default width
            
        # Calculate stroke width as twice the median distance
        stroke_width = np.median(widths) * 2
        
        # Normalize stroke width to reasonable range
        return max(1.0, min(stroke_width, 4.0))
    
    def is_outer_contour(self, contour):
        """Determine if contour is likely an outer contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return area > 1000 or circularity > 0.1

def process_image(image_path):
    """Pre-process image for better contour detection"""
    # Read image
    img = cv2.imread(str(image_path))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def convert_to_optimized_svg(input_path, output_path, min_length=10):
    """Convert PNG to optimized SVG with improved stroke handling"""
    # Process image
    binary = process_image(input_path)
    
    # Initialize stroke analyzer
    analyzer = StrokeAnalyzer(binary)
    
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,  # Retrieve contours in hierarchical order
        cv2.CHAIN_APPROX_TC89_KCOS
    )
    
    # Create SVG
    dwg = svgwrite.Drawing(output_path, profile='tiny')
    
    # Calculate viewBox
    if contours:
        all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours])
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        margin = 10
        dwg.viewbox(
            x_min - margin,
            y_min - margin,
            (x_max - x_min) + 2 * margin,
            (y_max - y_min) + 2 * margin
        )
    
    # Process contours in two passes - outer contours first, then inner details
    for pass_num in range(2):
        for i, contour in enumerate(contours):
            if len(contour) < min_length:
                continue
                
            is_outer = analyzer.is_outer_contour(contour)
            
            # First pass: process outer contours, Second pass: process inner details
            if (pass_num == 0 and not is_outer) or (pass_num == 1 and is_outer):
                continue
            
            # Estimate appropriate stroke width
            base_width = analyzer.estimate_stroke_width(contour)
            stroke_width = base_width * (1.5 if is_outer else 1.0)
            
            # Simplify contour
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Create path data
            points = approx.reshape(-1, 2)
            path_data = []
            for j, point in enumerate(points):
                cmd = 'M' if j == 0 else 'L'
                path_data.append(f"{cmd}{point[0]:.1f},{point[1]:.1f}")
            
            # Close the path if it's a closed contour
            if np.allclose(points[0], points[-1]):
                path_data.append('Z')
            
            # Add path to SVG
            path = dwg.path(
                d=' '.join(path_data),
                stroke='black',
                stroke_width=f"{stroke_width:.1f}",
                stroke_linecap='round',
                stroke_linejoin='round',
                fill='none'
            )
            dwg.add(path)
    
    # Save SVG
    dwg.save(pretty=True)

def batch_convert(input_folder, output_folder):
    """Process multiple images"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    for img_path in input_path.glob('*.png'):
        try:
            out_file = output_path / f"{img_path.stem}_optimized.svg"
            convert_to_optimized_svg(img_path, out_file)
            print(f"Successfully converted: {img_path.name}")
        except Exception as e:
            print(f"Error converting {img_path.name}: {str(e)}")

if __name__ == "__main__":
    # Example usage
     batch_convert("processed_images_clean", "processed_svgs_clean")