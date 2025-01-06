import numpy as np
import cv2
import svgwrite
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import find_contours
from scipy.ndimage import distance_transform_edt
from pathlib import Path

class SkeletonSVGConverter:
    def __init__(self, min_stroke_width=1.0, max_stroke_width=8.0):
        self.min_stroke_width = min_stroke_width
        self.max_stroke_width = max_stroke_width
        
    def preprocess_image(self, image_path):
        """Preprocess image for skeletonization"""
        # Read and convert to grayscale
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
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

    def create_skeleton(self, binary_image):
        """Create skeleton and get distance transform"""
        # Create skeleton
        skeleton = skeletonize(binary_image > 0)
        
        # Get distance transform for stroke width
        dist_transform = distance_transform_edt(binary_image > 0)
        
        return skeleton, dist_transform
        
    def extract_paths(self, skeleton):
        """Extract paths from skeleton"""
        # Find branch points and endpoints
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ])
        conv = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        branch_points = conv > 11
        end_points = conv == 11
        
        # Create visited mask
        visited = np.zeros_like(skeleton, dtype=bool)
        paths = []
        
        # Function to trace path
        def trace_path(y, x):
            path = [(y, x)]
            visited[y, x] = True
            
            while True:
                # Get 8-connected neighbors
                y, x = path[-1]
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < skeleton.shape[0] and 
                            0 <= nx < skeleton.shape[1] and 
                            skeleton[ny, nx] and 
                            not visited[ny, nx]):
                            neighbors.append((ny, nx))
                
                if not neighbors:
                    break
                    
                # Follow path
                next_point = neighbors[0]
                path.append(next_point)
                visited[next_point] = True
                
            return path
        
        # Find all paths
        for y in range(skeleton.shape[0]):
            for x in range(skeleton.shape[1]):
                if skeleton[y, x] and not visited[y, x]:
                    if branch_points[y, x] or end_points[y, x]:
                        path = trace_path(y, x)
                        if len(path) > 2:
                            paths.append(path)
        
        return paths
        
    def smooth_path(self, path, smoothing=0.2):
        """Apply Chaikin smoothing to path"""
        if len(path) < 3:
            return path
            
        points = np.array(path)
        smooth_points = []
        
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            
            q = tuple((1 - smoothing) * np.array(p0) + smoothing * np.array(p1))
            r = tuple(smoothing * np.array(p0) + (1 - smoothing) * np.array(p1))
            
            smooth_points.extend([q, r])
            
        return smooth_points
        
    def get_stroke_width(self, path, dist_transform):
        """Get varying stroke width along path"""
        widths = []
        for y, x in path:
            width = dist_transform[int(y), int(x)] * 2
            widths.append(width)
            
        # Smooth width variations
        smoothed_widths = np.convolve(widths, np.ones(3)/3, mode='same')
        return np.clip(smoothed_widths, self.min_stroke_width, self.max_stroke_width)
        
    def convert_to_svg(self, image_path, output_path):
        """Convert image to SVG using skeletonization"""
        # Preprocess image
        binary = self.preprocess_image(image_path)
        
        # Create skeleton and get distance transform
        skeleton, dist_transform = self.create_skeleton(binary)
        
        # Extract paths
        paths = self.extract_paths(skeleton)
        
        # Create SVG
        dwg = svgwrite.Drawing(output_path, profile='tiny')
        
        # Set viewbox
        height, width = binary.shape
        dwg.viewbox(0, 0, width, height)
        
        # Process each path
        for path in paths:
            # Smooth path
            smoothed = self.smooth_path(path)
            
            # Get varying stroke widths
            widths = self.get_stroke_width(path, dist_transform)
            
            # Create SVG path
            path_data = []
            for i, (y, x) in enumerate(smoothed):
                cmd = 'M' if i == 0 else 'L'
                path_data.append(f"{cmd}{x:.1f},{y:.1f}")
                
            # Create path with varying stroke width
            avg_width = np.mean(widths)
            path = dwg.path(
                d=' '.join(path_data),
                stroke='black',
                stroke_width=f"{avg_width:.1f}",
                stroke_linecap='round',
                stroke_linejoin='round',
                fill='none'
            )
            dwg.add(path)
            
        # Save SVG
        dwg.save(pretty=True)
        
def batch_convert(input_folder, output_folder):
    """Process multiple images"""
    converter = SkeletonSVGConverter()
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    for img_path in input_path.glob('*.png'):
        try:
            out_file = output_path / f"{img_path.stem}.svg"
            converter.convert_to_svg(img_path, out_file)
            print(f"Successfully converted: {img_path.name}")
        except Exception as e:
            print(f"Error converting {img_path.name}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    batch_convert("processed_images_clean", "processed_svgs_clean")