from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import cv2
import svgwrite
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import find_contours

# Constants
KERNEL_SIZE = (2, 2)
BRANCH_KERNEL = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2
SMOOTHING_FACTOR = 0.2
WIDTH_SMOOTHING_WINDOW = 3

class SkeletonSVGConverter:
    """Converts images to SVG using skeletonization technique."""
    
    def __init__(self, min_stroke_width: float = 1.0, max_stroke_width: float = 8.0):
        """
        Initialize the converter with stroke width parameters.
        
        Args:
            min_stroke_width: Minimum stroke width in the output SVG
            max_stroke_width: Maximum stroke width in the output SVG
        """
        self.min_stroke_width = min_stroke_width
        self.max_stroke_width = max_stroke_width
    
    def preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess image for skeletonization.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed binary image
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C
        )
        
        kernel = np.ones(KERNEL_SIZE, np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def create_skeleton(self, binary_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create skeleton and distance transform from binary image.
        
        Args:
            binary_image: Preprocessed binary image
            
        Returns:
            Tuple of (skeleton, distance transform)
        """
        skeleton = skeletonize(binary_image > 0)
        dist_transform = distance_transform_edt(binary_image > 0)
        return skeleton, dist_transform

    def _find_skeleton_points(self, skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find branch points and endpoints in skeleton."""
        conv = cv2.filter2D(skeleton.astype(np.uint8), -1, BRANCH_KERNEL)
        return conv > 11, conv == 11

    def _trace_single_path(self, skeleton: np.ndarray, start_y: int, start_x: int, 
                          visited: np.ndarray) -> List[Tuple[int, int]]:
        """Trace a single path from given starting point."""
        path = [(start_y, start_x)]
        visited[start_y, start_x] = True
        
        while True:
            y, x = path[-1]
            neighbors = []
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < skeleton.shape[0] and 
                        0 <= nx < skeleton.shape[1] and 
                        skeleton[ny, nx] and 
                        not visited[ny, nx]):
                        neighbors.append((ny, nx))
            
            if not neighbors:
                break
                
            next_point = neighbors[0]
            path.append(next_point)
            visited[next_point] = True
        
        return path

    def extract_paths(self, skeleton: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Extract all paths from skeleton."""
        branch_points, end_points = self._find_skeleton_points(skeleton)
        visited = np.zeros_like(skeleton, dtype=bool)
        paths = []
        
        for y in range(skeleton.shape[0]):
            for x in range(skeleton.shape[1]):
                if skeleton[y, x] and not visited[y, x] and (branch_points[y, x] or end_points[y, x]):
                    path = self._trace_single_path(skeleton, y, x, visited)
                    if len(path) > 2:
                        paths.append(path)
        
        return paths

    def smooth_path(self, path: List[Tuple[int, int]], smoothing: float = SMOOTHING_FACTOR) -> List[Tuple[float, float]]:
        """Apply Chaikin smoothing to path."""
        if len(path) < 3:
            return path
            
        points = np.array(path)
        smooth_points = []
        
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            q = tuple((1 - smoothing) * p0 + smoothing * p1)
            r = tuple(smoothing * p0 + (1 - smoothing) * p1)
            smooth_points.extend([q, r])
            
        return smooth_points

    def get_stroke_width(self, path: List[Tuple[int, int]], dist_transform: np.ndarray) -> np.ndarray:
        """Calculate stroke width along path."""
        widths = [dist_transform[int(y), int(x)] * 2 for y, x in path]
        smoothed = np.convolve(widths, np.ones(WIDTH_SMOOTHING_WINDOW)/WIDTH_SMOOTHING_WINDOW, mode='same')
        return np.clip(smoothed, self.min_stroke_width, self.max_stroke_width)

    def convert_to_svg(self, image_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        """Convert image to SVG using skeletonization."""
        binary = self.preprocess_image(image_path)
        skeleton, dist_transform = self.create_skeleton(binary)
        paths = self.extract_paths(skeleton)
        
        dwg = svgwrite.Drawing(str(output_path), profile='tiny')
        height, width = binary.shape
        dwg.viewbox(0, 0, width, height)
        
        for path in paths:
            smoothed = self.smooth_path(path)
            widths = self.get_stroke_width(path, dist_transform)
            
            path_data = [f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" 
                        for i, (y, x) in enumerate(smoothed)]
            
            dwg.add(dwg.path(
                d=' '.join(path_data),
                stroke='black',
                stroke_width=f"{np.mean(widths):.1f}",
                stroke_linecap='round',
                stroke_linejoin='round',
                fill='none'
            ))
            
        dwg.save(pretty=True)

def batch_convert(input_folder: Union[str, Path], output_folder: Union[str, Path]) -> None:
    """
    Process multiple images in a folder.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder for output SVGs
    """
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
    batch_convert("processed_images_clean", "processed_svgs_clean")
