# trace_skeleton.py
# Trace skeletonization result into polylines
#
# Lingdong Huang 2020

import numpy as np
import cv2
import svgwrite
from scipy.spatial.distance import euclidean

# binary image thinning (skeletonization) in-place.
# implements Zhang-Suen algorithm.
# http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf
# @param im   the binary image
def thinningZS(im):
  prev = np.zeros(im.shape,np.int64);
  while True:
    im = thinningZSIteration(im,1);
    im = thinningZSIteration(im,0)
    diff = np.sum(np.abs(prev-im));
    if not diff:
      break
    prev = im
  return im

# 1 pass of Zhang-Suen thinning 
def thinningZSIteration(im, iter):
  marker = np.zeros(im.shape,np.int64);
  for i in range(1,im.shape[0]-1):
    for j in range(1,im.shape[1]-1):
      p2 = im[(i-1),j]  ;
      p3 = im[(i-1),j+1];
      p4 = im[(i),j+1]  ;
      p5 = im[(i+1),j+1];
      p6 = im[(i+1),j]  ;
      p7 = im[(i+1),j-1];
      p8 = im[(i),j-1]  ;
      p9 = im[(i-1),j-1];
      A  = (p2 == 0 and p3) + (p3 == 0 and p4) + \
           (p4 == 0 and p5) + (p5 == 0 and p6) + \
           (p6 == 0 and p7) + (p7 == 0 and p8) + \
           (p8 == 0 and p9) + (p9 == 0 and p2);
      B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
      m1 = (p2 * p4 * p6) if (iter == 0 ) else (p2 * p4 * p8);
      m2 = (p4 * p6 * p8) if (iter == 0 ) else (p2 * p6 * p8);

      if (A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0):
        marker[i,j] = 1;

  return np.bitwise_and(im,np.bitwise_not(marker))


def thinningSkimage(im):
  from skimage.morphology import skeletonize
  return skeletonize(im).astype(np.int64)

def thinning(im):
  try:
    return thinningSkimage(im)
  except:
    return thinningZS(im)

#check if a region has any white pixel
def notEmpty(im, x, y, w, h):
  return np.sum(im) > 0


# merge ith fragment of second chunk to first chunk
# @param c0   fragments from first  chunk
# @param c1   fragments from second chunk
# @param i    index of the fragment in first chunk
# @param sx   (x or y) coordinate of the seam
# @param isv  is vertical, not horizontal?
# @param mode 2-bit flag, 
#             MSB = is matching the left (not right) end of the fragment from first  chunk
#             LSB = is matching the right (not left) end of the fragment from second chunk
# @return     matching successful?             
# 
def mergeImpl(c0, c1, i, sx, isv, mode):

  B0 = (mode >> 1 & 1)>0; # match c0 left
  B1 = (mode >> 0 & 1)>0; # match c1 left
  mj = -1;
  md = 4; # maximum offset to be regarded as continuous
  
  p1 = c1[i][0 if B1 else -1];
  
  if (abs(p1[isv]-sx)>0): # not on the seam, skip
    return False
  
  # find the best match
  for j in range(len(c0)):
    p0 = c0[j][0 if B0 else -1];
    if (abs(p0[isv]-sx)>1): # not on the seam, skip
      continue
    
    d = abs(p0[not isv] - p1[not isv]);
    if (d < md):
      mj = j;
      md = d;

  if (mj != -1): # best match is good enough, merge them
    if (B0 and B1):
      c0[mj] = list(reversed(c1[i])) + c0[mj]
    elif (not B0 and B1):
      c0[mj]+=c1[i]
    elif (B0 and not B1):
      c0[mj] = c1[i] + c0[mj]
    else:
      c0[mj] += list(reversed(c1[i]))
    
    c1.pop(i);
    return True;
  return False;

HORIZONTAL = 1;
VERTICAL = 2;

# merge fragments from two chunks
# @param c0   fragments from first  chunk
# @param c1   fragments from second chunk
# @param sx   (x or y) coordinate of the seam
# @param dr   merge direction, HORIZONTAL or VERTICAL?
# 
def mergeFrags(c0, c1, sx, dr):
  for i in range(len(c1)-1,-1,-1):
    if (dr == HORIZONTAL):
      if (mergeImpl(c0,c1,i,sx,False,1)):continue;
      if (mergeImpl(c0,c1,i,sx,False,3)):continue;
      if (mergeImpl(c0,c1,i,sx,False,0)):continue;
      if (mergeImpl(c0,c1,i,sx,False,2)):continue;
    else:
      if (mergeImpl(c0,c1,i,sx,True,1)):continue;
      if (mergeImpl(c0,c1,i,sx,True,3)):continue;
      if (mergeImpl(c0,c1,i,sx,True,0)):continue;
      if (mergeImpl(c0,c1,i,sx,True,2)):continue;      
    
  c0 += c1


# recursive bottom: turn chunk into polyline fragments;
# look around on 4 edges of the chunk, and identify the "outgoing" pixels;
# add segments connecting these pixels to center of chunk;
# apply heuristics to adjust center of chunk
# 
# @param im   the bitmap image
# @param x    left of   chunk
# @param y    top of    chunk
# @param w    width of  chunk
# @param h    height of chunk
# @return     the polyline fragments
# 
def chunkToFrags(im, x, y, w, h):
  frags = []
  on = False; # to deal with strokes thicker than 1px
  li=-1; lj=-1;
  
  # walk around the edge clockwise
  for k in range(h+h+w+w-4):
    i=0; j=0;
    if (k < w):
      i = y+0; j = x+k;
    elif (k < w+h-1):
      i = y+k-w+1; j = x+w-1;
    elif (k < w+h+w-2):
      i = y+h-1; j = x+w-(k-w-h+3); 
    else:
      i = y+h-(k-w-h-w+4); j = x+0;
    
    if (im[i,j]): # found an outgoing pixel
      if (not on):     # left side of stroke
        on = True;
        frags.append([[j,i],[x+w//2,y+h//2]])
    else:
      if (on):# right side of stroke, average to get center of stroke
        frags[-1][0][0]= (frags[-1][0][0]+lj)//2;
        frags[-1][0][1]= (frags[-1][0][1]+li)//2;
        on = False;
    li = i;
    lj = j;
  
  if (len(frags) == 2): # probably just a line, connect them
    f = [frags[0][0],frags[1][0]];
    frags.pop(0);
    frags.pop(0);
    frags.append(f);
  elif (len(frags) > 2): # it's a crossroad, guess the intersection
    ms = 0;
    mi = -1;
    mj = -1;
    # use convolution to find brightest blob
    for i in range(y+1,y+h-1):
      for j in range(x+1,x+w-1):
        s = \
          (im[i-1,j-1]) + (im[i-1,j]) +(im[i-1,j+1])+\
          (im[i,j-1]  ) +   (im[i,j]) +    (im[i,j+1])+\
          (im[i+1,j-1]) + (im[i+1,j]) +  (im[i+1,j+1]);
        if (s > ms):
          mi = i;
          mj = j;
          ms = s;
        elif (s == ms and abs(j-(x+w//2))+abs(i-(y+h//2)) < abs(mj-(x+w//2))+abs(mi-(y+h//2))):
          mi = i;
          mj = j;
          ms = s;

    if (mi != -1):
      for i in range(len(frags)):
        frags[i][1]=[mj,mi]
  return frags;


# Trace skeleton from thinning result.
# Algorithm:
# 1. if chunk size is small enough, reach recursive bottom and turn it into segments
# 2. attempt to split the chunk into 2 smaller chunks, either horizontall or vertically;
#    find the best "seam" to carve along, and avoid possible degenerate cases
# 3. recurse on each chunk, and merge their segments
# 
# @param im      the bitmap image
# @param x       left of   chunk
# @param y       top of    chunk
# @param w       width of  chunk
# @param h       height of chunk
# @param csize   chunk size
# @param maxIter maximum number of iterations
# @param rects   if not null, will be populated with chunk bounding boxes (e.g. for visualization)
# @return        an array of polylines
# 
def traceSkeleton(im, x, y, w, h, csize, maxIter, rects):
  
  frags = []
  
  if (maxIter == 0): # gameover
    return frags;
  if (w <= csize and h <= csize): # recursive bottom
    frags += chunkToFrags(im,x,y,w,h);
    return frags;
  
  ms = im.shape[0]+im.shape[1]; # number of white pixels on the seam, less the better
  mi = -1; # horizontal seam candidate
  mj = -1; # vertical   seam candidate
  
  if (h > csize): # try splitting top and bottom
    for i in range(y+3,y+h-3):
      if (im[i,x]  or im[(i-1),x]  or im[i,x+w-1]  or im[(i-1),x+w-1]):
        continue
      
      s = 0;
      for j in range(x,x+w):
        s += im[i,j];
        s += im[(i-1),j];
      
      if (s < ms):
        ms = s; mi = i;
      elif (s == ms  and  abs(i-(y+h//2))<abs(mi-(y+h//2))):
        # if there is a draw (very common), we want the seam to be near the middle
        # to balance the divide and conquer tree
        ms = s; mi = i;
  
  if (w > csize): # same as above, try splitting left and right
    for j in range(x+3,x+w-2):
      if (im[y,j] or im[(y+h-1),j] or im[y,j-1] or im[(y+h-1),j-1]):
        continue
      
      s = 0;
      for i in range(y,y+h):
        s += im[i,j];
        s += im[i,j-1];
      if (s < ms):
        ms = s;
        mi = -1; # horizontal seam is defeated
        mj = j;
      elif (s == ms  and  abs(j-(x+w//2))<abs(mj-(x+w//2))):
        ms = s;
        mi = -1;
        mj = j;

  nf = []; # new fragments
  if (h > csize  and  mi != -1): # split top and bottom
    L = [x,y,w,mi-y];    # new chunk bounding boxes
    R = [x,mi,w,y+h-mi];
    
    if (notEmpty(im,L[0],L[1],L[2],L[3])): # if there are no white pixels, don't waste time
      if(rects!=None):rects.append(L);
      nf += traceSkeleton(im,L[0],L[1],L[2],L[3],csize,maxIter-1,rects) # recurse
    
    if (notEmpty(im,R[0],R[1],R[2],R[3])):
      if(rects!=None):rects.append(R);
      mergeFrags(nf,traceSkeleton(im,R[0],R[1],R[2],R[3],csize,maxIter-1,rects),mi,VERTICAL);
    
  elif (w > csize  and  mj != -1): # split left and right
    L = [x,y,mj-x,h];
    R = [mj,y,x+w-mj,h];
    if (notEmpty(im,L[0],L[1],L[2],L[3])):
      if(rects!=None):rects.append(L);
      nf+=traceSkeleton(im,L[0],L[1],L[2],L[3],csize,maxIter-1,rects);
    
    if (notEmpty(im,R[0],R[1],R[2],R[3])):
      if(rects!=None):rects.append(R);
      mergeFrags(nf,traceSkeleton(im,R[0],R[1],R[2],R[3],csize,maxIter-1,rects),mj,HORIZONTAL);
    
  frags+=nf;
  if (mi == -1  and  mj == -1): # splitting failed! do the recursive bottom instead
    frags += chunkToFrags(im,x,y,w,h);
  
  return frags

def simplify_path(points, tolerance=2.0):
    """
    Simplify a path using the Ramer-Douglas-Peucker algorithm
    """
    if len(points) < 3:
        return points
        
    # Find point with maximum distance
    max_dist = 0
    index = 0
    first = points[0]
    last = points[-1]
    
    for i in range(1, len(points) - 1):
        dist = abs(np.cross(
            np.append(np.array(last) - np.array(first), 0),  
            np.append(np.array(points[i]) - np.array(first), 0)  # Extend to 3D
        )[2]) / euclidean(first, last)  

        if dist > max_dist:
            index = i
            max_dist = dist
            
    if max_dist > tolerance:
        left = simplify_path(points[:index + 1], tolerance)
        right = simplify_path(points[index:], tolerance)
        return left[:-1] + right
    else:
        return [first, last]

def estimate_stroke_width(point, im, radius=3):
    """
    Estimate stroke width by analyzing local neighborhood
    """
    x, y = int(point[0]), int(point[1])
    if x < 0 or y < 0 or x >= im.shape[1] or y >= im.shape[0]:
        return 2.0
        
    x_start = max(0, x - radius)
    x_end = min(im.shape[1], x + radius + 1)
    y_start = max(0, y - radius)
    y_end = min(im.shape[0], y + radius + 1)
    
    neighborhood = im[y_start:y_end, x_start:x_end]
    if np.any(neighborhood):
        width = float(np.sum(neighborhood)) / float(np.max(neighborhood))
        return min(max(width * 0.8, 2.0), 6.0)
    return 2.0

def merge_similar_paths(polys, threshold=5):
    """
    Merge paths that are likely part of the same stroke
    """
    merged = []
    used = set()
    
    for i, poly1 in enumerate(polys):
        if i in used:
            continue
            
        current = poly1.copy()
        used.add(i)
        
        changed = True
        while changed:
            changed = False
            for j, poly2 in enumerate(polys):
                if j in used or j == i:
                    continue
                    
                if (euclidean(current[-1], poly2[0]) < threshold or
                    euclidean(current[-1], poly2[-1]) < threshold or
                    euclidean(current[0], poly2[0]) < threshold or
                    euclidean(current[0], poly2[-1]) < threshold):
                    
                    dists = [
                        (euclidean(current[-1], poly2[0]), 'forward'),
                        (euclidean(current[-1], poly2[-1]), 'reverse'),
                        (euclidean(current[0], poly2[0]), 'reverse_first'),
                        (euclidean(current[0], poly2[-1]), 'forward_first')
                    ]
                    
                    best_merge = min(dists, key=lambda x: x[0])
                    if best_merge[1] == 'forward':
                        current.extend(poly2)
                    elif best_merge[1] == 'reverse':
                        current.extend(reversed(poly2))
                    elif best_merge[1] == 'reverse_first':
                        current = list(reversed(current))
                        current.extend(poly2)
                    else:
                        current = list(reversed(current))
                        current.extend(reversed(poly2))
                        
                    used.add(j)
                    changed = True
                    break
                    
        merged.append(current)
    return merged

def safe_sum(a, b):
    """
    Safely sum two numbers avoiding overflow
    """
    try:
        return float(a) + float(b)
    except:
        return float(0)

def process_and_save_image(input_image_path, output_svg_path):
    # Read and prepare image
    im0 = cv2.imread(input_image_path)
    im = (im0[:, :, 0] < 240).astype(np.int64)
    im = thinning(im)

    # Get initial polylines
    rects = []
    polys = traceSkeleton(im, 0, 0, im.shape[1], im.shape[0], 10, 999, rects)
    
    # Convert to point format and simplify
    processed_polys = []
    for poly in polys:
        points = [(p[0], p[1]) for p in poly]
        if len(points) >= 2:
            simplified = simplify_path(points, tolerance=2.0)
            if len(simplified) >= 2:
                processed_polys.append(simplified)
    
    # Merge similar paths
    merged_polys = merge_similar_paths(processed_polys, threshold=5)
    
    # Calculate padding and centered viewBox
    padding = 50  # Padding around the image in pixels
    width = im.shape[1]
    height = im.shape[0]
    
    # Create SVG with centered viewBox
    dwg = svgwrite.Drawing(output_svg_path)
    
    # Calculate the viewBox with padding
    viewbox_x = -padding
    viewbox_y = -padding
    viewbox_width = width + (2 * padding)
    viewbox_height = height + (2 * padding)
    
    # Set viewBox and ensure the SVG is responsive
    dwg.attribs['viewBox'] = f"{viewbox_x} {viewbox_y} {viewbox_width} {viewbox_height}"
    dwg.attribs['width'] = '100%'
    dwg.attribs['height'] = '100%'
    dwg.attribs['preserveAspectRatio'] = 'xMidYMid meet'
    
    # Create a group to hold all paths and translate it to account for padding
    g = dwg.g()
    
    # Add paths with individual styling
    for poly in merged_polys:
        # Calculate average stroke width for path
        widths = [estimate_stroke_width(p, im) for p in poly]
        avg_width = sum(widths) / len(widths) if widths else 2.0
        
        # Create path with inline styling
        path = dwg.polyline(
            points=poly,
            fill='none',
            stroke='black',
            stroke_width=avg_width,
            stroke_linecap='round',
            stroke_linejoin='round',
            opacity=0.85
        )
        g.add(path)
    
    # Add the group to the drawing
    dwg.add(g)
    
    dwg.save(pretty=True)

def batch_process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output directory does not exist. Creating: {output_folder}")
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"processed_{os.path.splitext(filename)[0]}.svg")
            
            try:
                process_and_save_image(input_path, output_path)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    import os
    input_folder = './processed_images_clean'
    output_folder = './processed_svgs_clean'
    batch_process_images(input_folder, output_folder)
