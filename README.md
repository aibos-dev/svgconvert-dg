# svgconvert-dg
convert the generated  files into a more practical SVG format.

# Intoduction
AI-generated line art illustrations require conversion into vector graphics to allow designers to make manual edits. This involves transforming the raw images into formats compatible with illustration software, such as EPS files. The process includes binarization, skeletonization, vectorization, and final output as an EPS file readable by Adobe Illustrator.

In this module we make use of topological skeleton as a set of polylines from binary images as our input to obtain practical SVGs

# characteristics of practical SVGs
1. Path parameters are classified into reusable styles (e.g., CSS classes).
2. The number of paths is reduced.
