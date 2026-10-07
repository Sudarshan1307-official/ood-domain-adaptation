import numpy as np
import cv2
import joblib
from sklearn.svm import SVC
from skimage.feature import hog

np.random.seed(42)

CLASSES = [
    "Circle", "Square", "Triangle", "Rectangle", "Oval_Ellipse", "Rhombus", "Parallelogram", "Trapezoid", "Kite", "Pentagon",
    "Hexagon", "Heptagon", "Octagon", "Nonagon", "Decagon", "Hendecagon", "Dodecagon", "Tridecagon", "Tetradecagon", "Pentadecagon",
    "Equilateral_Triangle", "Isosceles_Triangle", "Scalene_Triangle", "Right_Angled_Triangle", "Acute_Triangle", "Obtuse_Triangle",
    "Cube", "Cuboid", "Sphere", "Cylinder", "Cone", "Pyramid", "Prism", "Tetrahedron", "Octahedron", "Dodecahedron",
    "Icosahedron", "Hemisphere", "Torus", "Ellipsoid", "Frustum", "Triangular_Prism", "Hexagonal_Prism", "Square_Pyramid",
    "Star", "Heart", "Crescent", "Cross", "Arrow", "Diamond", "Semicircle", "Ring", "Spiral", "Teardrop",
    "Shield", "Hexagram", "Pentagram", "Chevron", "Plus_Symbol", "Infinity_Symbol", "Hexadecagon", "Heptadecagon", "Octadecagon", "Enneadecagon",
    "Icosagon", "Triacontagon", "Tetracontagon", "Pentacontagon", "Hexacontagon", "Heptacontagon", "Octacontagon", "Enneacontagon", "Hectagon", "Astroid",
    "Cardioid", "Cycloid", "Hyperboloid", "Paraboloid", "Deltoid", "Trefoil_Knot", "Quatrefoil", "Superellipse", "Folium", "Cissoid",
    "Conchoid", "Lissajous_Curve", "Ovoid", "Snub_Cube", "Rhombicosidodecahedron", "Great_Dodecahedron", "Catenary", "Tractrix", "Lituus", "Archimedean_Spiral",
    "Logarithmic_Spiral", "Helix", "Mobius_Strip", "Klein_Bottle", "Bipyramid", "Apeirogon",
]


def generate_synthetic_shape_matrix(shape_name, apply_ood_noise=False):
    img = np.zeros((64, 64), dtype=np.uint8)
    center = (32, 32)
    if "Circle" in shape_name or "Oval" in shape_name or "Torus" in shape_name:
        cv2.circle(img, center, 20, 255, -1)
    elif "Triangle" in shape_name:
        pts = np.array([[32, 12], [12, 52], [52, 52]], np.int32)
        cv2.fillPoly(img, [pts], 255)
    elif "Diamond" in shape_name or "Rhombus" in shape_name or "Kite" in shape_name:
        pts = np.array([[32, 10], [14, 32], [32, 54], [50, 32]], np.int32)
        cv2.fillPoly(img, [pts], 255)
    else:
        cv2.rectangle(img, (14, 14), (50, 50), 255, -1)

    if apply_ood_noise:
        if np.random.rand() > 0.5:
            kernel = np.zeros((5, 5))
            kernel[2, :] = np.ones(5) / 5
            img = cv2.filter2D(img, -1, kernel)
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)


if __name__ == "__main__":
    X_base, X_rand, y = [], [], []
    for class_idx, name in enumerate(CLASSES):
        for _ in range(15):
            X_base.append(generate_synthetic_shape_matrix(name, apply_ood_noise=False))
            X_rand.append(generate_synthetic_shape_matrix(name, apply_ood_noise=True))
            y.append(class_idx)

    base_model = SVC(probability=True, kernel="linear")
    base_model.fit(np.array(X_base), np.array(y))

    randomized_model = SVC(probability=True, kernel="linear")
    randomized_model.fit(np.array(X_rand), np.array(y))

    # compress=3 keeps the files well under GitHub's 25MB web-upload limit
    joblib.dump(base_model, "baseline_model.pkl", compress=3)
    joblib.dump(randomized_model, "randomized_model.pkl", compress=3)
    print("Success! Compressed models generated.")
