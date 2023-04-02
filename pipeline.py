import os
import cv2
import pickle
import numpy as np
import skimage.filters as filters
from tqdm import tqdm

import scripts.models as models
import scripts.extract as extract

# Save data as a pickle file
def save_data(data, path):
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

# Load data from a pickle file
def load_data(path):
    pickle_in = open(path,"rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data

# Apply a Gaussian blur
def blur(img):
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    return blurred

# Correct the lighting of an image to remove differences in illumination
def correct_lighting(img):

    # Generate a blurred copy of the original image
    smooth = cv2.GaussianBlur(img, (33,33), 0)

    # Divide the original image by the blurred copy
    division = cv2.divide(img, smooth, scale=255)

    # Apply an unsharp mask to the divided copy
    sharp = filters.unsharp_mask(division, radius=1, amount=0.1, channel_axis=True, preserve_range=False)

    # Clip the pixel brightness from 0 to 255
    sharp = (255*sharp).clip(0,255).astype(np.uint8)
    return sharp

def adjust_img(img, resolution=(100, 100), is_correct_lighting=True, is_blur=True, is_extract_face=True):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_correct_lighting:
        img = correct_lighting(img)
    if is_blur:
        img = blur(img)
    img = cv2.resize(img, resolution)
    return img

def preprocess(db_path, resolution=(100, 100), is_correct_lighting=True, is_blur=True, is_extract_face=True, scale=1, silent=False):

    file_name = f"/processed.pkl"
    if os.path.exists(db_path + file_name):

        if not silent:
            print(
                f"WARNING: Processed images in {db_path} folder were previously stored"
                + f" in {file_name}. If you added new instances after the creation, then please "
                + "delete this file and call find function again. It will create it again."
            )

        with open(f"{db_path}/{file_name}", "rb") as f:
            obj = pickle.load(f)

        if not silent:
            print("There are ", len(obj["vectors"]), " processed images found in ", file_name)
    
    else:
        img_paths = []
        labels = []
        imgs = []
        
        if is_extract_face:
            face_detector = extract.RetinaFace.build_model()

        for r, _, f in tqdm(os.walk(db_path)):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "\\" + file

                    img = cv2.imread(exact_path)
                    if is_extract_face:
                        output = extract.extract_face(img, face_detector=face_detector, resolution=resolution, scale=scale)
                        max_index = np.argmax([output[i]["coords"]["w"] for i in range(len(output))])
                        value = output[max_index]
                        img = value["face_img"]
                        cv2.imwrite(exact_path, img)
                    
                    img = adjust_img(
                        img = img, 
                        resolution=resolution,
                        is_correct_lighting=is_correct_lighting,
                        is_blur=is_blur,
                        is_extract_face=is_extract_face
                    )

                    imgs.append(img)
                    img_paths.append(exact_path)
                    labels.append(r.split("\\")[-1])

        if len(img_paths) == 0:
            raise ValueError(
                "There is no image in ",
                db_path,
                " folder! Validate .jpg or .png files exist in this path.",
            )
        
        labels = np.array(labels)
        vectors = np.array([img.flatten() for img in imgs])
        
        obj = {
            "img_paths": img_paths,
            "labels": labels,
            "imgs": imgs,
            "vectors": vectors
            }

        save_data(obj, os.path.join(db_path + file_name))

    return obj

def build_representations(X, y, resolution=(100, 100), n_components=50, exclude_indices=[], is_scaler=True, is_show_metrics=True, save_destination=None):
    pca = models.Principal_Component_Analysis(n_components, exclude_indices=exclude_indices)

    if is_scaler:
        scaler = models.Standard_Scaler()
        X = scaler.fit_transform(X)
    weights = pca.fit_transform(X)

    decomposition = {"weights": weights, "labels": y, "scaler": scaler, "pca": pca, "resolution": resolution}

    if is_show_metrics:
        pca.show_metrics(resolution)
    if save_destination:
        save_path = os.path.join(save_destination, "pca_decomposition.pkl")
        save_data(decomposition, save_path)
        print(f"PCA Decomposition was saved to {save_path}.")
    return decomposition

def run_pipeline(
        db_path, 
        resolution=(100, 100), 
        n_components=50, 
        is_correct_lighting=False, 
        is_blur=False,
        is_extract_face=False,
        scale=0.9,
        exclude_indices=[],
        is_scaler=True,
        is_evaluate=True,
        is_show_metrics=True,
        classifier="linearsvc",
    ):
    obj = preprocess(db_path, resolution=resolution, is_correct_lighting=is_correct_lighting, is_blur=is_blur, is_extract_face=is_extract_face, scale=scale)
    if is_evaluate:
        models.evaluate_models(obj["vectors"], obj["labels"], resolution=resolution, n_components=n_components, exclude_indices=exclude_indices, is_scaler=is_scaler, is_show_metrics=is_show_metrics, classifier=classifier)
    decomposition = build_representations(obj["vectors"], obj["labels"], resolution=resolution, n_components=n_components, is_show_metrics=is_show_metrics, save_destination=db_path)
    return decomposition

if __name__ == "__main__":
    run_pipeline(
        db_path="datasets/lfw_50_cropped", 
        exclude_indices=[], 
        resolution=(64, 64), 
        classifier="linearsvc", 
        is_show_metrics=True, 
        is_extract_face=False,
        is_scaler=True,
        is_blur=False,
        scale=0.9
    )











