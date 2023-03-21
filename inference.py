import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

import scripts.models as models
import scripts.extract
import pipeline

def build_model(X_train, y_train, classifier="euclidean"):
    if classifier not in ["euclidean", "knn", "linearsvc"]:
        print("Invalid classifier. Choose from euclidean, knn and svc.")
        return
    if classifier == "euclidean":
        built_model = models.Euclidean_Distance_Classifier()
    elif classifier == "knn":
        built_model = models.K_Nearest_Neighbors(neighbors=5)
    elif classifier == "linearsvc":
        built_model = models.svm.LinearSVC(dual=True)
    built_model.fit(X_train, y_train)
    return built_model

def predict(face, decomposition, model):
    # Train and evaluate classifier model
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, decomposition["resolution"])
    vector = np.array([face.flatten()])
    vector = decomposition["scaler"].transform(vector)
    weights = decomposition["pca"].transform(vector)
    return model.predict(weights)

def scalef(coeff, width, is_round=True):
    value = coeff * width ** 0.7
    return round(value) if is_round else value

def threshold_to_confidence(distance, threshold):
    return 100 * (1 - distance / threshold)

def show_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.gca().set_position((0, 0, 1, 1))
    plt.imshow(img)
    plt.show()

def recognize_image(img, model, decomposition, single=False, scale=1, show=True):

    face_detector = scripts.extract.RetinaFace.build_model()
    print("Extracting faces...")
    predictions = []
    results = scripts.extract.extract_face(img, face_detector, resolution=decomposition["resolution"], single=single, scale=scale)

    line_color = (3, 173, 255)
    for result in results:
        face_img, coords = result["face_img"], result["coords"]
        prediction = predict(face_img, decomposition, model)[0]
        predictions.append(prediction)
        x, y, w, h = coords["x"], coords["y"], coords["w"], coords["h"]

        cv2.rectangle(img, (x, y), (x + w, y + h), line_color, round(w / 40))
        cv2.putText(img, str(prediction), (x + scalef(1.7, w), y - scalef(0.4, w)), cv2.FONT_HERSHEY_SIMPLEX, scalef(0.05, w, is_round=False), (0, 0, 0), scalef(0.18, w))
        cv2.putText(img, str(prediction), (x + scalef(1.7, w), y - scalef(0.4, w)), cv2.FONT_HERSHEY_SIMPLEX, scalef(0.05, w, is_round=False), line_color, scalef(0.1, w))
    if show:
        show_image(img)

    return predictions, result, img


def model_inference(
        img_path, 
        db_path, 
        resolution=(100, 100), 
        n_components=50, 
        is_correct_lighting=False, 
        is_blur=False,
        is_extract_face=False,
        scale=0.9,
        exclude_indices=[],
        is_scaler=True,
        is_show_metrics=False,
        classifier="linearsvc"
    ):

    kwargs = {key: value for key, value in locals().items() if key not in ["img_path"]}

    file_name = os.path.join(db_path, "pca_decomposition.pkl")
    if os.path.exists(file_name):
        print(f"Found PCA decomposition weights in {file_name}. If you added new instances after the creation, "
              + "then please delete this file and call <model_inference> again.")
        decomposition = models.load_data(file_name)
    else:
        decomposition = pipeline.run_pipeline(**kwargs)

    model = build_model(decomposition["weights"], decomposition["labels"], classifier=classifier)

    predictions, result, img = recognize_image(cv2.imread(img_path), model, decomposition, scale=scale)
    return predictions, result, img

model_inference("test_images/flower.jpg", "datasets/lfw_10_cropped_09", resolution=(64, 64), scale=0.9)

