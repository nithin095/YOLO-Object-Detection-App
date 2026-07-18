import cv2
import numpy as np
import pandas as pd


def convert_to_rgb(image):
    """
    Convert BGR image to RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_object_counts(results):
    """
    Count detected objects
    """

    counts = {}

    for box in results[0].boxes:

        cls = int(box.cls[0])

        label = results[0].names[cls]

        counts[label] = counts.get(label, 0) + 1

    return counts


def create_dataframe(counts):
    """
    Convert dictionary to DataFrame
    """

    df = pd.DataFrame(
        list(counts.items()),
        columns=["Object", "Count"]
    )

    return df


def draw_title():
    """
    Streamlit Title
    """

    return """
    ## 🎯 YOLOv8 Object Detection Dashboard
    Upload an image, video, or use your webcam.
    """


def total_objects(counts):
    """
    Total detected objects
    """

    return sum(counts.values())


def unique_objects(counts):
    """
    Number of unique classes
    """

    return len(counts)


def save_image(path, image):
    """
    Save image
    """

    cv2.imwrite(path, image)


def resize_image(image, width=800):

    h, w = image.shape[:2]

    ratio = width / w

    height = int(h * ratio)

    image = cv2.resize(image, (width, height))

    return image


def get_classes(results):

    classes = []

    for box in results[0].boxes:

        cls = int(box.cls[0])

        classes.append(results[0].names[cls])

    return classes


def detection_summary(results):

    counts = get_object_counts(results)

    summary = []

    for name, count in counts.items():

        summary.append(
            {
                "Object": name,
                "Count": count
            }
        )

    return pd.DataFrame(summary)