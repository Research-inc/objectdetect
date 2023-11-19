from ultralytics import YOLO
from PIL import Image

def detectWithSearchName(input_image, search_name="book"):
    # Load a pretrained YOLOv8n-seg Segment model
    model = YOLO('yolov8n.pt')

    search_ind =  list(model.names.keys())[list(model.names.values()).index(search_name)]

    # Run inference on an image
    results = model(input_image, classes=search_ind)

    if len(results[0].boxes.cls) == 0:
        return None

    # We extract the detected object's x,y coordinates and bounded box width and height
    x0  = results[0].boxes.xywh.numpy()[0][0]
    y0  = results[0].boxes.xywh.numpy()[0][1]
    w   = results[0].boxes.xywh.numpy()[0][2]
    h = results[0].boxes.xywh.numpy()[0][3]

    # In the next step we find the mid point of detected object with respect to the overall image
    objx = x0 + (w / 2)
    objy = y0 + (h / 2)

    # We determine the width and height of the original image
    orig_height = results[0].boxes.orig_shape[0]
    orig_width = results[0].boxes.orig_shape[1]

    # We determine center of the original image. We use this center point as the origin for the cartesian coordinate system
    origin_x = orig_width/2
    origin_y = orig_height/2

    # We use the origin_x and origin_y along with objx, objy to determine of the location of object across four quadrants.
    quad = None

    if origin_x <= objx and origin_y >= objy:
        quad = "first"

    elif origin_x < objx and origin_y < objy:
        quad = "second"

    elif origin_x > objx and origin_y < objy:
        quad = "third"

    elif origin_x > objx and origin_y > objy:
        quad = "fourth"

    # Process results list
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('result.jpg')  # save image

    return quad


def isObjectPresent(input_image, search_name="book"):
    # Load a pretrained YOLOv8n-seg Segment model
    model = YOLO('yolov8n.pt')
    w = 0 
    h = 0

    search_ind =  list(model.names.keys())[list(model.names.values()).index(search_name)]

    # Run inference on an image
    results = model(input_image, classes=search_ind)

    if len(results[0].boxes.cls) == 0:
        return False
    return True

def detectWithSearchName_v2(input_image, search_name="book"):
    # Load a pretrained YOLOv8n-seg Segment model
    model = YOLO('yolov8n.pt')
    w = 0 
    h = 0

    search_ind =  list(model.names.keys())[list(model.names.values()).index(search_name)]

    # Run inference on an image
    results = model(input_image, classes=search_ind)

    if len(results[0].boxes.cls) == 0:
        return None

    # We extract the detected object's x,y coordinates and bounded box width and height
    x0  = results[0].boxes.xywh.cpu().numpy()[0][0]
    y0  = results[0].boxes.xywh.cpu().numpy()[0][1]
    w   = results[0].boxes.xywh.cpu().numpy()[0][2]
    h = results[0].boxes.xywh.cpu().numpy()[0][3]

    # In the next step we find the mid point of detected object with respect to the overall image
    objx = x0 + (w / 2)
    objy = y0 + (h / 2)

    # We determine the width and height of the original image
    orig_height = results[0].boxes.orig_shape[0]
    orig_width = results[0].boxes.orig_shape[1]

    # We determine center of the original image. We use this center point as the origin for the cartesian coordinate system
    origin_x = orig_width/2
    origin_y = orig_height/2

    # We use the origin_x and origin_y along with objx, objy to determine of the location of object across four quadrants.
    quad = "first"

    if origin_x <= objx and origin_y >= objy:
        quad = "first"

    elif origin_x < objx and origin_y < objy:
        quad = "second"

    elif origin_x > objx and origin_y < objy:
        quad = "third"

    elif origin_x > objx and origin_y > objy:
        quad = "fourth"

    # Process results list
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('result.jpg')  # save image

    if w==None:
        w = 0

    if h== None:
        h = 0

    return quad, w, h

def magnifier(orig_height, new_height):

    return new_height/orig_height


def closenessHelper(magnification_val, quad):

    direction = None
    closeness = None
    if quad == "first" or quad == "second":
        direction = "right"
    else:
        direction = "left"

    if magnification_val > 1.0:
        closeness = "close"
    else:
        closeness = "far"

    print("Object is", closeness, ". But on the", direction, ".")