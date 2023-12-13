# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
#from google.colab import files
import random
from math import exp
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

CANVAS_SIZES = {
    'A0': (9933, 14043),
    'A1': (7016, 9933),
    'A2': (4960, 7016),
    'A3': (3508, 4960),
    'A4': (2480, 3508),
    'A5': (1748, 2480),
    'A6': (1240, 1748),
    'A7': (874, 1240),
    'A8': (620, 874),
    'A9': (437, 620),
    'A10': (310, 437),
    # Additional size for square meter
    '1m2': (11811, 11811)  # Square meter, assuming 300 DPI and 1 meter = ~39.37 inches
}

def remove_padding(image):
    """
    Remove the padding added around the image.

    Parameters:
    - image: Input image with padding.

    Returns:
    - Image with padding removed.
    """
    height, width = image.shape[:2]

    new_height = height // 4  # Because 3 times padding + 1 original = 4
    new_width = width // 4

    top = new_height
    left = new_width

    return image[top:top + new_height, left:left + new_width]

def enlarge_with_large_frame(image):
    """
    Enlarge the image with a white frame that's 3 times the image size.
    """
    # Calculate padding
    pad_left = image.shape[1] * 2
    pad_right = image.shape[1] * 2
    pad_top = image.shape[0] * 2
    pad_bottom = image.shape[0] * 2

    # Add padding around the image
    enlarged_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return enlarged_image

class Shape:
    def __init__(self, image):
        """
        Initialize the Shape object.

        Parameters:
        - image (numpy.array): The associated image.
        """
        self.image = image
        self.original_contour = self.extract_contour()
        self.contour = self.original_contour.copy() if self.original_contour is not None else None
        self.hull = None
        if self.contour is not None and len(self.contour) > 0:
            self.hull = self.compute_convex_hull()
        self.centroid = self.compute_centroid()
        self.n_i = 0

    def compute_convex_hull(self):
        """Compute and return the convex hull for the shape."""
        if self.contour is not None and len(self.contour) > 0:
            return cv2.convexHull(self.contour)
        if self.contour is None:
            print("Warning: Cannot compute convex hull without a valid contour.")
            return None
        return cv2.convexHull(self.contour)

    
    def enlarge_with_large_frame(self, image):
        """
        Enlarge the image with a white frame so that the resulting image's
        size is 2 times the maximum of the image's height and width.
        """
        # Calculate desired output dimension
        desired_dim = 2 * max(image.shape[0], image.shape[1])

        # Calculate padding for each side
        pad_top = (desired_dim - image.shape[0]) // 2
        pad_bottom = desired_dim - image.shape[0] - pad_top
        pad_left = (desired_dim - image.shape[1]) // 2
        pad_right = desired_dim - image.shape[1] - pad_left

        # Add padding around the image
        enlarged_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return enlarged_image


    def extract_contour(self):
        """Extract the primary contour from the associated image."""
        # Check if image is RGB or RGBA, and convert to grayscale if needed
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grayscale, 250, 255, cv2.THRESH_BINARY_INV)

        # Erode and dilate to handle outliers
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if contours list is empty
        if len(contours) == 0:
            print("Warning: No contour found for the shape.")
            total_pixel_value = np.sum(eroded)
            #if total_pixel_value == 0:
            #    print("The kernel image is all black.")
            #elif total_pixel_value == dilated.shape[0] * dilated.shape[1] * 255:
            #    print("The kernel image is all white.")
            #else:
            #    print(f"Total pixel value of the dilated image: {total_pixel_value}")
            return None

        # Combine all contour points to form a single contour
        all_contour_points = np.concatenate(contours, axis=0)
        combined_hull = cv2.convexHull(all_contour_points)

        return combined_hull


    def compute_convex_hull(self):
        """Compute and return the convex hull for the shape."""
        return cv2.convexHull(self.contour)

    def compute_centroid(self):
        """Compute and return the centroid of the shape."""
        M = cv2.moments(self.contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        return None

    def rotate_new(self, angle):

        # Enlarge image
        self.image = enlarge_with_large_frame(self.image)
        # Center of the enlarged image
        center_x = self.image.shape[1] // 2
        center_y = self.image.shape[0] // 2

        # Rotate the image
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotated_image = cv2.warpAffine(self.image, rot_mat, self.image.shape[1::-1], flags=cv2.INTER_CUBIC)


        # Mask out unwanted black areas
        _, mask = cv2.threshold(self.image, 1, 255, cv2.THRESH_BINARY)
        rotated_mask = cv2.warpAffine(mask, rot_mat, mask.shape[1::-1], flags=cv2.INTER_LINEAR)
        _, rotated_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(rotated_mask, kernel, iterations=1)
        rotated_image[dilated_mask == 0] = 255

    def rotate(self, angle):
        # Create a copy of the image to work on
        work_image = self.enlarge_with_large_frame(self.image.copy())

        # Center of the enlarged image
        center_x = work_image.shape[1] // 2
        center_y = work_image.shape[0] // 2

        # Rotate the image
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotated_image = cv2.warpAffine(work_image, rot_mat, work_image.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Compute the mask of non-white areas (assumes the background is white)
        gray_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_rotated, 250, 255, cv2.THRESH_BINARY_INV)

        # Find the bounding rectangle of the non-white areas
        x, y, w, h = cv2.boundingRect(mask)

        # Crop the rotated image to this bounding rectangle
        cropped_rotated_image = rotated_image[y:y+h, x:x+w]
        new_shape = Shape(cropped_rotated_image)
        new_shape.n_i = self.n_i  # Carry over the value from the original shape
        return new_shape

    def crop(self):
        """Crop and return the image with a transparent background outside the contour."""
        # Create an all-black image with the same dimensions as the original
        #search
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        # Fill the detected contour with white on the black mask
        # Dilate the grayscale mask
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # Add an alpha channel. Set it to 1 (fully opaque) where the mask is white and 0 (fully transparent) elsewhere
        alpha_channel = np.where(dilated == 255, 255, 0).astype(np.uint8)

        if len(self.image.shape) == 2 or self.image.shape[2] == 1:  # Grayscale
            rgba_image = cv2.merge([self.image, self.image, self.image, alpha_channel])
        elif self.image.shape[2] == 3:  # RGB
            rgba_image = cv2.merge([self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2], alpha_channel])
        elif self.image.shape[2] == 4:  # RGBA
            rgba_image = self.image.copy()
            rgba_image[:, :, 3] = alpha_channel

        x, y, w, h = cv2.boundingRect(self.contour)
        cropped_image = rgba_image[y:y+h, x:x+w]
        if self.contour is None:
            print("Warning: Cannot crop the shape without a valid contour.")
            print(f" explained {self}")
            return self.image
        return Shape(cropped_image)

    def highlight_contour_with_hull(self):
        """Highlight and return the image with its detected contour and convex hull drawn."""

        # Extract primary contour
        primary_contour = self.extract_contour()

        # Compute convex hull for the contour
        convex_hull = cv2.convexHull(primary_contour)

        # Clone the original image
        highlighted_image = self.image.copy()

        # If it's a grayscale image, convert it to color (RGB)
        if len(highlighted_image.shape) == 2 or highlighted_image.shape[2] == 1:
            highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_GRAY2RGB)

        # Draw the contour on the image using green color
        cv2.drawContours(highlighted_image, [primary_contour], -1, (255, 0, 0), thickness=2)

        # Draw the convex hull on the image using red color
        cv2.drawContours(highlighted_image, [convex_hull], -1, (0, 0, 255), thickness=2)

        # Create an image with white background
        white_background = np.ones_like(highlighted_image, dtype=np.uint8) * 255

        # Draw the contour on the white background using green color
        cv2.drawContours(white_background, [primary_contour], -1, (255, 0, 0), thickness=2)

        # Draw the convex hull on the white background using red color
        cv2.drawContours(white_background, [convex_hull], -1, (0, 0, 255), thickness=2)

        return [Shape(highlighted_image), Shape(white_background)]

    def copy(self):
            """Returns a copy of the Shape object."""
            new_shape = Shape(self.image.copy())
            new_shape.original_contour = self.original_contour.copy()
            new_shape.contour = self.contour.copy()
            new_shape.hull = self.hull.copy()
            new_shape.n_i = self.n_i
            if self.centroid:
                new_shape.centroid = (self.centroid[0], self.centroid[1])
            return new_shape

    def reset(self):
        """Reset the shape to its original contour."""
        self.contour = self.original_contour.copy()
        self.hull = self.compute_convex_hull()
        self.centroid = self.compute_centroid()

    def set_image(self, image):
        """Set a new associated image for the shape."""
        self.image = image

    def get_image(self):
        """Retrieve the associated image of the shape."""
        return self.image

    def display(self):
      plt.imshow(self.get_image())
      plt.title(self)
      plt.show()

class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas
        self.mask = np.zeros((height, width), dtype=np.uint8)  # Black mask to start with
        self.ri_values = {}  # Dictionary to store r_i values
        self.available_area = 99.9
        self.shapes = []

    def compute_ri(self, shapes):
            # Calculate total area (sum of all area_of_shape_i * n_i)
        
            total_area = sum([shape.crop().image.size * shape.n_i for shape in shapes])
            # Calculate r_i for each shape and store in the ri_values dictionary
            for shape in shapes:
                ri = (shape.crop().image.size * shape.n_i) / total_area
                self.ri_values[shape] = ri
        
    def add_shape(self, image):
        shape_obj = Shape(image)
        # you can set the n_i value here if needed
        shape_obj.n_i = 1
        self.shapes.append(shape_obj)

    def copy(self):
        """Return a copy of the canvas."""
        new_canvas = Canvas(self.width, self.height)
        new_canvas.canvas = self.canvas.copy()
        new_canvas.mask = self.mask.copy()
        return new_canvas

    def show_ris(self):
        print(f"  ris inside of fill with image: { self.ri_values }")

    def wasted_space(self):
        # Convert canvas to grayscale for easy thresholding
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        # Threshold the grayscale image to get a binary mask of white areas
        _, white_mask = cv2.threshold(gray_canvas, 250, 255, cv2.THRESH_BINARY)
        # Calculate the percentage of white pixels
        total_pixels = self.width * self.height
        white_pixels = np.sum(white_mask == 255)
        white_percentage = (white_pixels / total_pixels) * 100
        return float(white_percentage)

    def place_image(self, image, x, y):
        h, w = image.shape[:2]
        end_y = min(y + h, self.canvas.shape[0])
        end_x = min(x + w, self.canvas.shape[1])

        if image.shape[2] == 4:  # If the image is RGBA
            image = image[:, :, :3]  # Convert to RGB by slicing away the alpha channel

        self.canvas[y:end_y, x:end_x] = image[:end_y-y, :end_x-x]
        self.mask[y:end_y, x:end_x] = 255

    def test_fill_rotation(self, image):
        img_h, img_w = image.shape[:2]

        # Calculate the rotation incrementation
        rot_incrementation = 90 // max(img_w / img_h, img_h / img_w)

        best_waste = 100  # start with a high value for comparison
        best_angle = 0  # starting angle

        for angle in range(0, 90, int(rot_incrementation)):
            # Make a copy of the canvas for individual rotation testing
            temp_canvas = Canvas(self.width, self.height)

            # Rotate the image by the current angle
            rot_mat = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle, 1.0)
            rotated_img = cv2.warpAffine(image, rot_mat, (img_w, img_h), flags=cv2.INTER_LINEAR)
            # Use the fill_with_image function on the rotated image
            # Calculate wasted space
            waste_percentage = self.wasted_space()
            # Update the best angle if this rotation has less waste
            if waste_percentage < best_waste:
                best_waste = waste_percentage
                best_angle = angle

        return best_angle

    def fill_with_shapes(self, shapes):
        for idx, shape in enumerate(shapes):
            self.fill_with_image(shape.crop().get_image())


    def fill_with_shape(self, shape,flag, limit):
          if border_bool == 0:
              pass
          elif border_bool == 1:
              shape = shape.highlight_contour_with_hull()[0]
          elif border_bool == 2:
              shape = shape.highlight_contour_with_hull()[1]
          self.fill_with_image(shape.crop().get_image(), flag, limit)


    def display(self, save_path="output_filename_pil.png"):
        pil_image = Image.fromarray(self.canvas.astype('uint8'))
        pil_image.save(save_path)
        return save_path

    def best_fit_placement(self, shape):
        best_fit = None
        best_rotation = None
        optimal_rotated_shape = None
        minimum_waste = float(100)

        # Determine the rotation incrementation based on the aspect ratio
        aspect_ratio = shape.image.shape[1] / shape.image.shape[0]  # width / height
        angle_incrementation = max(10, int(90 // max(aspect_ratio, 1 / aspect_ratio)))  # At least 10 degrees

        rotations = list(range(0, 360, angle_incrementation))

        # Define step sizes based on a fraction of the image size.
        step_x = max(1, shape.image.shape[1] // 10)  # Using 1/10th of the image width as an example
        step_y = max(1, shape.image.shape[0] // 10)  # Using 1/10th of the image height

        for angle in rotations:
            rotated_shape = shape.rotate(angle)
            cropped_rotated_shape = rotated_shape.crop()

            for y in range(0, self.height - cropped_rotated_shape.image.shape[0] + 1, step_y):
                for x in range(0, self.width - cropped_rotated_shape.image.shape[1] + 1, step_x):
                    # Bounding box check
                    if np.all(self.mask[y:y+cropped_rotated_shape.image.shape[0], x:x+cropped_rotated_shape.image.shape[1]] == 0):
                        canvas_copy = self.copy()
                        canvas_copy.place_image(cropped_rotated_shape.get_image(), x, y)
                        waste = canvas_copy.wasted_space()

                        if waste < minimum_waste:
                            best_fit = (x, y)
                            best_rotation = angle
                            optimal_rotated_shape = cropped_rotated_shape
                            minimum_waste = waste

        if best_fit:
            self.place_image(optimal_rotated_shape.get_image(), best_fit[0], best_fit[1])

        return self

    def rank_and_fill_shapes(self, shapes):
        """Rank shapes by size and fill the canvas."""
        # Rank shapes by size
        ranked_shapes = sorted(shapes, key=lambda s: s.image.size, reverse=True)

        # Place each shape in sequence
        for shape in ranked_shapes:
            self.best_fit_placement(shape)
        return self

    def unified_fill_method(self, shapes):
        """Unified method to handle placement based on the number of shapes."""
        if len(shapes) == 1:
            return self.best_fit_placement(shapes[0])
        else:
            return self.rank_and_fill_shapes(shapes)

    def fill_with_image(self, image, flag, limit):
        img_h, img_w = image.shape[:2]
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        r_img_h, r_img_w = rotated_img.shape[:2]
        mask = self.mask.copy()

        def can_place(image, top_left_x, top_left_y):
            if top_left_x + image.shape[1] > mask.shape[1] or top_left_y + image.shape[0] > mask.shape[0]:
                return False
            region = mask[top_left_y:top_left_y + image.shape[0], top_left_x:top_left_x + image.shape[1]]
            return np.all(region == 0)

        MAX_ITERATIONS = 10
        iteration = 0
        while iteration < MAX_ITERATIONS:
            y = 0
            while y < mask.shape[0]:
                x = 0
                while x < mask.shape[1]:
                    if can_place(image, x, y) : #and self.wasted_space() > self.available_area * 0.9
                        self.place_image(image, x, y)
                        mask[y:y+img_h, x:x+img_w] = 255
                        x += img_w
                        if(flag and limit == None):
                            continue
                        elif flag and (100-(self.wasted_space()) > limit): #self.available_area
                            return
                    elif can_place(rotated_img, x, y) :
                        self.place_image(rotated_img, x, y)
                        mask[y:y+r_img_h, x:x+r_img_w] = 255
                        x += r_img_w
                    else:
                        x += 50
                y += 50
            iteration += 1

    def place_optimally(self, shapes):
        shapes_sorted = sorted(shapes, key=lambda s: s.image.size, reverse=True)
        flag = False
        values_list = list(self.ri_values.values())
        print(f" the proportions will be : {values_list}")
        limit = 0
        i=0
        for shape in shapes_sorted:
            limit += 100*values_list[i]
            i+=1
            best_waste = 100
            best_rotated_shape = shape
            if shape == shapes[-1]:
                self.available_area = 100
                limit = None
            aspect_ratio = shape.image.shape[1] / shape.image.shape[0]
            rot_incrementation = int(90 // max(aspect_ratio, 1 / aspect_ratio))
            rotations = list(range(0, 360, rot_incrementation))
            for angle in rotations:
                rotated_shape = shape.rotate(angle)
                temp_canvas = self.copy()
                temp_canvas.ri_values = self.ri_values
                flag = False

                temp_canvas.fill_with_shape(rotated_shape, flag, 100)
                waste = temp_canvas.wasted_space()

                if waste < best_waste:
                    best_waste = waste
                    best_rotated_shape = rotated_shape
            if best_rotated_shape is None:
                print("No suitable rotation found for the shape!")
                continue
            flag = True
            self.fill_with_shape(best_rotated_shape,flag, limit)
            flag = False
    def load_shapes_from_images(self, images, ni_values=None):
        if not images:
            print("No images provided.")
            return []

        shapes = []
        default_ni_values = [5, 20]  # Default list of n_i values. Adjust as needed.

        if not ni_values:
            ni_values = default_ni_values

        for idx, img in enumerate(images):
            if len(img.shape) == 3 and img.shape[2] == 4:  # If image has an alpha channel
                # Convert alpha image to 3 channel
                img = img[:, :, :3]  # Retain only the RGB channels, drop the alpha

            # Convert BGR to RGB before displaying with matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Create Shape object and assign n_i value
            shape_obj = Shape(img_rgb)
            if idx < len(ni_values):
                shape_obj.n_i = ni_values[idx]
            else:
                shape_obj.n_i = 0  # Default value for n_i if not provided in ni_values
            shapes.append(shape_obj)

        return shapes

    def load_shapes(self):
        # Filter for image and PDF files
        extensions = ['.png', '.pdf', '.jpeg', '.jpg']
        image_paths = [f for f in os.listdir() if any(f.endswith(ext) for ext in extensions)]

        if not image_paths:
            print("No images or PDFs found with the given extensions.")
            return
        shapes = []
        ni_values = [5, 20]  # List of n_i values. Adjust as needed.

        for idx, image_path in enumerate(image_paths):
            if image_path.endswith('.pdf'):
                # Convert the PDF to a PNG
                pages = convert_from_path(image_path)
                if pages:
                    img_path_temp = "temp_image.png"
                    pages[0].save(img_path_temp, 'PNG')  # Use first page if the PDF has multiple pages
                    img = cv2.imread(img_path_temp, cv2.IMREAD_UNCHANGED)
                    os.remove(img_path_temp)  # Clean up the temporary PNG image
                else:
                    print(f"Could not convert {image_path} to image.")
                    continue
            else:
                # If it's an image, directly read it
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if len(img.shape) == 3 and img.shape[2] == 4:  # If image has an alpha channel
                # Convert alpha image to 3 channel
                img = img[:, :, :3]  # Retain only the RGB channels, drop the alpha

            # Convert BGR to RGB before displaying with matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Create Shape object and assign n_i value
            shape_obj = Shape(img_rgb) # Chat GPT this line has a major issue, how to fix it? it stops the code 
            if idx < len(ni_values):
                shape_obj.n_i = ni_values[idx]
            else:
                shape_obj.n_i = 0  # Default value for n_i if not provided in ni_values
            shapes.append(shape_obj)

        return shapes

    def load_shapes_modified(self ,files):
        shapes = []
        ni_values = [5, 20]  # List of n_i values. Adjust as needed.

        for idx, file in enumerate(files):
            # Check if the file is a path or a file-like object (uploaded file)
            if isinstance(file, str) and file.endswith('.pdf'):
                # Convert the PDF to a PNG
                pages = convert_from_path(file)
                if pages:
                    img_path_temp = "temp_image.png"
                    pages[0].save(img_path_temp, 'PNG')  # Use first page if the PDF has multiple pages
                    img = cv2.imread(img_path_temp, cv2.IMREAD_UNCHANGED)
                    os.remove(img_path_temp)  # Clean up the temporary PNG image
                else:
                    print(f"Could not convert {file} to image.")
                    continue
            elif not isinstance(file, str):  # Assuming it's an uploaded file-like object
                print(f" {file} is a file-like object.")
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            else:  # Directly read the image if it's a file path
                img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

            if len(img.shape) == 3 and img.shape[2] == 4:  # If image has an alpha channel
                # Convert alpha image to 3 channel
                img = img[:, :, :3]  # Retain only the RGB channels, drop the alpha

            # Convert BGR to RGB before displaying with matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Create Shape object and assign n_i value
            shape_obj = Shape(img_rgb)  # This line might still raise an error if the Shape class isn't defined in this context
            if idx < len(ni_values):
                shape_obj.n_i = ni_values[idx]
            else:
                shape_obj.n_i = 0  # Default value for n_i if not provided in ni_values
            shapes.append(shape_obj)

        return shapes

    # NOTE: This modified function assumes the Shape class is defined elsewhere in the code.

    def set_border_option(self, option):
        global border_bool
        if option == "placer":
            border_bool = 0
        elif option == "border":
            border_bool = 1
        elif option == "decouper":
            border_bool = 2

    def main_new_approach(self ,files):
        # Initialize an A4 canvas
        A4_WIDTH, A4_HEIGHT = CANVAS_SIZES['A4']

        # Load shapes
        shapes = self.load_shapes_modified(files)


        if shapes:
            print("it goes here ")
            self.compute_ri(shapes)
        # Use the new strategy to place the shapes optimally
            self.place_optimally(shapes)
        # Display the final filled canvas
            self.display()
        else : 
            print("No shapes loaded.")
        return self.display()
# Test the new approach
border_bool = 0 # when = 0 placer, when =1 border, when = 2 decouper
#canvas_obj = Canvas(CANVAS_SIZES['A4'][0], CANVAS_SIZES['A4'][1])
#canvas_obj.main_new_approach()