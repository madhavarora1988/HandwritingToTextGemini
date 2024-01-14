from PIL import Image
import os
import base64
from io import BytesIO
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part


def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(folder_path, filename))
            images.append((filename, img))
    return images


def convert_to_base64(images):
    base64_images = []
    for filename, img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")  # Change format to "JPEG" if needed
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        base64_images.append({'name': filename, 'base64': img_base64})
    return base64_images


def generate(image):
    combined_response = ""
    image_decoded = Part.from_data(data=base64.b64decode(image), mime_type="image/jpeg")
    model = GenerativeModel("gemini-pro-vision")
    responses = model.generate_content(
        [image_decoded, """Read the text in this image."""],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32
        },
        stream=True,
    )

    for response in responses:
        # print(response.text, end="")
        combined_response += response.text
    return combined_response


def save_text_to_files(text_for_images, predefined_folder):
    """
    Save Base64 representations of images to text files.

    Parameters:
    base64_images (list of dict): List containing dictionaries with image names and Base64 strings.
    predefined_folder (str): Path to the folder where text files should be saved.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(predefined_folder):
        os.makedirs(predefined_folder)

    for item in text_for_images:
        # Extract the image name (without extension) and Base64 string
        image_name = os.path.splitext(item['name'])[0]
        base64_string = item['text']

        # Construct the full path for the text file
        file_path = os.path.join(predefined_folder, image_name + '.txt')

        # Write the Base64 string to the file
        with open(file_path, 'w') as file:
            file.write(base64_string)

    print("handwritten text has been saved to text files.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    vertexai.init(project="project-name")
    input_folder_path = 'path_to_images_with_handwritten_text'
    output_folder_path = 'path_where_text_files_will_be_generated'

    images = read_images_from_folder(input_folder_path)
    base64_images = convert_to_base64(images)

    text_for_images = []
    for image in base64_images:
        print(image['name']);
        # print(image['base64']);
        response = generate(image['base64'])
        text_for_images.append({'name': image['name'], 'text': response})

    save_text_to_files(text_for_images, output_folder_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
