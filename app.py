from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import BaseModel, OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()
OPENAIAPI = os.getenv("OPENAIAPI")

client = OpenAI(api_key=OPENAIAPI)
import base64
import logging

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows any method (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows any header
)


class DeviceData(BaseModel):
    device_model: str
    screen_width: int
    screen_height: int
    density: float

from PIL import Image

# Function to get the max x and y axis limits from an image
def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        # Rotate the image if it is in landscape mode
        if width > height:
            img = img.rotate(90, expand=True)
            width, height = img.size
    return width, height


def encode_image(file: UploadFile):
    image_content = file.file.read()
    return base64.b64encode(image_content).decode("utf-8")

@app.post("/image-completion")
async def image_completion(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    device_data: str = Form(...)
):
    try:
        base64_image = encode_image(file)
        device_data_dict = json.loads(device_data)
        device_info = DeviceData(**device_data_dict)

        MODEL = "gpt-4o-mini"  # Ensure the model name is correctly set

        read_example_image_1 = open("assets/example_twitter.jpeg", "rb")
        example_image_base64_1 = base64.b64encode(read_example_image_1.read()).decode("utf-8")

        read_example_image_2 = open("assets/example_wifi.jpeg", "rb")
        example_image_base64_2 = base64.b64encode(read_example_image_2.read()).decode("utf-8")

        read_example_image_3 = open("assets/example_spotify.jpeg", "rb")
        example_image_base64_3 = base64.b64encode(read_example_image_3.read()).decode("utf-8")

        read_example_image_4 = open("assets/example_aeroplane.jpeg", "rb")
        example_image_base64_4 = base64.b64encode(read_example_image_4.read()).decode("utf-8")

        read_example_image_5 = open("assets/example_spotisearch.jpeg", "rb")
        example_image_base64_5 = base64.b64encode(read_example_image_5.read()).decode("utf-8")
        width, height = get_image_dimensions(file.file)

        SYSTEM_PROMPT = f"""
        
        You are a helpful assistant designed to provide coordinates for user actions on a mobile screen based on given images. The coordinate system starts from the top-left corner of the screen, where (0,0) represents the top-leftmost point. Respond in the following JSON format:

        If the requested action is not needed (e.g., WiFi is already off), set "show_arrow" to false and provide an appropriate message in the "status" field. Otherwise, set "show_arrow" to true and provide the coordinates for the action.

        
        {{
          "question_or_request_from_user": "{prompt}",
          "immediate_action_coordinate": {{
            "x_axis": "x_coordinate",
            "y_axis": "y_coordinate"
          }},
          "show_arrow": boolean,
          "status": "Message indicating if WiFi is already off/on or any other relevant status"
          "rotation": which rotation to perform"
        }}
        
        Analyze the provided screenshot and user request to determine the appropriate coordinates for the action and the status.

        Follow this thinking logic:

        Split the image into 4 quadrants.
        Determine in which quadrant the app or button lies.
        Given an arrow which is by default pointing upwards, determine how much you would move it in x and y axis with one of the following rotations: right, up, left, upsidedown,bottomRight,bottomLeft,upRight,upLeft.
        The x and y axis's (0,0) is from the top left of the screenshot.
        
        Device Info: {device_info.dict()}
        
        The width of the image is {width} and the height of the image is {height}.
        Following are some examples to help you understand the task better:
        """

        response = client.chat.completions.create(model=MODEL,
        messages=[
            {"role": "user", "content": [SYSTEM_PROMPT,

                {
                    "type": "text",
                    "text": """User request How to post
                    "Your response should be like this"
                    "question_or_request_from_user": "How to post",
                    "show_arrow": true,
                    "status": "Click that icon",
                    "rotation": "right"
                    "immediate_action_coordinate": 
                    {{
                      "x_axis": 970,
                      "y_axis": 2000
                    }}"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{example_image_base64_2}"
                    },
                },{
                    "type": "text",
                    "text": """User request Turn off WiFi
                    "Your response should be like this"
                    "question_or_request_from_user": "Turn off WiFi",
                    "show_arrow": true,
                    "status": "Tap the WiFi icon to turn it off",
                    "rotation": "up"
                    "immediate_action_coordinate": 
                    {{
                      "x_axis": 720,
                      "y_axis": 300
                    }}
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{example_image_base64_1}"
                    },
                },{
                    "type": "text",
                    "text": """User request open Spotify how
                    "Your response should be like this"
                    "question_or_request_from_user": "open Spotify how",
                    "show_arrow": true,
                    "status": "Tap the Spotify icon to open it",
                    "rotation": "right"
                    "immediate_action_coordinate": 
                    {{
                      "x_axis": 880,
                      "y_axis": 1880
                    }}
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{example_image_base64_3}"
                    }
                },{
                    "type": "text",
                    "text": """User request How to turn on aeroplane mode
                    "Your response should be like this"
                    "question_or_request_from_user": "How to turn on aeroplane mode",
                    "show_arrow": true,
                    "status": "Tap the aeroplane icon to turn it on",
                    "rotation": "upleft"
                    "immediate_action_coordinate": 
                    {{
                       "x_axis": "250",
                        "y_axis": "1000"
                    }}
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{example_image_base64_4}"
                    }
                },{
                    "type": "text",
                    "text": """User request How to search here
                    "Your response should be like this"
                    "question_or_request_from_user": "How to search here",
                    "show_arrow": true,
                    "status": "Tap the search icon to search",
                    "rotation": "upsidedown"
                    "immediate_action_coordinate": 
                    {{
                       "x_axis": "24",
                        "y_axis": "1500"
                    }}
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{example_image_base64_5}"
                    }
                },
                {
                    "type": "text",
                    "text": """User request Where user page
                    "Your response should be like this"
                    "question_or_request_from_user": "Where user page",
                    "show_arrow": true,
                    "status": "Tap the user icon to go to the user page",
                    "rotation": "upleft"
                    "immediate_action_coordinate": 
                    {{
                       "x_axis": "24",
                        "y_axis": "300"
                    }}
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{example_image_base64_5}"
                    }
                },
                {"type": "text", "text": f"""Starts here the User request: {prompt}"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ]}
        ],
        temperature=0.0)
        return {"response": response.choices[0].message.content}
    except Exception as e:
        logging.error(f"Error in image completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=0000)
