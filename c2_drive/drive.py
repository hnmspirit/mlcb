import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dense, Flatten, Dropout

import utils
from utils import INPUT_SHAPE

sio = socketio.Server()
app = Flask(__name__)

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        throttle = float(data["throttle"])
        steering_angle = float(data["steering_angle"])
        speed = float(data["speed"])

        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)
            image = utils.preprocess(image)
            image = np.array([image])
            print('*****************************************************')
            steering_angle = float(model.predict(image, batch_size=1))

			# Tốc độ ta để trong khoảng từ 10 đến 25
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # giảm tốc độ
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))

            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        # if args.image_folder != '':
        #     timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        #     image_filename = os.path.join(args.image_folder, timestamp)
        #     image.save('{}.jpg'.format(image_filename))
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('--model', type=str, default='model-0007.h5')
    parser.add_argument('--image_folder', type=str, default='')
    args = parser.parse_args()

    # Load model mà ta đã train được từ bước trước
    model = Sequential([
        Lambda(lambda x: x/127.5 - 1, input_shape=INPUT_SHAPE),
        Conv2D(24, 5, 2, activation='elu'),
        Conv2D(36, 5, 2, activation='elu'),
        Conv2D(48, 5, 2, activation='elu'),
        Conv2D(64, 3, 1, activation='elu'),
        Conv2D(64, 3, 1, activation='elu'),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='elu'),
        Dropout(0.5),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(1),
    ])
    model.load_weights(args.model)
    # model = load_model(args.model)

    # if args.image_folder != '':
    #     print("Creating image folder at {}".format(args.image_folder))
    #     if not os.path.exists(args.image_folder):
    #         os.makedirs(args.image_folder)
    #     else:
    #         shutil.rmtree(args.image_folder)
    #         os.makedirs(args.image_folder)
    #     print("RECORDING THIS RUN ...")
    # else:
    #     print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
