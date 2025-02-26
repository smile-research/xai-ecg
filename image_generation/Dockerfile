FROM python:3.10.1
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
WORKDIR /challenge
COPY ./reqs.txt /challenge/reqs.txt
RUN pip install -r reqs.txt
COPY ./ /challenge


## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
WORKDIR /challenge/ecg-image-generator
ENTRYPOINT [ "python", "gen_ecg_image_from_data.py" ] 