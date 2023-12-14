FROM yaelvinker/clipasso_docker
RUN echo 'alias ll="ls -l"' >> ~/.bashrc
RUN mkdir /home/code
WORKDIR /home/code
RUN pip install cairosvg
RUN apt-get update -y
RUN apt-get install python3-cffi python3-brotli libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 libgtk-3-dev gcc -y
RUN pip3 install -U scikit-learn scipy matplotlib
RUN pip install ipdb
RUN pip install webp
RUN pip install kornia==0.5.0
RUN pip install opencv-python==4.5.4.60 # avoid connected components segfault