# LipReading Using Deep Learning

- A Lipreading app which generates spoken text based on lip movement using deep learning
- Load and convert the video into a GIF using `OpenCv` and `Imageio`, and then to text using `Tensorflow` and creating data pipeline, to take the audio of the video out of the equation
- The GIF is passed as an input to the deep learning model constructed using `3D CNNs` and `Bi-Directional LSTMs`. The model generates number tokens which we convert to characters to get the generated text
- Developed a web application using `Streamlit` where a user can select a video and get the text generated.


## Website
Checkout the web application [here](https://tvr28-lipnet-appstreamlitapp-9ps3vy.streamlit.app/)
