#Importing dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf

from utils import load_alignments, num_to_char, load_data, load_data_gif
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')


# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2019/12/ONE-POINT-01-1.png')
    st.title('Whisperly')
    st.write('This application is developed using deep learning model **_LipNet_**.')
    st.info('The feature of uploading random videos to generate text can be added. Will be added soon!..')

st.title('LipReading App') 
# Generating a list of options or videos 

options = os.listdir(os.path.join('.', 'data', 's1'))
selected_video = st.selectbox('Choose a video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.subheader('Converted video to .mp4 format')
        file_path = os.path.join('.','data','s1', selected_video)
        f'ffmpeg -i {file_path} test_video.mp4 -y'
        print('Succesful!')

        # Rendering inside of the app
        video = open('app/test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.subheader("Generate the spoken text")
        button = st.button("Generate",type = 'primary')
        if button:
            # st.info('This is all the machine learning model sees when making a prediction')
            print(file_path)
            video, annotations = load_data_gif(tf.convert_to_tensor(file_path))

            imageio.mimsave('animation.gif', video, fps=10)
            # st.image('animation.gif', width=350) 

            # st.info('This is the output of the machine learning model as tokens')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            # st.text(decoder)

            # Convert prediction to text
            # st.info('Decode the raw tokens into words')

            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)
            st.header('')
            with st.expander("How the deep learning model works?"):
                st.write("**This is what the deep learning model sees when making a prediction**")
                st.image('animation.gif', width=350)
                st.info('The model generates the text based on the generated gif, which will not have any audio')
                st.write('**This is the generated output of the deep learning model as tokens**')
                st.text(decoder)
                st.write('**The final text we get is**')
                st.text(converted_prediction)
