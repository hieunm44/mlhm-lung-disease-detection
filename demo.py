import streamlit as st
from PIL import Image
import numpy as np
from shvit import SHVIT
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings("ignore")


@st.cache_resource()
def load_model(image_size, model_path):
    shvit_settings = {
        'embed_dim': [128, 224, 320],
        'depth': [2, 4, 5],
        'partial_dim': [32, 48, 68],
        'types' : ['i', 's', 's']
    }
    shvit = SHVIT(**shvit_settings)
    inp = Input(shape=(image_size, image_size, 1))
    outp = shvit(inp)
    shvit_64 = Model(inputs=inp, outputs=outp)
    shvit_64.load_weights(model_path)

    return shvit_64


if __name__ == '__main__':
    st.set_page_config(
        page_title="Lung Disease Detection",
        page_icon = ":mango:",
        initial_sidebar_state = 'auto'
    )

    st.title("Lung Disease Detection")
    st.subheader("Detection of lung diseases present in the X-ray image.")
    img_file = st.file_uploader("", type=["jpg", "png"])
    
    image_size_64 = 64
    shvit_64_path = f'./saved_models/shvit_64.weights.h5'

    with st.spinner('Model is being loaded..'):
        model=load_model(image_size_64, shvit_64_path)

    class_names = {'0': 'atelectasis', '1': 'cardiomegaly', '2': 'effusion', '3': 'infiltration', '4': 'mass', '5': 'nodule', '6': 'pneumonia', '7': 'pneumothorax',
                   '8': 'consolidation', '9': 'edema', '10': 'emphysema', '11': 'fibrosis', '12': 'pleural', '13': 'hernia'}

    
    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, use_column_width=True)
        img = np.asarray(image)
        img = img.reshape(1, img.shape[0], img.shape[0], 1)/255
        pred = model.predict(img)[0]
        pred_ids = np.where(pred>0.5)[0]

        if len(pred_ids)==0:
            st.markdown(f'<div style="text-align: center; font-size: 40px">No disease detected</div>', unsafe_allow_html=True)
        else:
            string = 'Detected diseases:'
            for i in pred_ids:
                string += ' ' + class_names[str(i)]
            st.markdown(f'<div style="text-align: center; font-size: 40px">{string}</div>', unsafe_allow_html=True)