import streamlit as st
import pickle
from PIL import Image

def main():
    st.title(":rainbow[DIABATIC FAILURE PREDICTION]")
    image = Image.open('dowload.jpeg')
    st.image(image,width=800)

    pregnancies = st.text_input(":red[pregnanices]","Type_here")

    Glucose = st.text_input(":green[Glucose]","Type_here")

    Bloodpressure = st.text_input(":blue[Bloodpressure]","Type_here")

    SkinThickness = st.text_input(":orange[SkinThickness]","Type_here")

    Insulin = st.text_input("Insulin","Type_here")

    BMI = st.text_input("BMI","Type_here")

    DiabeticsPedigreeFunction =st.text_input("DiabeticsPreigreeFunction","Type_here")

    Age =st.text_input("Age","Type_here")

    features =[pregnancies,Glucose,Bloodpressure,SkinThickness,Insulin,BMI,DiabeticsPedigreeFunction,Age]

    model = pickle.load(open('model.sav','rb'))    #read binary
    scaler =pickle.load(open('scaler.sav','rb'))

    pred = st.button('PREDICT')

    if pred:
        prediction =model.predict(scaler.transform([features]))

        if prediction ==0:
            st.write("not suffering heart disease")

        else:
            st.write("suffering heart disease")

main()

    